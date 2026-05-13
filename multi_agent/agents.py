"""
multi_agent/agents.py — All 7 Specialized Agents.
Each agent is a class with a single run(state) -> state method.
They read/write shared MultiAgentState and emit stream events.
"""
import json
from .state import (
    MultiAgentState, ReflectionEntry,
    NODE_MEMORY, NODE_REASONING, NODE_TOOL, NODE_ML,
    NODE_FUSION, NODE_REFLECTION, NODE_REPORT, NODE_WAIT, NODE_END
)
from .memory import HybridMemory
from rag_engine import call_llm
from tools import TOOL_REGISTRY, TOOL_DESCRIPTIONS, execute_tool
from database import get_patient_clinical_data, get_patient_ml_prediction, supabase
from questionnaire import (
    calculate_section_scores,
    ml_neuropathy_prediction,
    final_decision as compute_final_decision
)


# ══════════════════════════════════════════════════════════════════
# 1. PLANNER AGENT
# ══════════════════════════════════════════════════════════════════
class PlannerAgent:
    name = "PlannerAgent"

    def run(self, state: MultiAgentState) -> MultiAgentState:
        state.log(self.name, "Planning diagnostic session")
        state.emit("agent_start", "Generating diagnostic plan...", self.name)

        has_nss = bool(state.clinical_scores.get("nss_score"))
        has_nds = bool(state.clinical_scores.get("nds_score"))
        has_ml  = state.has_ml_data()
        has_rag = bool(state.long_term)

        prompt = f"""You are a Medical Diagnostic Planner.
Patient: {state.patient_info.get('name')}, Age: {state.patient_info.get('age')}
Existing data: NSS_stored={has_nss}, NDS_stored={has_nds}, ML_stored={has_ml}, RAG_loaded={has_rag}

Generate a concise 4-6 step diagnostic plan to reach a confident diagnosis.
Respond ONLY as a JSON array of strings.
Example: ["Load patient history from RAG memory", "Ask NSS symptom questions", ...]"""

        raw = call_llm(prompt)
        try:
            cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            plan = json.loads(cleaned)
            if isinstance(plan, list):
                state.plan = plan
                state.plan_version += 1
        except Exception:
            state.plan = [
                "Retrieve patient history from RAG memory",
                "Collect NSS/NDS symptoms from patient",
                "Run ML neuropathy prediction",
                "Calculate weighted fusion score",
                "Apply self-reflection check",
                "Generate final structured report"
            ]

        state.emit("plan", "\n".join([f"{i+1}. {s}" for i, s in enumerate(state.plan)]), self.name)
        state.log(self.name, "Plan generated", {"plan": state.plan, "version": state.plan_version})
        state.next_node = NODE_MEMORY
        return state

    def replan(self, state: MultiAgentState, reason: str) -> MultiAgentState:
        """Dynamic re-planning mid-execution."""
        state.emit("replan", f"Re-planning because: {reason}", self.name)

        prompt = f"""You are a Medical Diagnostic Planner. The current plan needs updating.

Reason: {reason}
Current state: {state.to_summary()}
Answers collected: {len(state.answers)}

Generate a revised 3-5 step plan to complete the diagnosis.
Respond ONLY as a JSON array of strings."""

        raw = call_llm(prompt)
        try:
            cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            plan = json.loads(cleaned)
            if isinstance(plan, list):
                state.plan = plan
                state.plan_version += 1
        except Exception:
            pass

        state.log(self.name, "Re-plan completed", {"reason": reason, "new_plan": state.plan})
        return state


# ══════════════════════════════════════════════════════════════════
# 2. MEMORY / RAG AGENT
# ══════════════════════════════════════════════════════════════════
class MemoryRAGAgent:
    name = "MemoryRAGAgent"

    def run(self, state: MultiAgentState, memory: HybridMemory) -> MultiAgentState:
        state.log(self.name, "Loading memory layers")
        state.emit("agent_start", "Searching long-term memory...", self.name)

        # Long-term vector search
        queries = [
            "neuropathy symptoms pain burning feet",
            f"{state.patient_info.get('name')} diagnosis history"
        ]
        all_results = []
        for q in queries:
            results = memory.search_long_term(q, limit=3)
            all_results.extend(results)

        # Deduplicate by content
        seen = set()
        unique = []
        for r in all_results:
            key = r.get("content", "")[:80]
            if key not in seen:
                seen.add(key)
                unique.append(r)

        state.long_term = unique
        state.emit("memory", f"Retrieved {len(unique)} relevant records from long-term memory", self.name)

        # Episodic memory (past decisions)
        episodic = memory.load_episodic()
        state.episodic = episodic
        if episodic:
            state.emit("memory", f"Loaded {len(episodic)} past diagnostic sessions", self.name)

        state.log(self.name, "Memory loaded", {"rag_count": len(unique), "episodic_count": len(episodic)})
        state.next_node = NODE_REASONING
        return state


# ══════════════════════════════════════════════════════════════════
# 3. CLINICAL REASONING AGENT
# ══════════════════════════════════════════════════════════════════
class ClinicalReasoningAgent:
    name = "ClinicalReasoningAgent"

    SYSTEM = """You are a Clinical Reasoning Agent. You operate in a ReAct loop.
Your ONLY job: decide the next diagnostic action based on current state.

{tool_descriptions}

Respond ONLY in strict JSON:
{{
  "thought": "Internal clinical reasoning (NOT shown to patient)",
  "action": {{
    "type": "tool",
    "name": "tool_name",
    "args": {{}}
  }},
  "message": "Optional patient-facing message",
  "confidence": 0.0,
  "continue": true
}}

When FULL data is collected (ML + fusion + enough answers), use:
{{
  "thought": "Data sufficient. Proceeding to ML inference.",
  "action": {{ "type": "route", "name": "ml_node", "args": {{}} }},
  "confidence": 0.7,
  "continue": true
}}

When FUSION is done and confidence is high:
{{
  "thought": "Fusion complete. Generating final report.",
  "action": {{ "type": "route", "name": "report_node", "args": {{}} }},
  "confidence": 0.95,
  "continue": false
}}"""

    def run(self, state: MultiAgentState) -> MultiAgentState:
        state.log(self.name, "Reasoning step", {"iteration": state.iteration})
        state.emit("agent_start", f"Clinical reasoning (iteration {state.iteration})...", self.name)

        # Build RAG context
        rag_ctx = ""
        if state.long_term:
            rag_ctx = "\nRAG Memory:\n" + "\n".join([
                f"  - {r.get('content','')[:120]}" for r in state.long_term[:3]
            ])

        # Build prompt
        system = self.SYSTEM.format(tool_descriptions=TOOL_DESCRIPTIONS)
        user_prompt = f"""=== CURRENT STATE ===
{state.to_summary()}
{rag_ctx}

=== PLAN ===
{chr(10).join([f'{i+1}. {s}' for i, s in enumerate(state.plan)])}

Decide your next action to advance the diagnosis."""

        raw = call_llm(system + "\n\n" + user_prompt)

        # Parse response
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:]
            if cleaned.startswith("```"): cleaned = cleaned[3:]
            if cleaned.endswith("```"): cleaned = cleaned[:-3]
            llm_out = json.loads(cleaned.strip())
        except Exception as e:
            print(f"[{self.name}] Parse error: {e}")
            llm_out = {
                "thought": "Fallback: ask basic symptom question",
                "action": {"type": "tool", "name": "ask_patient_question",
                           "args": {"question": "Do you experience burning pain in your feet?",
                                    "options": ["Yes, frequently", "Sometimes", "No"],
                                    "key": f"sym_{state.iteration}"}},
                "confidence": 0.1,
                "continue": True
            }

        thought = llm_out.get("thought", "")
        action = llm_out.get("action", {})
        message = llm_out.get("message", "")
        confidence = float(llm_out.get("confidence", 0.0))
        should_continue = llm_out.get("continue", True)

        state.add_reasoning(thought, action)
        state.confidence = max(state.confidence, confidence)
        state.emit("thought", thought, self.name)

        if message:
            state.add_message("assistant", message)

        # Route based on action
        action_type = action.get("type")
        action_name = action.get("name", "")

        if not should_continue or action_type == "route" and action_name == "report_node":
            state.next_node = NODE_REPORT
        elif action_type == "route" and action_name == "ml_node":
            state.next_node = NODE_ML
        elif action_type == "route" and action_name == "fusion_node":
            state.next_node = NODE_FUSION
        elif action_type == "tool":
            state.pending_tool_call = action
            state.next_node = NODE_TOOL
        else:
            # Default: advance based on data state
            if not should_continue:
                state.next_node = NODE_REPORT
            elif state.is_data_sufficient() and not state.has_ml_data():
                state.next_node = NODE_ML
            elif state.has_ml_data() and not state.has_fusion():
                state.next_node = NODE_FUSION
            elif state.has_fusion():
                state.next_node = NODE_REPORT
            else:
                state.pending_tool_call = action
                state.next_node = NODE_TOOL

        state.log(self.name, "Reasoning complete", {"next": state.next_node, "confidence": confidence})
        return state


# ══════════════════════════════════════════════════════════════════
# 4. TOOL EXECUTION NODE (routes to correct tool)
# ══════════════════════════════════════════════════════════════════
class ToolNode:
    name = "ToolNode"

    def run(self, state: MultiAgentState) -> MultiAgentState:
        tool_call = state.pending_tool_call
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})

        state.log(self.name, f"Executing tool: {tool_name}", tool_args)
        state.emit("tool_call", f"Calling: {tool_name}", self.name)

        # Use existing tool registry (wraps state differently — adapt here)
        from agent_state import AgentState as LegacyState

        legacy = LegacyState(
            patient_id=state.patient_id,
            patient_info=state.patient_info,
            answers=state.answers,
            clinical_data=state.clinical_scores,
            ml_results=state.ml_results,
            retrieved_memory=state.long_term,
        )

        observation = execute_tool(tool_name, legacy, tool_args)

        # Sync back relevant state
        state.answers = legacy.answers
        if legacy.clinical_data:
            state.clinical_scores.update({k: v for k, v in legacy.clinical_data.items() if v is not None})
        if legacy.ml_results:
            state.ml_results.update(legacy.ml_results)

        state.last_tool_observation = observation
        state.add_reasoning("", tool_call, observation)
        state.emit("observation", f"{tool_name} → {str(observation)[:150]}", self.name)

        # Handle patient question pause
        if tool_name == "ask_patient_question" and legacy.waiting_for_patient:
            state.waiting_for_patient = True
            state.pending_question = legacy.pending_question
            state.next_node = NODE_WAIT
        else:
            state.next_node = NODE_REFLECTION

        state.log(self.name, "Tool complete", {"observation": str(observation)[:200]})
        return state


# ══════════════════════════════════════════════════════════════════
# 5. ML INFERENCE AGENT
# ══════════════════════════════════════════════════════════════════
class MLInferenceAgent:
    name = "MLInferenceAgent"

    def run(self, state: MultiAgentState) -> MultiAgentState:
        state.log(self.name, "Running ML neuropathy inference")
        state.emit("agent_start", "Running ML neuropathy prediction model...", self.name)

        # Try loading stored prediction first
        try:
            stored = get_patient_ml_prediction(state.patient_id)
            if stored and stored.get("predicted_class") is not None:
                state.ml_results = {
                    "predicted_class": stored["predicted_class"],
                    "predicted_probability": stored["predicted_probability"],
                    "ai_prediction": "مريض" if stored["predicted_class"] == 1 else "سليم",
                    "source": "database"
                }
                state.emit("ml_result",
                    f"Stored ML: class={stored['predicted_class']}, prob={stored['predicted_probability']:.2f}",
                    self.name)
                state.next_node = NODE_FUSION
                return state
        except Exception:
            pass

        # Compute from answers
        scores = calculate_section_scores(state.answers)
        nss = scores.get("nss_score", 0)
        age = state.patient_info.get("age", 50)

        result = ml_neuropathy_prediction(state.answers, nss, age)
        state.ml_results = {**result, "source": "computed"}

        # Persist to DB
        try:
            f = result.get("features", {})
            supabase.table("ml_neuropathy_predictions").insert({
                "patient_id": state.patient_id,
                "nss_score": f.get("nss", nss),
                "bmi_baseline": f.get("bmi", 22),
                "age_baseline": f.get("age", age),
                "hba1c_baseline": f.get("hba1c", 7),
                "heat_avg": (f.get("heat_right", 38) + f.get("heat_left", 38)) / 2,
                "cold_avg": (f.get("cold_right", 20) + f.get("cold_left", 20)) / 2,
                "predicted_class": result["predicted_class"],
                "predicted_probability": result["predicted_probability"]
            }).execute()
        except Exception:
            pass

        state.emit("ml_result",
            f"ML computed: class={result['predicted_class']}, prob={result['predicted_probability']:.2f} ({result['ai_prediction']})",
            self.name)

        state.log(self.name, "ML inference done", result)
        state.next_node = NODE_FUSION
        return state


# ══════════════════════════════════════════════════════════════════
# 6. FUSION & DECISION AGENT
# ══════════════════════════════════════════════════════════════════
class FusionDecisionAgent:
    name = "FusionDecisionAgent"

    def run(self, state: MultiAgentState) -> MultiAgentState:
        state.log(self.name, "Computing fusion decision")
        state.emit("agent_start", "Computing weighted diagnostic fusion score...", self.name)

        ai_pred = state.ml_results.get("ai_prediction", "سليم")

        # Get scores from clinical_scores or recompute
        if state.clinical_scores.get("nds_score") is not None:
            nds = state.clinical_scores["nds_score"]
            nss = state.clinical_scores.get("nss_score", 0)
        else:
            scores = calculate_section_scores(state.answers)
            nds = scores["nds_score"]
            nss = scores["nss_score"]
            gum = scores["gum_score"]
            ulcer = scores["ulcer_score"]
            state.clinical_scores.update({"nss_score": nss, "nds_score": nds,
                                          "gum_score": gum, "ulcer_score": ulcer})

        result = compute_final_decision(ai_pred, nds, nss)
        state.fusion_results = result
        state.confidence = result["fusion_score"] / 3.1  # max possible = 3.1

        # Append to decision path
        state.decision_path.append(
            f"AI({ai_pred})×1.0 + NDS({nds}≥6:{result['nds_binary']})×1.2 + NSS({nss}≥5:{result['nss_binary']})×0.9"
            f" = {result['fusion_score']} → {result['final_decision']}"
        )

        # Save to DB
        try:
            supabase.table("final_diagnostic_decisions").insert({
                "patient_id": state.patient_id,
                "ai_prediction": ai_pred,
                "nds_score": nds,
                "nss_score": nss,
                "calculated_score": result["fusion_score"],
                "final_decision": result["final_decision"]
            }).execute()
        except Exception:
            pass

        state.emit("fusion",
            f"Fusion score: {result['fusion_score']:.2f} / threshold {result['threshold']} → {result['final_decision']} ({result['confidence']} confidence)",
            self.name)

        state.log(self.name, "Fusion complete", result)
        state.next_node = NODE_REFLECTION
        return state


# ══════════════════════════════════════════════════════════════════
# 7. REFLECTION AGENT
# ══════════════════════════════════════════════════════════════════
class ReflectionAgent:
    name = "ReflectionAgent"

    PROMPT = """You are a Medical Reflection Agent. Your job: evaluate the diagnostic reasoning for consistency, completeness, and clinical validity.

Current diagnostic state:
{state_summary}

Recent reasoning steps:
{reasoning}

Check for:
1. Are there any logical inconsistencies?
2. Is there missing critical data (NSS/NDS/ML)?
3. Are clinical rules followed (e.g., not diagnosing without data)?
4. Is the confidence level justified?
5. Should the plan be revised?

Respond ONLY as JSON:
{{
  "is_consistent": true/false,
  "issues": ["issue 1", "issue 2"],
  "suggestions": ["suggestion 1"],
  "should_replan": true/false,
  "confidence_valid": true/false
}}"""

    def run(self, state: MultiAgentState) -> MultiAgentState:
        state.log(self.name, "Self-reflection check")
        state.emit("agent_start", "Running self-reflection and consistency check...", self.name)

        recent_reasoning = "\n".join([
            f"[{r['iteration']}] thought: {r['thought'][:100]} | action: {r['action'].get('name','?')}"
            for r in state.reasoning_chain[-4:]
        ])

        prompt = self.PROMPT.format(
            state_summary=state.to_summary(),
            reasoning=recent_reasoning or "No reasoning yet."
        )

        raw = call_llm(prompt)
        try:
            cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            result = json.loads(cleaned)
        except Exception:
            result = {"is_consistent": True, "issues": [], "suggestions": [], "should_replan": False, "confidence_valid": True}

        reflection = ReflectionEntry(
            iteration=state.iteration,
            issues=result.get("issues", []),
            suggestions=result.get("suggestions", []),
            is_consistent=result.get("is_consistent", True),
            should_replan=result.get("should_replan", False)
        )
        state.reflections.append(reflection)
        state.reflection_count += 1

        if reflection.issues:
            state.emit("reflection", f"Issues: {'; '.join(reflection.issues[:2])}", self.name)
        else:
            state.emit("reflection", "Reasoning is consistent.", self.name)

        # Decide next node
        if state.has_fusion():
            state.next_node = NODE_REPORT
        elif reflection.should_replan and state.reflection_count < 3:
            state.next_node = NODE_REASONING  # re-reason after reflection
        else:
            state.next_node = NODE_REASONING

        state.log(self.name, "Reflection done", result)
        return state


# ══════════════════════════════════════════════════════════════════
# 8. REPORT GENERATOR AGENT
# ══════════════════════════════════════════════════════════════════
class ReportGeneratorAgent:
    name = "ReportGeneratorAgent"

    def run(self, state: MultiAgentState) -> MultiAgentState:
        state.log(self.name, "Generating final medical report")
        state.emit("agent_start", "Generating structured medical report...", self.name)

        scores = state.clinical_scores
        nss = scores.get("nss_score", 0)
        nds = scores.get("nds_score", 0)
        gum = scores.get("gum_score", 0)
        ulcer = scores.get("ulcer_score", 0)

        nss_cat = "Normal" if nss <= 2 else "Mild" if nss <= 4 else "Moderate" if nss <= 6 else "Severe"
        nds_cat = "Normal" if nds <= 5 else "Mild" if nds <= 10 else "Moderate" if nds <= 16 else "Severe"
        gum_cat = "Healthy" if gum <= 5 else "Mild gingivitis" if gum <= 11 else "Moderate gingivitis" if gum <= 18 else "Severe gingivitis"
        ulcer_cat = "Low risk" if ulcer < 4 else "Moderate risk" if ulcer < 8 else "High risk"

        ml = state.ml_results
        fusion = state.fusion_results
        reflections_summary = "; ".join([
            f"[{r.iteration}] {'OK' if r.is_consistent else ', '.join(r.issues[:1])}"
            for r in state.reflections[-3:]
        ])

        decision_path = "\n".join(state.decision_path) or "Direct fusion computation"
        rag_context = "\n".join([
            f"  - {r.get('content','')[:80]}" for r in state.long_term[:3]
        ]) or "No prior history"

        prompt = f"""Generate a comprehensive, patient-friendly medical diagnostic report in English.

=== PATIENT ===
Name: {state.patient_info.get('name')}, Age: {state.patient_info.get('age')}

=== CLINICAL SCORES ===
NSS: {nss}/14 ({nss_cat}) | NDS: {nds}/23 ({nds_cat})
Gum Health: {gum} ({gum_cat}) | Ulcer Risk: {ulcer} ({ulcer_cat})

=== ML INFERENCE ===
Predicted Class: {ml.get('predicted_class', 'N/A')} ({'Neuropathy' if ml.get('predicted_class') == 1 else 'Healthy'})
Probability: {ml.get('predicted_probability', 0):.1%} | Source: {ml.get('source', 'N/A')}

=== FUSION DECISION ===
Score: {fusion.get('fusion_score', 0)} / threshold {fusion.get('threshold', 1.55)}
Decision: {fusion.get('final_decision', 'N/A')}
Confidence: {fusion.get('confidence', 'N/A')}
Decision Path: {decision_path}

=== AGENT REFLECTIONS ===
{reflections_summary or 'No inconsistencies detected.'}

=== RAG MEMORY CONTEXT ===
{rag_context}

=== REASONING ITERATIONS ===
Total iterations: {state.iteration}
Tools called: {len([e for e in state.audit_log if 'Executing' in e.action])}

Write the final report with:
1. 🏁 Final Diagnosis (one clear sentence)
2. 🧠 Neuropathy Assessment (NSS + NDS + ML model)
3. 🦷 Gum Health Assessment
4. 🩹 Foot Ulcer Risk Assessment
5. 📊 Decision Explanation (how the score was computed, in simple terms)
6. 🔍 Uncertainty Note (what we are not certain about)
7. 📌 Recommendations (4-5 bullet points)
8. 🚨 Disclaimer: This is AI-assisted analysis only. Always consult a certified physician."""

        report = call_llm(prompt)
        state.final_report = report
        state.is_complete = True
        state.next_node = NODE_END

        state.emit("final_report", report, self.name)
        state.log(self.name, "Report generated", {"length": len(report)})
        return state
