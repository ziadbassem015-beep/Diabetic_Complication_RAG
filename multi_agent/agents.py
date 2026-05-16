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
from core.rag_engine import call_llm
from core.services.diagnostic_service import DiagnosticService
from core.tools import TOOL_REGISTRY, TOOL_DESCRIPTIONS, execute_tool


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

        # Build RAG and Episodic context
        rag_ctx = ""
        if state.long_term:
            rag_ctx += "\n[RAG Conversation History]:\n" + "\n".join([
                f"  - {r.get('content','')[:120]}" for r in state.long_term[:3]
            ])
            
        if hasattr(state, 'episodic') and state.episodic:
            rag_ctx += "\n[Past Medical History (Previous Diagnoses)]:\n" + "\n".join([
                f"  - {r.get('created_at', '')[:10]}: {r.get('final_decision')} (NSS:{r.get('nss_score')}, NDS:{r.get('nds_score')})" 
                for r in state.episodic[:3]
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
            stored = DiagnosticService.get_ml_prediction(state.patient_id)
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
        nss = state.clinical_scores.get("nss_score", 0)
        age = state.patient_info.get("age", 50)
        result = DiagnosticService.run_ml_inference(
            patient_id=state.patient_id,
            answers=state.answers,
            nss_score=nss,
            age=age,
        )
        state.ml_results = {**result, "source": "computed"}

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

        # Get scores from clinical_scores or recompute via service
        if state.clinical_scores.get("nds_score") is not None:
            nds = state.clinical_scores["nds_score"]
            nss = state.clinical_scores.get("nss_score", 0)
        else:
            scores = DiagnosticService.save_clinical_data(state.patient_id, state.answers)
            nds = scores.get("nds_score", 0)
            nss = scores.get("nss_score", 0)
            gum = scores.get("gum_score", 0)
            ulcer = scores.get("ulcer_score", 0)
            state.clinical_scores.update({"nss_score": nss, "nds_score": nds,
                                          "gum_score": gum, "ulcer_score": ulcer})

        result = DiagnosticService.compute_fusion(
            patient_id=state.patient_id,
            ai_prediction=ai_pred,
            nds_score=nds,
            nss_score=nss,
        )
        state.fusion_results = result
        state.confidence = result["fusion_score"] / 3.1  # max possible = 3.1

        # Append to decision path
        state.decision_path.append(
            f"AI({ai_pred})×1.0 + NDS({nds}≥6:{result['nds_binary']})×1.2 + NSS({nss}≥5:{result['nss_binary']})×0.9"
            f" = {result['fusion_score']} → {result['final_decision']}"
        )

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

    @staticmethod
    def _safe_get(data: dict, key: str, default: str = "") -> str:
        if not data:
            return default
        val = data.get(key)
        if val is None or val == "":
            return default
        return str(val)

    @staticmethod
    def _format_risk(value) -> str:
        if value is None or value == "":
            return "Insufficient data for this section"
        if isinstance(value, (int, float)):
            if isinstance(value, float) and 0 <= value <= 1:
                return f"{value * 100:.1f}% estimated probability"
            return str(value)
        return str(value)

    @staticmethod
    def _format_section(title: str, content: str) -> str:
        body = content.strip() if content and content.strip() else "Insufficient data for this section"
        return f"{title}\n{body}\n"

    @staticmethod
    def _severity_label(score, thresholds: list) -> str:
        if score is None:
            return "Insufficient data for this section"
        for limit, label in thresholds:
            if score <= limit:
                return label
        return thresholds[-1][1] if thresholds else "Insufficient data for this section"

    @classmethod
    def _questionnaire_summary(cls, state: MultiAgentState) -> str:
        if not state.answers:
            return "Insufficient data for this section"
        return f"{len(state.answers)} clinical responses recorded during this assessment."

    @classmethod
    def _section_gestational_overview(cls, gender: str, gd: dict) -> str:
        if gender == "Male":
            return "Not applicable for male patients"
        if gd.get("skipped"):
            return "Not applicable for male patients"
        ml_gd = gd.get("ml_result") or {}
        if not ml_gd and not gd.get("gd_scores"):
            return "Insufficient data for this section"
        parts = []
        if gd.get("gd_scores"):
            parts.append(f"Screening score: {cls._safe_get(gd['gd_scores'], 'gd_score', 'N/A')}")
        if ml_gd:
            parts.append(f"Risk level: {cls._safe_get(ml_gd, 'risk_level', 'N/A')}")
            parts.append(f"Estimated probability: {cls._format_risk(ml_gd.get('predicted_probability'))}")
        return "\n".join(parts) if parts else "Insufficient data for this section"

    @classmethod
    def _section_heart_overview(cls, hr: dict) -> str:
        ml_hr = hr.get("ml_result") or {}
        if not ml_hr and not hr.get("hr_scores"):
            return "Insufficient data for this section"
        parts = []
        if hr.get("hr_scores"):
            parts.append(f"Screening score: {cls._safe_get(hr['hr_scores'], 'hr_score', 'N/A')}")
        if ml_hr:
            parts.append(f"Risk level: {cls._safe_get(ml_hr, 'risk_level', 'N/A')}")
            parts.append(f"Estimated probability: {cls._format_risk(ml_hr.get('predicted_probability'))}")
        return "\n".join(parts) if parts else "Insufficient data for this section"

    @classmethod
    def _section_gestational_report(cls, gender: str, gd: dict) -> str:
        if gender == "Male" or gd.get("skipped"):
            return "Not applicable for male patients"
        ml_gd = gd.get("ml_result") or {}
        scores = gd.get("gd_scores") or {}
        if not ml_gd and not scores:
            return "Insufficient data for this section"
        lines = ["Gestational diabetes screening (secondary assessment)."]
        if scores:
            lines.append(
                f"Combined screening score: {scores.get('gd_score', 'N/A')} "
                f"(maximum reference {scores.get('gd_max_score', 12)})."
            )
        if ml_gd:
            lines.append(f"Risk classification: {cls._safe_get(ml_gd, 'risk_level', 'N/A')}.")
            prob = ml_gd.get("predicted_probability")
            if prob is not None:
                lines.append(f"Estimated probability: {prob * 100:.1f}%.")
        lines.append(
            "This result is supplementary and does not replace prenatal care or laboratory glucose testing."
        )
        return "\n".join(lines)

    @classmethod
    def _section_heart_report(cls, hr: dict) -> str:
        ml_hr = hr.get("ml_result") or {}
        scores = hr.get("hr_scores") or {}
        if not ml_hr and not scores:
            return "Insufficient data for this section"
        lines = ["Cardiovascular risk screening (secondary assessment)."]
        if scores:
            lines.append(
                f"Combined screening score: {scores.get('hr_score', 'N/A')} "
                f"(maximum reference {scores.get('hr_max_score', 15)})."
            )
        if ml_hr:
            lines.append(f"Risk classification: {cls._safe_get(ml_hr, 'risk_level', 'N/A')}.")
            prob = ml_hr.get("predicted_probability")
            if prob is not None:
                lines.append(f"Estimated probability: {prob * 100:.1f}%.")
        lines.append(
            "This result supports lifestyle and medical follow-up; confirm with clinical evaluation."
        )
        return "\n".join(lines)

    @classmethod
    def generate_report(cls, state: MultiAgentState) -> str:
        info = state.patient_info or {}
        scores = state.clinical_scores or {}
        ml = state.ml_results or {}
        fusion = state.fusion_results or {}
        gd = state.gestational_results or {}
        hr = state.heart_risk_results or {}

        gender = cls._safe_get(info, "gender", "Unknown")
        name = cls._safe_get(info, "name", "Patient")
        age = info.get("age")
        age_text = str(age) if age is not None else "Not recorded"
        diabetes_type = cls._safe_get(info, "diabetes_type", "Not recorded")
        diabetes_duration = info.get("diabetes_duration")
        duration_text = (
            f"{diabetes_duration} year(s)" if diabetes_duration is not None else "Not recorded"
        )

        nss = scores.get("nss_score")
        nds = scores.get("nds_score")
        gum = scores.get("gum_score")
        ulcer = scores.get("ulcer_score")

        nss_cat = cls._severity_label(
            nss, [(2, "Normal"), (4, "Mild"), (6, "Moderate"), (999, "Severe")]
        )
        nds_cat = cls._severity_label(
            nds, [(5, "Normal"), (10, "Mild"), (16, "Moderate"), (999, "Severe")]
        )
        gum_cat = cls._severity_label(
            gum, [(5, "Healthy"), (11, "Mild gingivitis"), (18, "Moderate gingivitis"), (999, "Severe gingivitis")]
        )
        ulcer_cat = cls._severity_label(
            ulcer, [(3, "Low risk"), (7, "Moderate risk"), (999, "High risk")]
        )

        primary_decision = cls._safe_get(fusion, "final_decision", "")
        if not primary_decision:
            primary_decision = "Insufficient data for this section"
        fusion_confidence = cls._safe_get(fusion, "confidence", "Not stated")
        fusion_score = fusion.get("fusion_score")
        fusion_score_text = (
            f"{fusion_score:.2f}" if fusion_score is not None else "Not recorded"
        )

        ml_class = ml.get("predicted_class")
        if ml_class is None:
            ml_interpretation = "Insufficient data for this section"
        else:
            ml_interpretation = (
                "Findings suggest possible peripheral neuropathy."
                if ml_class == 1
                else "Findings do not suggest peripheral neuropathy at this time."
            )
        ml_prob = ml.get("predicted_probability")
        ml_prob_text = (
            f"{ml_prob * 100:.1f}%" if ml_prob is not None else "Not recorded"
        )

        gestational_overview = cls._section_gestational_overview(gender, gd)
        heart_overview = cls._section_heart_overview(hr)

        risk_lines = []
        if nss is not None:
            risk_lines.append(f"Neuropathy symptom burden (NSS): {nss}/14 ({nss_cat}).")
        else:
            risk_lines.append("Neuropathy symptom burden (NSS): Insufficient data for this section.")
        if nds is not None:
            risk_lines.append(f"Neuropathy disability score (NDS): {nds}/23 ({nds_cat}).")
        else:
            risk_lines.append("Neuropathy disability score (NDS): Insufficient data for this section.")
        if gum is not None:
            risk_lines.append(f"Gum health score: {gum} ({gum_cat}).")
        if ulcer is not None:
            risk_lines.append(f"Foot ulcer risk score: {ulcer} ({ulcer_cat}).")
        if ml_prob is not None:
            risk_lines.append(f"Neuropathy model estimate: {ml_prob_text}.")
        risk_summary = "\n".join(risk_lines)

        secondary_lines = [
            "Secondary assessments are provided for additional context only.",
            "They do not change the primary neuropathy (PDN) conclusion.",
            "",
            "5.1 Gestational Diabetes",
            gestational_overview,
            "",
            "5.2 Heart Risk Assessment",
            heart_overview,
        ]

        insights = []
        if nss is not None and nds is not None:
            insights.append(
                f"Symptom and disability scores (NSS {nss}/14, NDS {nds}/23) inform neuropathy status."
            )
        if gum is not None:
            insights.append(f"Oral health screening indicates {gum_cat.lower()} gum status.")
        if ulcer is not None:
            insights.append(f"Foot ulcer screening indicates {ulcer_cat.lower()} risk.")
        if gender == "Female" and gestational_overview != "Not applicable for male patients":
            if "Insufficient" not in gestational_overview:
                insights.append("Gestational diabetes screening completed as a secondary check.")
        if "Insufficient" not in heart_overview:
            insights.append("Cardiovascular risk screening completed as a secondary check.")
        if not insights:
            insights.append("Insufficient data for this section")
        clinical_insights = "\n".join(insights)

        recommendations = [
            "Follow up with your physician to review these results and confirm any diagnosis.",
            "Maintain blood glucose, blood pressure, and foot care as advised by your care team.",
            "Seek urgent care for new numbness, wounds that do not heal, chest pain, or severe symptoms.",
        ]
        if ml_class == 1 or (nds is not None and nds >= 6) or (nss is not None and nss >= 5):
            recommendations.append(
                "Discuss nerve-related symptoms and protective foot care at your next visit."
            )
        if gender == "Female" and gestational_overview != "Not applicable for male patients":
            if "Insufficient" not in gestational_overview:
                recommendations.append(
                    "If pregnant or planning pregnancy, discuss glucose monitoring with your obstetric team."
                )
        if "Insufficient" not in heart_overview:
            recommendations.append(
                "Discuss cholesterol, blood pressure, exercise, and smoking cessation with your clinician."
            )

        stratification = [
            f"Primary (PDN): {primary_decision} (confidence: {fusion_confidence}).",
            f"Combined neuropathy score: {fusion_score_text}.",
        ]
        if gender == "Female" and gestational_overview != "Not applicable for male patients":
            stratification.append(f"Gestational diabetes (secondary): {gestational_overview.split(chr(10))[0]}.")
        elif gender == "Male":
            stratification.append("Gestational diabetes (secondary): Not applicable for male patients.")
        if "Insufficient" not in heart_overview:
            stratification.append(f"Cardiovascular risk (secondary): {heart_overview.split(chr(10))[0]}.")
        else:
            stratification.append("Cardiovascular risk (secondary): Insufficient data for this section.")

        follow_up = [
            "Schedule a routine review with your primary care physician or diabetes specialist.",
            "Bring this summary and any recent laboratory results to your appointment.",
            "Repeat screening intervals should be decided by your treating clinician.",
        ]

        neuropathy_interp_lines = []
        if nss is not None:
            neuropathy_interp_lines.append(f"NSS total: {nss}/14 ({nss_cat}).")
        if nds is not None:
            neuropathy_interp_lines.append(f"NDS total: {nds}/23 ({nds_cat}).")
        if ml_class is not None:
            neuropathy_interp_lines.append(f"Model estimate: {ml_interpretation} ({ml_prob_text}).")
        if fusion_score is not None:
            neuropathy_interp_lines.append(
                f"Overall neuropathy assessment score: {fusion_score_text} "
                f"(confidence: {fusion_confidence})."
            )
        if not neuropathy_interp_lines:
            neuropathy_interp_lines.append("Insufficient data for this section")
        neuropathy_interpretation = "\n".join(neuropathy_interp_lines)

        patient_summary = (
            f"Name: {name}\n"
            f"Age: {age_text}\n"
            f"Gender: {gender}\n"
            f"Diabetes type: {diabetes_type}\n"
            f"Duration of diabetes: {duration_text}\n"
            f"{cls._questionnaire_summary(state)}"
        )

        sections = [
            "## Patient Diagnostic Report",
            "",
            cls._format_section(
                "1. Patient Summary",
                patient_summary,
            ),
            cls._format_section(
                "2. Primary Diagnosis (Neuropathy - PDN)",
                (
                    f"Primary diagnosis (peripheral diabetic neuropathy assessment): {primary_decision}\n"
                    f"Assessment confidence: {fusion_confidence}."
                ),
            ),
            cls._format_section(
                "3. Neuropathy Clinical Interpretation",
                neuropathy_interpretation,
            ),
            cls._format_section(
                "4. Risk Factors Summary",
                risk_summary,
            ),
            cls._format_section(
                "5. Secondary Assessment Overview",
                "\n".join(secondary_lines),
            ),
            cls._format_section(
                "6. Clinical Insights",
                clinical_insights,
            ),
            cls._format_section(
                "7. Recommendations",
                "\n".join(f"- {r}" for r in recommendations),
            ),
            cls._format_section(
                "8. Risk Stratification Summary",
                "\n".join(stratification),
            ),
            cls._format_section(
                "9. Follow-up Advice",
                "\n".join(follow_up),
            ),
            cls._format_section(
                "10. Gestational Diabetes Report",
                cls._section_gestational_report(gender, gd),
            ),
            cls._format_section(
                "11. Cardiovascular Risk Report",
                cls._section_heart_report(hr),
            ),
            (
                "Disclaimer: This report is generated from structured screening responses "
                "and is for educational support only. It does not replace examination, "
                "laboratory testing, or advice from a licensed healthcare provider."
            ),
        ]

        report = "\n".join(sections)
        words = report.split()
        if len(words) > 900:
            report = " ".join(words[:900]) + "\n\n[Report truncated to 900 words.]"
        return report

    def run(self, state: MultiAgentState) -> MultiAgentState:
        state.log(self.name, "Generating final medical report")
        state.emit("agent_start", "Generating structured medical report...", self.name)

        report = self.generate_report(state)
        state.final_report = report
        state.is_complete = True
        state.next_node = NODE_END

        state.emit("final_report", report, self.name)
        state.log(self.name, "Report generated", {"length": len(report)})
        return state
