"""
agent.py — Core Autonomous Medical Diagnostic Agent (ReAct / AutoGPT-style).

Architecture:
  1. Planning Phase  → LLM generates a step-by-step plan
  2. Agent Loop      → Thought → Action → Observation → State Update
  3. Stop Conditions → High confidence | is_diagnosis_complete | max iterations

Each LLM call returns strict JSON:
{
  "thought": "reasoning",
  "action": { "type": "tool|final", "name": "...", "args": {} },
  "message": "optional patient-facing text",
  "continue": true/false
}
"""

import json
from agent_state import AgentState
from tools import TOOL_DESCRIPTIONS, execute_tool
from rag_engine import call_llm
from database import save_conversation_memory, get_all_patients, supabase


# ══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an Autonomous Medical Diagnostic Agent specializing in Diabetic Peripheral Neuropathy (DPN).

You operate in a ReAct reasoning loop:
  Thought → Action → Observation → Update State → Repeat

{tool_descriptions}

STRICT RULES:
- You MUST always respond with valid JSON only. No markdown, no explanation outside JSON.
- You MUST call tools dynamically — never make assumptions about data you haven't retrieved.
- You MUST ask the patient questions to collect missing clinical data.
- You MUST call calculate_fusion_score BEFORE generating the final report.
- You MUST generate a final report when: fusion is computed AND confidence is high.
- You MUST NOT loop more than {max_iter} times. Stop with final report if limit is near.

JSON format at EVERY step:
{{
  "thought": "Your internal reasoning (NOT shown to patient)",
  "action": {{
    "type": "tool",
    "name": "tool_name",
    "args": {{}}
  }},
  "message": "Optional patient-facing message (shown in chat)",
  "continue": true
}}

When ready for final diagnosis, use:
{{
  "thought": "All data collected. Generating final report.",
  "action": {{
    "type": "final",
    "name": "generate_report",
    "args": {{}}
  }},
  "message": "...",
  "continue": false
}}
"""

PLANNING_PROMPT = """You are planning a diagnostic session for a diabetic patient.

Patient: {patient_name}, Age: {patient_age}
Existing data in DB: NSS={has_nss}, NDS={has_nds}, ML Prediction={has_ml}

Generate a concise step-by-step plan (3-6 steps) to reach a confident diagnosis.
The plan must cover: memory retrieval, data collection, ML prediction, fusion, and report.

Respond ONLY with a JSON array of strings:
["step 1", "step 2", ...]
"""

REACT_PROMPT = """
=== CURRENT AGENT STATE ===
{state_context}

=== YOUR TASK ===
Based on the state above, decide your next action.
Choose the most logical next tool to call, or generate the final report if ready.

Remember:
- If clinical data is missing, ask the patient questions.
- If ML results are missing, call compute_ml_prediction (after enough answers).
- If fusion score is missing, call calculate_fusion_score.
- If everything is ready, generate the final report.

Respond ONLY in strict JSON format as specified.
"""

FINAL_REPORT_PROMPT = """
Generate a comprehensive, patient-friendly medical diagnostic report.

=== FULL DIAGNOSTIC DATA ===
Patient: {patient_name}, Age: {patient_age}

NSS Score: {nss_score}/14 ({nss_cat})
NDS Score: {nds_score}/23 ({nds_cat})
Gum Health Score: {gum_score} ({gum_cat})
Ulcer Risk Score: {ulcer_score} ({ulcer_cat})

ML Prediction: Class={ml_class}, Probability={ml_prob:.1%}
Fusion Score: {fusion_score} / threshold {threshold}

🏁 FINAL DECISION: {final_decision}
Confidence: {confidence}

Reasoning Chain Summary:
{reasoning_summary}

Write the report with these sections:
1. 🏁 Final Diagnosis (one clear sentence)
2. 🧠 Neuropathy Assessment (NSS + NDS + ML combined)
3. 🦷 Gum Health
4. 🩹 Foot Ulcer Risk
5. 📊 Decision Calculation (1.0×AI + 1.2×NDS + 0.9×NSS = {fusion_score})
6. 📌 Recommendations (3-5 bullet points)
7. 🚨 Disclaimer: This is AI-assisted analysis only. Always consult a certified physician.
"""


# ══════════════════════════════════════════════════════════════════
# AGENT CLASS
# ══════════════════════════════════════════════════════════════════

class MedicalDiagnosticAgent:
    """
    Autonomous Medical Diagnostic Agent — ReAct / AutoGPT-style.
    """

    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations

    # ── Planning Phase ─────────────────────────────────────────────
    def plan(self, state: AgentState) -> list[str]:
        """LLM generates a diagnostic plan before the agent loop starts."""
        prompt = PLANNING_PROMPT.format(
            patient_name=state.patient_info.get("name", "Unknown"),
            patient_age=state.patient_info.get("age", "N/A"),
            has_nss=bool(state.clinical_data.get("nss_score")),
            has_nds=bool(state.clinical_data.get("nds_score")),
            has_ml=bool(state.ml_results.get("predicted_class") is not None),
        )

        raw = call_llm(prompt)
        try:
            cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            plan = json.loads(cleaned)
            if isinstance(plan, list):
                return plan
        except Exception:
            pass

        # Fallback plan
        return [
            "Search patient memory for prior history",
            "Retrieve stored clinical data from DB",
            "Ask patient symptom questions if data is missing",
            "Run ML neuropathy prediction",
            "Calculate fusion score",
            "Generate final diagnosis report"
        ]

    # ── Single Iteration ───────────────────────────────────────────
    def think_and_act(self, state: AgentState) -> dict:
        """
        One ReAct iteration: Thought → Action → returns action dict.
        Does NOT execute the tool — caller does that.
        """
        system = SYSTEM_PROMPT.format(
            tool_descriptions=TOOL_DESCRIPTIONS,
            max_iter=self.max_iterations
        )
        context = REACT_PROMPT.format(state_context=state.to_context_string())
        full_prompt = system + "\n\n" + context

        raw = call_llm(full_prompt)

        # Parse JSON response
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            result = json.loads(cleaned.strip())
            return result
        except Exception as e:
            print(f"[Agent] Failed to parse LLM JSON: {e}\nRaw: {raw[:200]}")
            # Fallback: ask a generic question
            return {
                "thought": "Could not parse LLM response. Asking patient a basic question.",
                "action": {
                    "type": "tool",
                    "name": "ask_patient_question",
                    "args": {
                        "question": "Do you experience any numbness or burning sensation in your feet?",
                        "options": ["Yes, frequently", "Sometimes", "Rarely", "No"],
                        "key": f"fallback_q_{state.iteration}"
                    }
                },
                "message": "Do you experience any numbness or burning sensation in your feet?",
                "continue": True
            }

    # ── Final Report Generator ─────────────────────────────────────
    def generate_final_report(self, state: AgentState) -> str:
        """Generates the structured final medical report."""
        from questionnaire import calculate_section_scores

        scores = calculate_section_scores(state.answers) if state.answers else {}
        nss = state.clinical_data.get("nss_score") or scores.get("nss_score", 0)
        nds = state.clinical_data.get("nds_score") or scores.get("nds_score", 0)
        gum = state.clinical_data.get("gum_score") or scores.get("gum_score", 0)
        ulcer = state.clinical_data.get("ulcer_score") or scores.get("ulcer_score", 0)

        nss_cat = "Normal" if nss <= 2 else "Mild" if nss <= 4 else "Moderate" if nss <= 6 else "Severe"
        nds_cat = "Normal" if nds <= 5 else "Mild" if nds <= 10 else "Moderate" if nds <= 16 else "Severe"
        gum_cat = "Healthy" if gum <= 5 else "Mild gingivitis" if gum <= 11 else "Moderate gingivitis" if gum <= 18 else "Severe gingivitis"
        ulcer_cat = "Low risk" if ulcer < 4 else "Moderate risk" if ulcer < 8 else "High risk"

        ml = state.ml_results or {}
        fusion = state.fusion_results or {}

        reasoning_summary = "\n".join([
            f"  [{r['iteration']}] {r['thought'][:80]}..."
            for r in state.intermediate_reasoning[-5:]
        ])

        prompt = FINAL_REPORT_PROMPT.format(
            patient_name=state.patient_info.get("name", "Patient"),
            patient_age=state.patient_info.get("age", "N/A"),
            nss_score=nss, nss_cat=nss_cat,
            nds_score=nds, nds_cat=nds_cat,
            gum_score=gum, gum_cat=gum_cat,
            ulcer_score=ulcer, ulcer_cat=ulcer_cat,
            ml_class=ml.get("predicted_class", "N/A"),
            ml_prob=ml.get("predicted_probability", 0),
            fusion_score=fusion.get("fusion_score", 0),
            threshold=fusion.get("threshold", 1.55),
            final_decision=fusion.get("final_decision", "Insufficient data"),
            confidence=fusion.get("confidence", "Low"),
            reasoning_summary=reasoning_summary,
        )

        return call_llm(prompt)

    # ── Main Agent Step (called once per UI interaction) ───────────
    def step(self, state: AgentState, patient_answer: str = None) -> dict:
        """
        Advance the agent by one step.
        - If patient_answer is provided, resume from waiting state.
        - Returns: { "type": "question" | "message" | "final", "content": ..., "options": [...] }
        """

        # ── Resume after patient answered ──────────────────────────
        if state.waiting_for_patient and patient_answer is not None:
            pq = state.pending_question
            key = pq.get("key", f"q_{state.iteration}")
            state.answers[key] = patient_answer
            state.add_message("user", patient_answer)
            state.add_observation("patient_answer", {"key": key, "answer": patient_answer})
            state.waiting_for_patient = False
            state.pending_question = {}

            # Save to memory
            try:
                emb = None
                save_conversation_memory(
                    state.patient_id,
                    state.patient_info.get("session_id", ""),
                    "user",
                    patient_answer,
                    emb
                )
            except Exception:
                pass

        # ── Stop conditions ─────────────────────────────────────────
        if state.is_complete:
            return {"type": "final", "content": state.final_report, "options": []}

        if state.iteration >= self.max_iterations:
            state.is_complete = True
            state.final_report = self.generate_final_report(state)
            return {"type": "final", "content": state.final_report, "options": []}

        # ── Think & Act ────────────────────────────────────────────
        state.iteration += 1
        llm_output = self.think_and_act(state)

        thought = llm_output.get("thought", "")
        action = llm_output.get("action", {})
        message = llm_output.get("message", "")
        should_continue = llm_output.get("continue", True)

        state.add_thought(thought, action)

        # ── Final action ────────────────────────────────────────────
        if action.get("type") == "final" or not should_continue:
            state.final_report = self.generate_final_report(state)
            state.is_complete = True
            try:
                save_conversation_memory(
                    state.patient_id,
                    state.patient_info.get("session_id", ""),
                    "assistant",
                    state.final_report,
                    None
                )
            except Exception:
                pass
            return {"type": "final", "content": state.final_report, "options": []}

        # ── Tool action ─────────────────────────────────────────────
        tool_name = action.get("name", "")
        tool_args = action.get("args", {})

        observation = execute_tool(tool_name, state, tool_args)

        # ── Handle question tool (pause loop) ──────────────────────
        if tool_name == "ask_patient_question" and state.waiting_for_patient:
            q = state.pending_question
            if message:
                state.add_message("assistant", message)
            return {
                "type": "question",
                "content": q.get("question", message),
                "options": q.get("options", []),
                "key": q.get("key", "")
            }

        # ── Non-blocking tool: show message and auto-continue ───────
        response_msg = message or f"[{tool_name}] → {str(observation)[:100]}"
        if response_msg:
            state.add_message("assistant", response_msg)

        return {
            "type": "message",
            "content": response_msg,
            "options": [],
            "tool": tool_name,
            "observation": observation
        }


# ══════════════════════════════════════════════════════════════════
# FACTORY — Build a fresh agent + state for a patient
# ══════════════════════════════════════════════════════════════════

def create_agent_session(patient: dict, session_id: str) -> tuple[MedicalDiagnosticAgent, AgentState]:
    """Initialize a new agent + state for a patient session."""
    import uuid

    agent = MedicalDiagnosticAgent(max_iterations=10)
    state = AgentState(
        patient_id=patient["id"],
        patient_info={**patient, "session_id": session_id},
        max_iterations=10
    )

    # Planning phase
    state.plan = agent.plan(state)

    return agent, state
