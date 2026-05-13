"""
agent_state.py — Central state object for the Autonomous Medical Diagnostic Agent.
Holds all short-term, structured, and reasoning memory across iterations.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    # ── Patient Context ────────────────────────────────────────────
    patient_id: str = ""
    patient_info: dict = field(default_factory=dict)

    # ── Short-term Memory (current session) ───────────────────────
    chat_history: list[dict] = field(default_factory=list)

    # ── Observation Log (what the agent has discovered) ───────────
    observations: list[dict] = field(default_factory=list)

    # ── Long-term RAG Memory (Supabase vector search results) ─────
    retrieved_memory: list[dict] = field(default_factory=list)

    # ── Structured Medical State ───────────────────────────────────
    clinical_data: dict = field(default_factory=dict)    # NSS, NDS scores
    ml_results: dict = field(default_factory=dict)       # ML prediction
    fusion_results: dict = field(default_factory=dict)   # Final decision scores

    # ── Agent Planning & Reasoning ─────────────────────────────────
    plan: list[str] = field(default_factory=list)        # LLM-generated plan
    intermediate_reasoning: list[dict] = field(default_factory=list)  # thought chain

    # ── Control ────────────────────────────────────────────────────
    iteration: int = 0
    max_iterations: int = 10
    is_complete: bool = False
    waiting_for_patient: bool = False   # True when agent asks a question
    pending_question: dict = field(default_factory=dict)  # question + options
    final_report: str = ""

    # ── Diagnosis Answers Collected ────────────────────────────────
    answers: dict = field(default_factory=dict)

    def add_observation(self, tool: str, result: Any, iteration: int = None):
        self.observations.append({
            "iteration": iteration or self.iteration,
            "tool": tool,
            "result": result
        })

    def add_thought(self, thought: str, action: dict, observation: Any = None):
        self.intermediate_reasoning.append({
            "iteration": self.iteration,
            "thought": thought,
            "action": action,
            "observation": observation
        })

    def add_message(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content})

    def is_data_sufficient(self) -> bool:
        """Check if agent has collected enough data for diagnosis."""
        has_clinical = bool(self.clinical_data)
        has_answers = len(self.answers) >= 5
        return has_clinical or has_answers

    def to_context_string(self) -> str:
        """Serialize state into a compact context string for LLM prompts."""
        lines = []

        lines.append(f"Patient: {self.patient_info.get('name', 'Unknown')}, "
                     f"Age: {self.patient_info.get('age', 'N/A')}")
        lines.append(f"Iteration: {self.iteration}/{self.max_iterations}")

        if self.plan:
            lines.append(f"\nPlan: {' → '.join(self.plan)}")

        if self.clinical_data:
            lines.append(f"\nClinical Data Loaded: NSS={self.clinical_data.get('nss_score', 'N/A')}/14, "
                         f"NDS={self.clinical_data.get('nds_score', 'N/A')}/23")

        if self.ml_results:
            lines.append(f"ML Prediction: class={self.ml_results.get('predicted_class')}, "
                         f"prob={self.ml_results.get('predicted_probability', 0):.2f}")

        if self.fusion_results:
            lines.append(f"Fusion Score: {self.fusion_results.get('fusion_score')}, "
                         f"Decision: {self.fusion_results.get('final_decision', 'N/A')}")

        if self.answers:
            lines.append(f"\nPatient Answers Collected: {len(self.answers)}")
            for k, v in list(self.answers.items())[-5:]:
                lines.append(f"  - {k}: {v}")

        if self.retrieved_memory:
            lines.append(f"\nRAG Memory Retrieved: {len(self.retrieved_memory)} records")
            for m in self.retrieved_memory[:2]:
                lines.append(f"  - {m.get('content', '')[:100]}...")

        recent_thoughts = self.intermediate_reasoning[-3:]
        if recent_thoughts:
            lines.append("\nRecent Reasoning:")
            for t in recent_thoughts:
                lines.append(f"  [{t['iteration']}] Thought: {t['thought'][:120]}")
                lines.append(f"       Action: {t['action'].get('name', 'none')}")

        return "\n".join(lines)
