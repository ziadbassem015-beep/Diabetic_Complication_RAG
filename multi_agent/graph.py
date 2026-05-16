"""
multi_agent/graph.py — LangGraph-Style Stateful Graph Orchestrator.
"""
import logging
import uuid

logger = logging.getLogger(__name__)

"""

PHASE 1: Mandatory questionnaire (eligible questions from questionnaire.py)
PHASE 2a: ML inference + Fusion scoring (neuropathy PDN — unchanged)
PHASE 2b: Secondary assessments (gestational + heart risk)
PHASE 3: Reflection + Report generation
"""
import uuid
from .state import (
    MultiAgentState,
    NODE_PLANNER, NODE_MEMORY, NODE_REASONING, NODE_TOOL,
    NODE_ML, NODE_FUSION, NODE_SECONDARY, NODE_REFLECTION, NODE_REPORT,
    NODE_WAIT, NODE_END
)
from .memory import HybridMemory
from .agents import (
    PlannerAgent, MemoryRAGAgent, ClinicalReasoningAgent,
    ToolNode, MLInferenceAgent, FusionDecisionAgent,
    ReflectionAgent, ReportGeneratorAgent
)
from core.questionnaire import QUESTIONNAIRE, get_eligible_questions
from core.services.diagnostic_service import DiagnosticService

NODE_QUESTIONNAIRE = "questionnaire_node"

_SECTION_EMOJI = {
    "NSS": "🧠", "NDS": "🦾", "GUM": "🦷",
    "ULCER": "🩹", "ML": "📊",
    "GESTATIONAL": "🤰", "HEART_RISK": "❤️",
    "nss": "🧠", "nds": "🦾", "gum": "🦷",
    "ulcer": "🩹", "ml": "📊",
}


class DiagnosticGraph:
    """
    LangGraph-style stateful graph for the Multi-Agent Diagnostic System.

    Execution Phases (STRICT ORDER):
    ─────────────────────────────────────────────────────────────────
    Phase 1  [planner] → [memory] → [questionnaire] ×N eligible questions
    Phase 2a [ml_node] → [fusion_node]
    Phase 2b [secondary_assessment_node]
    Phase 3  [reflection] → [report] → END
    ─────────────────────────────────────────────────────────────────
    """

    def __init__(self, patient: dict, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.eligible_questions = get_eligible_questions(QUESTIONNAIRE, patient)

        # Initialize agents
        self.planner      = PlannerAgent()
        self.memory_agent = MemoryRAGAgent()
        self.reasoning    = ClinicalReasoningAgent()
        self.ml_agent     = MLInferenceAgent()
        self.fusion_agent = FusionDecisionAgent()
        self.reflector    = ReflectionAgent()
        self.reporter     = ReportGeneratorAgent()

        # Initialize shared state
        self.state = MultiAgentState(
            patient_id=patient["id"],
            patient_info={**patient, "session_id": self.session_id},
            current_node=NODE_PLANNER,
            next_node=NODE_PLANNER,
            max_iterations=len(self.eligible_questions) + 15,
        )

        if patient.get("gender") == "Male":
            self.state.skipped_assessments.append("gestational_diabetes")

        # Initialize memory system
        self.memory = HybridMemory(patient["id"], self.session_id)

    def _questionnaire_total(self) -> int:
        return len(self.eligible_questions)

    # ══════════════════════════════════════════════════════════════
    # QUESTIONNAIRE NODE — ask one eligible question at a time
    # ══════════════════════════════════════════════════════════════
    def _run_questionnaire_node(self):
        """Ask the next question in the eligible questionnaire list."""
        step = self.state.questionnaire_step
        total = self._questionnaire_total()

        if step >= total:
            self.state.emit("agent_start", "All questions collected. Running ML inference...", "graph")
            self.state.next_node = NODE_ML
            return

        q = self.eligible_questions[step]
        question_text = q["question"]
        options = q["options"]
        key = q["key"]
        section = q.get("section", "")
        section_emoji = _SECTION_EMOJI.get(section, "❓")

        self.state.emit(
            "agent_start",
            f"{section_emoji} Section: {section.upper()} — Question {step + 1}/{total}",
            "ClinicalReasoningAgent"
        )

        self.state.waiting_for_patient = True
        self.state.pending_question = {
            "question": f"{section_emoji} {question_text}",
            "options": [o["label"] for o in options],
            "score_map": {o["label"]: o["score"] for o in options},
            "key": key,
            "step": step,
            "section": section
        }
        self.state.next_node = NODE_WAIT

    # ══════════════════════════════════════════════════════════════
    # SECONDARY ASSESSMENT NODE — gestational + heart risk (no agent class)
    # ══════════════════════════════════════════════════════════════
    def _run_secondary_assessment_node(self):
        """Run secondary assessments after PDN fusion via DiagnosticService."""
        self.state.log("graph", "Running secondary assessments")
        self.state.emit(
            "agent_start",
            "Running gestational diabetes and cardiovascular risk assessments...",
            "graph",
        )

        outcome = DiagnosticService.run_secondary_assessments(
            self.state.patient_id,
            self.state.answers,
            self.state.patient_info,
        )

        self.state.gestational_results = outcome.get("gestational", {})
        self.state.heart_risk_results = outcome.get("heart_risk", {})
        for skipped in outcome.get("skipped_assessments", []):
            if skipped not in self.state.skipped_assessments:
                self.state.skipped_assessments.append(skipped)

        self.state.secondary_assessments_complete = True

        gd = self.state.gestational_results
        hr = self.state.heart_risk_results
        if gd.get("skipped"):
            self.state.emit("secondary", "Gestational diabetes: Not applicable (male patient)", "graph")
        elif gd.get("ml_result"):
            ml_gd = gd["ml_result"]
            self.state.emit(
                "secondary",
                f"Gestational diabetes: {ml_gd.get('risk_level', 'N/A')} risk "
                f"(prob={ml_gd.get('predicted_probability', 0):.1%})",
                "graph",
            )
        if hr.get("ml_result"):
            ml_hr = hr["ml_result"]
            self.state.emit(
                "secondary",
                f"Heart risk: {ml_hr.get('risk_level', 'N/A')} "
                f"(prob={ml_hr.get('predicted_probability', 0):.1%})",
                "graph",
            )

        self.state.log("graph", "Secondary assessments complete", outcome)
        self.state.next_node = NODE_REFLECTION

    # ── Node Execution Map ──────────────────────────────────────────
    def _run_node(self, node: str) -> str:
        """Execute a single graph node, return the next node."""
        self.state.current_node = node
        self.state.iteration += 1

        if node == NODE_PLANNER:
            self.state = self.planner.run(self.state)
            self.state.next_node = NODE_MEMORY

        elif node == NODE_MEMORY:
            self.state = self.memory_agent.run(self.state, self.memory)
            self.state.next_node = NODE_QUESTIONNAIRE

        elif node == NODE_QUESTIONNAIRE:
            self._run_questionnaire_node()

        elif node == NODE_ML:
            self.state = self.ml_agent.run(self.state)

        elif node == NODE_FUSION:
            self.state = self.fusion_agent.run(self.state)

        elif node == NODE_SECONDARY:
            self._run_secondary_assessment_node()

        elif node == NODE_REFLECTION:
            self.state = self.reflector.run(self.state)

        elif node == NODE_REPORT:
            self.state = self.reporter.run(self.state)

        elif node == NODE_WAIT:
            return NODE_WAIT

        return self.state.next_node

    # ── Conditional Edge Router ────────────────────────────────────
    def _route(self, current: str, next_proposed: str) -> str:
        """
        Hard routing rules — PHASE ORDER IS ENFORCED HERE.
        No LLM can skip a phase.
        """
        state = self.state
        total = self._questionnaire_total()

        if state.is_complete:
            return NODE_END

        if state.waiting_for_patient:
            return NODE_WAIT

        q_done = state.questionnaire_step >= total

        blocked_pre_questionnaire = (NODE_ML, NODE_FUSION, NODE_SECONDARY, NODE_REPORT, NODE_END)
        if not q_done and next_proposed in blocked_pre_questionnaire:
            return NODE_QUESTIONNAIRE

        if q_done and not state.has_ml_data() and next_proposed in (NODE_FUSION, NODE_SECONDARY, NODE_REPORT):
            return NODE_ML

        if state.has_ml_data() and not state.has_fusion() and next_proposed in (
            NODE_SECONDARY, NODE_REPORT, NODE_REFLECTION
        ):
            return NODE_FUSION

        if (
            state.has_fusion()
            and not state.secondary_assessments_complete
            and next_proposed in (NODE_REFLECTION, NODE_REPORT)
        ):
            return NODE_SECONDARY

        if (
            state.has_fusion()
            and state.secondary_assessments_complete
            and next_proposed not in (NODE_REFLECTION, NODE_REPORT, NODE_END)
        ):
            return NODE_REFLECTION

        return next_proposed

    # ═════════════════════════════════════════════════════════════
    # PUBLIC API — called by Streamlit
    # ═════════════════════════════════════════════════════════════

    def run_until_pause(self) -> list[dict]:
        """
        Run the graph until it pauses for patient input or reaches END.
        Returns list of stream events accumulated during this run.
        """
        self.state.stream_events.clear()
        current = self.state.next_node
        visited = 0

        while current not in (NODE_WAIT, NODE_END) and not self.state.is_complete:
            visited += 1
            if visited > 30:
                self.state.emit("warning", "Graph safety limit hit.", "graph")
                break

            next_node = self._run_node(current)
            next_node = self._route(current, next_node)
            current = next_node
            self.state.next_node = current

            if current == NODE_WAIT:
                break

        events = list(self.state.stream_events)
        self.state.stream_events.clear()
        return events

    def submit_patient_answer(self, key: str, answer: str) -> list[dict]:
        """
        Resume after patient answers. Advance questionnaire step, then continue.
        answer is the label string selected by the patient.
        """
        if not self.state.waiting_for_patient:
            return []

        score_map = self.state.pending_question.get("score_map", {})
        score_value = score_map.get(answer, answer)

        self.state.answers[key] = score_value
        self.state.add_message("user", f"{answer} (score: {score_value})")
        self.memory.save_short_term("user", answer)

        self.state.questionnaire_step += 1
        self.state.waiting_for_patient = False
        self.state.pending_question = {}

        total = self._questionnaire_total()
        if self.state.questionnaire_step < total:
            self.state.next_node = NODE_QUESTIONNAIRE
        else:
            self.state.emit("agent_start", "Questionnaire complete! Moving to analysis...", "graph")
            self.state.next_node = NODE_ML

        return self.run_until_pause()

    def initialize(self) -> list[dict]:
        """First call: plan → memory → start questionnaire."""
        self.state.stream_events.clear()
        self.state.next_node = NODE_PLANNER
        return self.run_until_pause()

    # ── Read-only Accessors ────────────────────────────────────────
    @property
    def is_complete(self) -> bool:
        return self.state.is_complete

    @property
    def is_waiting(self) -> bool:
        return self.state.waiting_for_patient

    @property
    def pending_question(self) -> dict:
        return self.state.pending_question

    @property
    def final_report(self) -> str:
        return self.state.final_report

    @property
    def confidence(self) -> float:
        return self.state.get_confidence()

    @property
    def progress(self) -> tuple[int, int]:
        """Returns (answered, total) for progress bar."""
        return self.state.questionnaire_step, self._questionnaire_total()

    @property
    def eligible_question_count(self) -> int:
        return self._questionnaire_total()

    def get_audit_log(self) -> list[dict]:
        return [
            {
                "iteration": e.iteration,
                "agent": e.agent,
                "action": e.action,
                "details": str(e.details)[:100]
            }
            for e in self.state.audit_log
        ]
