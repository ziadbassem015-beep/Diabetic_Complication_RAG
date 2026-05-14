"""
multi_agent/graph.py — LangGraph-Style Stateful Graph Orchestrator.

PHASE 1: Mandatory questionnaire (all questions from questionnaire.py)
PHASE 2: ML inference + Fusion scoring
PHASE 3: Reflection + Report generation
"""
import uuid
from .state import (
    MultiAgentState,
    NODE_PLANNER, NODE_MEMORY, NODE_REASONING, NODE_TOOL,
    NODE_ML, NODE_FUSION, NODE_REFLECTION, NODE_REPORT,
    NODE_WAIT, NODE_END
)
from .memory import HybridMemory
from .agents import (
    PlannerAgent, MemoryRAGAgent, ClinicalReasoningAgent,
    ToolNode, MLInferenceAgent, FusionDecisionAgent,
    ReflectionAgent, ReportGeneratorAgent
)
from core.questionnaire import QUESTIONNAIRE  # the full list of questions

NODE_QUESTIONNAIRE = "questionnaire_node"


class DiagnosticGraph:
    """
    LangGraph-style stateful graph for the Multi-Agent Diagnostic System.

    Execution Phases (STRICT ORDER):
    ─────────────────────────────────────────────────────────────────
    Phase 1  [planner] → [memory] → [questionnaire] ×N questions
    Phase 2  [ml_node] → [fusion_node]
    Phase 3  [reflection] → [report] → END
    ─────────────────────────────────────────────────────────────────
    The LLM reasoning agent is used only for reflection.
    The questionnaire is driven by the fixed question list from questionnaire.py.
    """

    def __init__(self, patient: dict, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())

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
            max_iterations=len(QUESTIONNAIRE) + 15,  # enough for all questions + phases
        )

        # Initialize memory system
        self.memory = HybridMemory(patient["id"], self.session_id)

    # ══════════════════════════════════════════════════════════════
    # QUESTIONNAIRE NODE — ask one question at a time
    # ══════════════════════════════════════════════════════════════
    def _run_questionnaire_node(self):
        """Ask the next question in the questionnaire list."""
        step = self.state.questionnaire_step

        if step >= len(QUESTIONNAIRE):
            # All questions answered — advance to ML
            self.state.emit("agent_start", "All questions collected. Running ML inference...", "graph")
            self.state.next_node = NODE_ML
            return

        q = QUESTIONNAIRE[step]
        question_text = q["question"]
        options = q["options"]
        key = q["key"]
        section = q.get("section", "")
        section_emoji = {
            "NSS": "🧠", "NDS": "🦾", "GUM": "🦷",
            "ULCER": "🩹", "ML": "📊",
            "nss": "🧠", "nds": "🦾", "gum": "🦷",
            "ulcer": "🩹", "ml": "📊"
        }.get(section, "❓")

        self.state.emit(
            "agent_start",
            f"{section_emoji} Section: {section.upper()} — Question {step + 1}/{len(QUESTIONNAIRE)}",
            "ClinicalReasoningAgent"
        )

        # Set waiting state
        self.state.waiting_for_patient = True
        self.state.pending_question = {
            "question": f"{section_emoji} {question_text}",
            "options": [o["label"] for o in options],       # strings only for UI
            "score_map": {o["label"]: o["score"] for o in options},  # label → score
            "key": key,
            "step": step,
            "section": section
        }
        self.state.next_node = NODE_WAIT

    # ── Node Execution Map ──────────────────────────────────────────
    def _run_node(self, node: str) -> str:
        """Execute a single graph node, return the next node."""
        self.state.current_node = node
        self.state.iteration += 1

        if node == NODE_PLANNER:
            self.state = self.planner.run(self.state)
            self.state.next_node = NODE_MEMORY  # always go to memory after planning

        elif node == NODE_MEMORY:
            self.state = self.memory_agent.run(self.state, self.memory)
            self.state.next_node = NODE_QUESTIONNAIRE  # always start questionnaire after memory

        elif node == NODE_QUESTIONNAIRE:
            self._run_questionnaire_node()

        elif node == NODE_ML:
            self.state = self.ml_agent.run(self.state)

        elif node == NODE_FUSION:
            self.state = self.fusion_agent.run(self.state)

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

        # Always stop if complete
        if state.is_complete:
            return NODE_END

        # Pause for patient input
        if state.waiting_for_patient:
            return NODE_WAIT

        # ── Phase 1: Questionnaire must complete before ML ──────────
        # If questionnaire not done, never allow ML/Fusion/Report
        q_done = state.questionnaire_step >= len(QUESTIONNAIRE)

        if not q_done and next_proposed in (NODE_ML, NODE_FUSION, NODE_REPORT, NODE_END):
            return NODE_QUESTIONNAIRE

        # ── Phase 2: ML must run before Fusion ─────────────────────
        if q_done and not state.has_ml_data() and next_proposed in (NODE_FUSION, NODE_REPORT):
            return NODE_ML

        # ── Phase 3: Fusion must run before Report ──────────────────
        if state.has_ml_data() and not state.has_fusion() and next_proposed == NODE_REPORT:
            return NODE_FUSION

        # ── After Fusion → Reflection → Report ─────────────────────
        if state.has_fusion() and next_proposed not in (NODE_REFLECTION, NODE_REPORT, NODE_END):
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
            if visited > 25:
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

        # Look up numeric score from score_map (for calculate_section_scores)
        score_map = self.state.pending_question.get("score_map", {})
        score_value = score_map.get(answer, answer)  # fallback to answer itself if not found

        # Store numeric score in answers (used for scoring)
        self.state.answers[key] = score_value
        self.state.add_message("user", f"{answer} (score: {score_value})")
        self.memory.save_short_term("user", answer)

        # Advance questionnaire step
        self.state.questionnaire_step += 1
        self.state.waiting_for_patient = False
        self.state.pending_question = {}

        # Route: more questions or ML phase
        if self.state.questionnaire_step < len(QUESTIONNAIRE):
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
        return self.state.questionnaire_step, len(QUESTIONNAIRE)

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
