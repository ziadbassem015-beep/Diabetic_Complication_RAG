"""
multi_agent/graph.py — LangGraph-Style Stateful Graph Orchestrator.

Nodes:  planner → memory → reasoning → tool → reflection → ml → fusion → report
Edges:  conditional routing based on MultiAgentState.next_node
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


class DiagnosticGraph:
    """
    LangGraph-style stateful graph for the Multi-Agent Diagnostic System.

    Graph Layout:
        [planner_node]
              ↓
        [memory_node]
              ↓
        [reasoning_node] ←────────────────┐
              ↓                            │
        [tool_node] → [patient_wait]       │
              ↓                            │
        [reflection_node] ─────────────────┘
              ↓ (when data sufficient)
        [ml_node]
              ↓
        [fusion_node]
              ↓
        [report_node]
              ↓
           [END]
    """

    MAX_REASONING_LOOPS = 6   # prevent runaway reasoning cycles

    def __init__(self, patient: dict, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())

        # Initialize agents
        self.planner        = PlannerAgent()
        self.memory_agent   = MemoryRAGAgent()
        self.reasoning      = ClinicalReasoningAgent()
        self.tool_node      = ToolNode()
        self.ml_agent       = MLInferenceAgent()
        self.fusion_agent   = FusionDecisionAgent()
        self.reflector      = ReflectionAgent()
        self.reporter       = ReportGeneratorAgent()

        # Initialize shared state
        self.state = MultiAgentState(
            patient_id=patient["id"],
            patient_info={**patient, "session_id": self.session_id},
            current_node=NODE_PLANNER,
            next_node=NODE_PLANNER,
        )

        # Initialize memory system
        self.memory = HybridMemory(patient["id"], self.session_id)

    # ── Node Execution Map ──────────────────────────────────────────
    def _run_node(self, node: str) -> str:
        """Execute a single graph node, return the next node."""
        self.state.current_node = node
        self.state.iteration += 1

        if node == NODE_PLANNER:
            self.state = self.planner.run(self.state)

        elif node == NODE_MEMORY:
            self.state = self.memory_agent.run(self.state, self.memory)

        elif node == NODE_REASONING:
            self.state.consecutive_reasoning += 1
            if self.state.consecutive_reasoning > self.MAX_REASONING_LOOPS:
                # Force advance: break reasoning loop
                self.state.emit("warning", "Reasoning loop limit reached — advancing to ML.", "graph")
                self.state.next_node = NODE_ML
                self.state.consecutive_reasoning = 0
            else:
                self.state = self.reasoning.run(self.state)

        elif node == NODE_TOOL:
            self.state.consecutive_reasoning = 0
            self.state = self.tool_node.run(self.state)

        elif node == NODE_ML:
            self.state.consecutive_reasoning = 0
            self.state = self.ml_agent.run(self.state)

        elif node == NODE_FUSION:
            self.state = self.fusion_agent.run(self.state)

        elif node == NODE_REFLECTION:
            self.state = self.reflector.run(self.state)

        elif node == NODE_REPORT:
            self.state = self.reporter.run(self.state)

        elif node == NODE_WAIT:
            # Graph pauses here — Streamlit will resume after patient answers
            return NODE_WAIT

        return self.state.next_node

    # ── Conditional Edge Router ────────────────────────────────────
    def _route(self, current: str, next_proposed: str) -> str:
        """Apply routing rules on top of agent-proposed next node."""
        state = self.state

        # Force completion if max iterations reached
        if state.iteration >= state.max_iterations:
            state.emit("warning", "Max iterations reached — forcing report.", "graph")
            return NODE_REPORT

        # Safety: if done, stay done
        if state.is_complete:
            return NODE_END

        # If patient is waiting, do not advance
        if state.waiting_for_patient:
            return NODE_WAIT

        # If fusion is computed, always go to report
        if state.has_fusion() and next_proposed not in (NODE_REPORT, NODE_END):
            return NODE_REPORT

        return next_proposed

    # ═════════════════════════════════════════════════════════════
    # PUBLIC API — called by Streamlit
    # ═════════════════════════════════════════════════════════════

    def run_until_pause(self) -> list[dict]:
        """
        Run the graph until it pauses for patient input or reaches END.
        Returns list of stream events accumulated during this run.

        Call this once to kick off the session, then again after each patient answer.
        """
        self.state.stream_events.clear()

        current = self.state.next_node
        visited_since_pause = 0

        while current not in (NODE_WAIT, NODE_END) and not self.state.is_complete:
            visited_since_pause += 1
            if visited_since_pause > 20:
                # Hard limit — prevent infinite loops
                self.state.emit("error", "Graph safety limit hit.", "graph")
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
        Resume the graph after patient answers a question.
        Stores the answer in state, clears waiting flag, and runs again.
        """
        if not self.state.waiting_for_patient:
            return []

        # Record answer
        self.state.answers[key] = answer
        self.state.add_message("user", answer)

        # Save to memory
        self.memory.save_short_term("user", answer)

        # Resume graph
        self.state.waiting_for_patient = False
        self.state.pending_question = {}
        self.state.next_node = NODE_REFLECTION  # always reflect after patient input

        return self.run_until_pause()

    def initialize(self) -> list[dict]:
        """
        First call: run planning + memory load, then start reasoning.
        """
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
