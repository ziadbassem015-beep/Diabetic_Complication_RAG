"""
runtime_health.py — In-memory full-graph simulation for runtime stability audits.

No Streamlit, no database writes. Uses real PDN logic with repository saves mocked out.
"""
from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from core.services.diagnostic_service import DiagnosticService

from .runtime_guard import (
    assert_pdn_isolated_from_secondary,
    safe_execute,
    validate_state_keys,
)
from .graph import NODE_QUESTIONNAIRE
from .state import (
    NODE_END,
    NODE_FUSION,
    NODE_MEMORY,
    NODE_ML,
    NODE_PLANNER,
    NODE_REFLECTION,
    NODE_REPORT,
    NODE_SECONDARY,
    NODE_WAIT,
)


@dataclass
class RuntimeHealthResult:
    scenario: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    nodes_executed: List[str] = field(default_factory=list)
    sections_seen: List[str] = field(default_factory=list)
    gestational_skipped: bool = False
    secondary_ran: bool = False
    report_sections_ok: bool = False
    pdn_isolated: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "passed": self.passed,
            "errors": self.errors,
            "nodes_executed": self.nodes_executed,
            "sections_seen": self.sections_seen,
            "gestational_skipped": self.gestational_skipped,
            "secondary_ran": self.secondary_ran,
            "report_sections_ok": self.report_sections_ok,
            "pdn_isolated": self.pdn_isolated,
        }


def _llm_side_effect(prompt: str) -> str:
    if "Planner" in prompt or "diagnostic plan" in prompt.lower():
        return json.dumps([
            "Load memory",
            "Collect questionnaire",
            "Run ML and fusion",
            "Secondary assessments",
            "Generate report",
        ])
    if "Reflection" in prompt or "is_consistent" in prompt:
        return json.dumps({
            "is_consistent": True,
            "issues": [],
            "suggestions": [],
            "should_replan": False,
            "confidence_valid": True,
        })
    return (
        "1. Final Diagnosis\n2. Neuropathy\n3. Gum\n4. Ulcer\n5. Decision\n"
        "6. History\n7. Uncertainty\n8. Recommendations\n9. Disclaimer\n"
        "10. Gestational Diabetes Screening\n11. Heart Risk Assessment"
    )


def _enter_runtime_patches(stack: ExitStack):
    stack.enter_context(patch.object(DiagnosticService, "store_memory", MagicMock()))
    stack.enter_context(
        patch("core.repositories.clinical_repo.ClinicalRepository.save_nss_assessment", return_value={})
    )
    stack.enter_context(
        patch("core.repositories.clinical_repo.ClinicalRepository.save_nds_assessment", return_value={})
    )
    stack.enter_context(
        patch("core.repositories.clinical_repo.ClinicalRepository.save_gum_assessment", return_value={})
    )
    stack.enter_context(
        patch("core.repositories.clinical_repo.ClinicalRepository.save_ulcer_assessment", return_value={})
    )
    stack.enter_context(
        patch("core.repositories.ml_repo.MLRepository.save_ml_prediction", return_value={})
    )
    stack.enter_context(
        patch("core.repositories.decision_repo.DecisionRepository.save_final_decision", return_value={})
    )
    stack.enter_context(
        patch("core.repositories.gestational_repo.GestationalRepository.save_gestational_assessment", return_value={})
    )
    stack.enter_context(
        patch("core.repositories.heart_risk_repo.HeartRiskRepository.save_heart_risk_assessment", return_value={})
    )
    stack.enter_context(
        patch("core.repositories.memory_repo.MemoryRepository.save_conversation_memory", MagicMock())
    )
    stack.enter_context(
        patch("core.repositories.memory_repo.MemoryRepository.match_memory", return_value=[])
    )
    stack.enter_context(
        patch("core.repositories.decision_repo.DecisionRepository.get_decisions", return_value=[])
    )
    stack.enter_context(patch.object(DiagnosticService, "retrieve_memory", return_value=[]))
    stack.enter_context(patch.object(DiagnosticService, "get_ml_prediction", return_value={}))
    stack.enter_context(patch("multi_agent.agents.call_llm", side_effect=_llm_side_effect))
    mock_mem = stack.enter_context(patch("multi_agent.graph.HybridMemory"))
    mock_mem.return_value.search_long_term.return_value = []
    mock_mem.return_value.load_episodic.return_value = []
    mock_mem.return_value.save_short_term = MagicMock()
    return mock_mem


def _assert_pipeline_order(post_questionnaire_nodes: List[str], errors: List[str]) -> None:
    required = [NODE_ML, NODE_FUSION, NODE_SECONDARY, NODE_REFLECTION, NODE_REPORT]
    indices = []
    for node in required:
        if node not in post_questionnaire_nodes:
            errors.append(f"Missing node in pipeline: {node}")
            return
        indices.append(post_questionnaire_nodes.index(node))
    if indices != sorted(indices):
        errors.append(
            f"Node order violation: expected ML→Fusion→Secondary→Reflection→Report, got {post_questionnaire_nodes}"
        )


def simulate_full_run(patient: Dict[str, Any]) -> RuntimeHealthResult:
    """Run the full DiagnosticGraph in-memory for one patient profile."""
    scenario = f"{patient.get('gender', 'Unknown')}_patient"
    result = RuntimeHealthResult(scenario=scenario, passed=True)
    nodes_executed: List[str] = []
    fusion_snapshot: Dict[str, Any] = {}
    fusion_before_secondary: Dict[str, Any] = {}

    try:
        with ExitStack() as stack:
            _enter_runtime_patches(stack)
            from multi_agent.graph import DiagnosticGraph

            graph = DiagnosticGraph(patient)
            original_run_node = graph._run_node

            def instrumented_run_node(node: str) -> str:
                nodes_executed.append(node)
                out = safe_execute(node, lambda: original_run_node(node), graph.state)
                if node == NODE_FUSION and graph.state.has_fusion():
                    fusion_before_secondary.update(dict(graph.state.fusion_results))
                return out

            graph._run_node = instrumented_run_node  # type: ignore[method-assign]

            safe_execute("initialize", graph.initialize, graph.state)

            max_steps = graph.eligible_question_count + 40
            steps = 0
            while not graph.is_complete and steps < max_steps:
                steps += 1
                if graph.is_waiting:
                    pq = graph.pending_question
                    section = pq.get("section", "")
                    result.sections_seen.append(section)
                    if section == "GESTATIONAL" and patient.get("gender") == "Male":
                        result.errors.append("GESTATIONAL question shown to male patient")
                    key = pq.get("key", "")
                    label = pq["options"][0] if pq.get("options") else ""
                    safe_execute(
                        "submit_answer",
                        lambda k=key, lbl=label: graph.submit_patient_answer(k, lbl),
                        graph.state,
                    )
                else:
                    safe_execute("run_until_pause", graph.run_until_pause, graph.state)

            fusion_snapshot = dict(graph.state.fusion_results) if graph.state.has_fusion() else {}

            result.nodes_executed = nodes_executed
            result.gestational_skipped = (
                patient.get("gender") == "Male"
                or graph.state.gestational_results.get("skipped") is True
            )
            result.secondary_ran = NODE_SECONDARY in nodes_executed

            eligible_sections = {q.get("section") for q in graph.eligible_questions}
            if patient.get("gender") == "Male" and "GESTATIONAL" in eligible_sections:
                result.errors.append("GESTATIONAL in eligible_questions for male")
            if patient.get("gender") == "Female" and "GESTATIONAL" not in eligible_sections:
                result.errors.append("GESTATIONAL missing from eligible_questions for female")

            result.errors.extend(validate_state_keys(graph.state, "report"))

            pdn_issues = assert_pdn_isolated_from_secondary(
                fusion_before_secondary or fusion_snapshot,
                dict(graph.state.fusion_results),
                graph.state.gestational_results,
                graph.state.heart_risk_results,
            )
            if pdn_issues:
                result.errors.extend(pdn_issues)
                result.pdn_isolated = False

            report = graph.final_report or ""
            has_s10 = "10" in report and "Gestational" in report
            has_s11 = "11" in report and "Heart Risk" in report
            result.report_sections_ok = has_s10 and has_s11
            if not result.report_sections_ok:
                result.errors.append("Report missing sections 10 and/or 11")

            if patient.get("gender") == "Female":
                if graph.state.gestational_results.get("skipped"):
                    result.errors.append("gestational incorrectly skipped for female")
                if not graph.state.gestational_results.get("ml_result") and not graph.state.gestational_results.get("gd_scores"):
                    result.errors.append("gestational_results not populated for female")
            elif not result.gestational_skipped:
                result.errors.append("gestational should be skipped for male")

            if not graph.state.heart_risk_results:
                result.errors.append("heart_risk_results empty")

            post_q = [
                n for n in nodes_executed
                if n not in (NODE_PLANNER, NODE_MEMORY, NODE_QUESTIONNAIRE, NODE_WAIT)
            ]
            _assert_pipeline_order(post_q, result.errors)

            if not graph.is_complete:
                result.errors.append("Graph did not reach is_complete")

    except Exception as exc:
        result.errors.append(f"Runtime exception: {type(exc).__name__}: {exc}")

    result.passed = len(result.errors) == 0
    return result
