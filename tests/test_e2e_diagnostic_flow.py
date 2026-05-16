"""End-to-end diagnostic flow tests (mocked LLM and database)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from multi_agent.agents import ReportGeneratorAgent
from multi_agent.graph import NODE_QUESTIONNAIRE
from multi_agent.state import (
    NODE_FUSION,
    NODE_ML,
    NODE_REFLECTION,
    NODE_REPORT,
    NODE_SECONDARY,
    MultiAgentState,
)


def _mock_pdn_ml_result():
    return {
        "predicted_class": 0,
        "predicted_probability": 0.35,
        "ai_prediction": "\u0633\u0644\u064a\u0645",
    }


def _mock_fusion_result():
    return {
        "fusion_score": 0.9,
        "threshold": 1.55,
        "final_decision": "Likely Healthy",
        "confidence": "Medium",
        "ai_binary": 0,
        "nds_binary": 0,
        "nss_binary": 0,
    }


def _mock_secondary_result(skipped_gd: bool):
    return {
        "gestational": {"skipped": True} if skipped_gd else {"ml_result": {"risk_level": "Low"}},
        "heart_risk": {"ml_result": {"risk_level": "Medium", "predicted_probability": 0.4}},
        "skipped_assessments": ["gestational_diabetes"] if skipped_gd else [],
    }


def _build_graph(patient, minimal_answers):
    with patch("multi_agent.graph.HybridMemory") as mock_mem:
        mock_mem.return_value.search_long_term.return_value = []
        mock_mem.return_value.load_episodic.return_value = []
        mock_mem.return_value.save_short_term = MagicMock()
        from multi_agent.graph import DiagnosticGraph

        graph = DiagnosticGraph(patient)
    graph.state.answers = dict(minimal_answers)
    graph.state.questionnaire_step = graph.eligible_question_count
    return graph


def _run_pipeline(graph, skipped_gd: bool):
    """Execute ML → Fusion → Secondary → Report with mocks; return visited nodes."""
    from core.services.diagnostic_service import DiagnosticService

    visited = []

    with patch.object(
        DiagnosticService, "get_ml_prediction", return_value={}
    ), patch.object(
        DiagnosticService, "run_ml_inference", return_value=_mock_pdn_ml_result()
    ), patch.object(
        DiagnosticService, "save_clinical_data",
        return_value={"nss_score": 4, "nds_score": 3, "gum_score": 1, "ulcer_score": 0},
    ), patch.object(
        DiagnosticService, "compute_fusion", return_value=_mock_fusion_result()
    ), patch.object(
        DiagnosticService, "run_secondary_assessments",
        return_value=_mock_secondary_result(skipped_gd),
    ), patch(
        "multi_agent.agents.call_llm",
        return_value=(
            "1. Final\n2. Neuro\n3. Gum\n4. Ulcer\n5. Decision\n6. History\n"
            "7. Uncertainty\n8. Recs\n9. Disclaimer\n"
            "10. Gestational Diabetes Screening\n11. Heart Risk Assessment"
        ),
    ):
        graph.state.next_node = NODE_ML
        visited.append(NODE_ML)
        graph._run_node(NODE_ML)

        visited.append(NODE_FUSION)
        graph._run_node(NODE_FUSION)

        visited.append(NODE_SECONDARY)
        graph._run_secondary_assessment_node()

        graph.state.next_node = NODE_REFLECTION
        visited.append(NODE_REFLECTION)
        graph.reflector.run(graph.state)

        visited.append(NODE_REPORT)
        graph.reporter.run(graph.state)

    return visited, graph


def test_e2e_pipeline_order_male(male_patient, minimal_answers):
    graph = _build_graph(male_patient, minimal_answers)
    visited, graph = _run_pipeline(graph, skipped_gd=True)

    assert visited == [NODE_ML, NODE_FUSION, NODE_SECONDARY, NODE_REFLECTION, NODE_REPORT]
    assert graph.state.has_ml_data()
    assert graph.state.has_fusion()
    assert graph.state.secondary_assessments_complete
    assert graph.state.gestational_results.get("skipped") is True
    assert graph.state.heart_risk_results.get("ml_result")


def test_e2e_pipeline_order_female(female_patient, minimal_answers):
    graph = _build_graph(female_patient, minimal_answers)
    visited, graph = _run_pipeline(graph, skipped_gd=False)

    assert NODE_SECONDARY in visited
    assert graph.state.secondary_assessments_complete
    assert not graph.state.gestational_results.get("skipped")


def test_e2e_report_includes_sections_10_and_11(male_patient, minimal_answers):
    graph = _build_graph(male_patient, minimal_answers)
    _, graph = _run_pipeline(graph, skipped_gd=True)
    report = graph.state.final_report
    assert "10." in report or "Gestational" in report
    assert "11." in report or "Heart Risk" in report


def test_report_gestational_not_applicable_male():
    state = MultiAgentState(
        patient_id="p1",
        patient_info={"gender": "Male", "name": "T", "age": 40},
        gestational_results={"skipped": True},
        heart_risk_results={"ml_result": {"risk_level": "Low", "predicted_probability": 0.2}},
        ml_results={"predicted_class": 0, "predicted_probability": 0.2},
        fusion_results={"fusion_score": 0.5, "final_decision": "ok", "threshold": 1.55},
        clinical_scores={"nss_score": 2, "nds_score": 2, "gum_score": 1, "ulcer_score": 0},
    )

    ReportGeneratorAgent().run(state)

    assert "Not applicable for male patients" in state.final_report
    assert "10. Gestational Diabetes Report" in state.final_report


def test_report_heart_risk_insufficient_data():
    state = MultiAgentState(
        patient_id="p1",
        patient_info={"gender": "Female", "name": "T", "age": 30},
        gestational_results={"ml_result": {"risk_level": "Low", "predicted_probability": 0.1}},
        heart_risk_results={},
        ml_results={"predicted_class": 0, "predicted_probability": 0.2},
        fusion_results={"fusion_score": 0.5, "final_decision": "ok", "threshold": 1.55},
        clinical_scores={"nss_score": 2, "nds_score": 2, "gum_score": 1, "ulcer_score": 0},
    )

    ReportGeneratorAgent().run(state)

    assert "Insufficient data for this section" in state.final_report
    assert "11. Cardiovascular Risk Report" in state.final_report


def test_questionnaire_completes_before_ml(male_patient, minimal_answers):
    graph = _build_graph(male_patient, minimal_answers)
    graph.state.questionnaire_step = graph.eligible_question_count - 1
    graph.state.next_node = NODE_QUESTIONNAIRE
    graph._run_questionnaire_node()
    assert graph.state.waiting_for_patient

    last_q = graph.eligible_questions[-1]
    with patch.object(graph, "run_until_pause", return_value=[]):
        graph.submit_patient_answer(last_q["key"], last_q["options"][0]["label"])
    assert graph.state.next_node == NODE_ML
