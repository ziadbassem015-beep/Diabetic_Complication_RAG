"""Secondary assessment pipeline and routing tests."""
from unittest.mock import patch

import pytest

from core.services.diagnostic_service import DiagnosticService
from multi_agent.state import NODE_FUSION, NODE_ML, NODE_REFLECTION, NODE_SECONDARY


def test_run_secondary_skips_gestational_for_male(male_patient, minimal_answers):
    with patch.object(
        DiagnosticService, "save_gestational_assessment"
    ) as mock_gd, patch.object(
        DiagnosticService, "save_heart_risk_assessment", return_value={"hr_scores": {}, "ml_result": {}}
    ) as mock_hr:
        result = DiagnosticService.run_secondary_assessments(
            male_patient["id"], minimal_answers, male_patient
        )

    mock_gd.assert_not_called()
    mock_hr.assert_called_once()
    assert result["gestational"].get("skipped") is True
    assert "gestational_diabetes" in result["skipped_assessments"]


def test_run_secondary_runs_gestational_for_female(female_patient, minimal_answers):
    with patch.object(
        DiagnosticService, "save_gestational_assessment", return_value={"ml_result": {}}
    ) as mock_gd, patch.object(
        DiagnosticService, "save_heart_risk_assessment", return_value={"ml_result": {}}
    ) as mock_hr:
        result = DiagnosticService.run_secondary_assessments(
            female_patient["id"], minimal_answers, female_patient
        )

    mock_gd.assert_called_once()
    mock_hr.assert_called_once()
    assert "gestational_diabetes" not in result.get("skipped_assessments", [])


def test_route_secondary_only_after_fusion(male_patient):
    from unittest.mock import patch

    with patch("multi_agent.graph.HybridMemory"):
        from multi_agent.graph import DiagnosticGraph

        graph = DiagnosticGraph(male_patient)
        state = graph.state

        state.questionnaire_step = graph.eligible_question_count
        state.ml_results = {"predicted_class": 0, "predicted_probability": 0.2}
        state.fusion_results = {}
        state.secondary_assessments_complete = False

        assert graph._route(NODE_ML, NODE_FUSION) == NODE_FUSION
        assert graph._route(NODE_FUSION, NODE_REFLECTION) == NODE_FUSION

        state.fusion_results = {"fusion_score": 1.0, "final_decision": "test"}
        assert graph._route(NODE_FUSION, NODE_REFLECTION) == NODE_SECONDARY

        state.secondary_assessments_complete = True
        assert graph._route(NODE_FUSION, NODE_REFLECTION) == NODE_REFLECTION


def test_route_blocks_secondary_before_fusion(male_patient):
    from unittest.mock import patch

    with patch("multi_agent.graph.HybridMemory"):
        from multi_agent.graph import DiagnosticGraph

        graph = DiagnosticGraph(male_patient)
        graph.state.questionnaire_step = graph.eligible_question_count
        graph.state.ml_results = {"predicted_class": 0, "predicted_probability": 0.3}
        graph.state.fusion_results = {}

        assert graph._route(NODE_ML, NODE_SECONDARY) == NODE_FUSION


@pytest.mark.parametrize("gender", ["Male", "Female"])
def test_heart_risk_always_invoked(gender, minimal_answers):
    patient = {"id": "x", "gender": gender, "age": 40}
    with patch.object(
        DiagnosticService, "save_gestational_assessment", return_value={}
    ), patch.object(
        DiagnosticService, "save_heart_risk_assessment", return_value={"ml_result": {}}
    ) as mock_hr:
        DiagnosticService.run_secondary_assessments("x", minimal_answers, patient)
    mock_hr.assert_called_once()
