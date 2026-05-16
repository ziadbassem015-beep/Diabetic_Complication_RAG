"""State validator unit tests."""
from multi_agent.state import MultiAgentState
from multi_agent.state_validator import (  # noqa: E402 — state only, not package graph export
    validate_before_fusion,
    validate_before_report,
    validate_before_secondary,
    validate_state,
)


def _base_state() -> MultiAgentState:
    return MultiAgentState(patient_id="p1", patient_info={"gender": "Male"})


def test_validate_before_fusion_requires_ml():
    state = _base_state()
    issues = validate_before_fusion(state)
    assert any("ml_results" in i for i in issues)

    state.ml_results = {"predicted_class": 0, "predicted_probability": 0.4}
    assert validate_before_fusion(state) == []


def test_validate_before_secondary_requires_fusion():
    state = _base_state()
    state.ml_results = {"predicted_class": 0, "predicted_probability": 0.4}
    issues = validate_before_secondary(state)
    assert any("fusion_results" in i for i in issues)

    state.fusion_results = {"fusion_score": 1.0, "final_decision": "ok"}
    assert validate_before_secondary(state) == []


def test_validate_before_report_requires_secondary_complete():
    state = _base_state()
    state.ml_results = {"predicted_class": 0, "predicted_probability": 0.4}
    state.fusion_results = {"fusion_score": 1.0, "final_decision": "ok"}
    issues = validate_before_report(state)
    assert any("secondary_assessments_complete" in i for i in issues)

    state.secondary_assessments_complete = True
    assert validate_before_report(state) == []


def test_validate_state_unknown_stage():
    assert "Unknown" in validate_state(_base_state(), "invalid")[0]
