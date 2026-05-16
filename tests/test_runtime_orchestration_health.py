"""
Runtime orchestration health checks — full in-memory graph execution.

Validates male/female flows under simulated real execution paths (no DB, no UI).
"""
from __future__ import annotations

import pytest

from multi_agent.runtime_health import RuntimeHealthResult, simulate_full_run
from multi_agent.state import NODE_FUSION, NODE_ML, NODE_REPORT, NODE_SECONDARY


@pytest.fixture
def runtime_male_patient():
    return {
        "id": "runtime-male-0001",
        "name": "Runtime Male",
        "age": 45,
        "gender": "Male",
        "diabetes_duration": 5,
    }


@pytest.fixture
def runtime_female_patient():
    return {
        "id": "runtime-female-0001",
        "name": "Runtime Female",
        "age": 30,
        "gender": "Female",
        "diabetes_duration": 2,
    }


def _assert_core_pipeline(result: RuntimeHealthResult) -> None:
    assert NODE_ML in result.nodes_executed
    assert NODE_FUSION in result.nodes_executed
    assert NODE_SECONDARY in result.nodes_executed
    assert NODE_REPORT in result.nodes_executed
    assert result.secondary_ran
    assert result.report_sections_ok
    assert result.pdn_isolated


class TestScenarioAMalePatient:
    """Scenario A — Male patient full graph execution."""

    def test_male_flow_passes(self, runtime_male_patient):
        result = simulate_full_run(runtime_male_patient)
        assert result.passed, result.errors

    def test_no_gestational_questions(self, runtime_male_patient):
        result = simulate_full_run(runtime_male_patient)
        assert "GESTATIONAL" not in result.sections_seen
        assert result.gestational_skipped

    def test_ml_fusion_heart_risk_report_order(self, runtime_male_patient):
        result = simulate_full_run(runtime_male_patient)
        _assert_core_pipeline(result)
        ml_i = result.nodes_executed.index(NODE_ML)
        fusion_i = result.nodes_executed.index(NODE_FUSION)
        secondary_i = result.nodes_executed.index(NODE_SECONDARY)
        report_i = result.nodes_executed.index(NODE_REPORT)
        assert ml_i < fusion_i < secondary_i < report_i


class TestScenarioBFemalePatient:
    """Scenario B — Female patient full graph execution."""

    def test_female_flow_passes(self, runtime_female_patient):
        result = simulate_full_run(runtime_female_patient)
        assert result.passed, result.errors

    def test_gestational_included(self, runtime_female_patient):
        result = simulate_full_run(runtime_female_patient)
        assert "GESTATIONAL" in result.sections_seen
        assert not result.gestational_skipped

    def test_secondary_and_report_sections(self, runtime_female_patient):
        result = simulate_full_run(runtime_female_patient)
        _assert_core_pipeline(result)
        assert NODE_SECONDARY in result.nodes_executed
