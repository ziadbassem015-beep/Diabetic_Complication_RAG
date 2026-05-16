"""
state_validator.py — Read-only validation of MultiAgentState before pipeline stages.

Validates required keys only; does not modify state or routing.
"""
from __future__ import annotations

from typing import List

from .state import MultiAgentState


def _missing_ml_results(state: MultiAgentState) -> List[str]:
    issues: List[str] = []
    if state.ml_results.get("predicted_class") is None:
        issues.append("ml_results.predicted_class is required")
    if state.ml_results.get("predicted_probability") is None:
        issues.append("ml_results.predicted_probability is required")
    return issues


def _missing_fusion_results(state: MultiAgentState) -> List[str]:
    issues: List[str] = []
    if state.fusion_results.get("fusion_score") is None:
        issues.append("fusion_results.fusion_score is required")
    if not state.fusion_results.get("final_decision"):
        issues.append("fusion_results.final_decision is required")
    return issues


def validate_before_fusion(state: MultiAgentState) -> List[str]:
    """Require ML neuropathy results before fusion."""
    return _missing_ml_results(state)


def validate_before_secondary(state: MultiAgentState) -> List[str]:
    """Require fusion results before secondary assessments."""
    return _missing_fusion_results(state)


def validate_before_report(state: MultiAgentState) -> List[str]:
    """Require ML, fusion, and completed secondary assessments before report."""
    issues = _missing_ml_results(state) + _missing_fusion_results(state)
    if not state.secondary_assessments_complete:
        issues.append("secondary_assessments_complete must be True")
    return issues


def validate_state(state: MultiAgentState, stage: str) -> List[str]:
    """
    Validate state for a named pipeline stage.

    stage: 'fusion' | 'secondary' | 'report'
    """
    validators = {
        "fusion": validate_before_fusion,
        "secondary": validate_before_secondary,
        "report": validate_before_report,
    }
    fn = validators.get(stage)
    if fn is None:
        return [f"Unknown validation stage: {stage}"]
    return fn(state)
