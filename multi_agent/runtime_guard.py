"""
runtime_guard.py — Diagnostic wrappers for safe graph execution (observability only).

Does not modify business logic, routing, or agent behavior. Logs and re-raises errors.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, TypeVar

from .state import MultiAgentState

logger = logging.getLogger(__name__)

T = TypeVar("T")

REQUIRED_STATE_KEYS = {
    "ml_results": ("predicted_class",),
    "fusion_results": ("fusion_score", "final_decision"),
}


def safe_execute(
    node_name: str,
    fn: Callable[..., T],
    state: Optional[MultiAgentState] = None,
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Execute a node callable safely; log structured errors and re-raise.
    """
    patient_id = getattr(state, "patient_id", None) if state is not None else None
    try:
        result = fn(*args, **kwargs) if args or kwargs else fn()
        if result is None:
            logger.error({
                "event": "runtime_error",
                "node": node_name,
                "patient_id": patient_id,
                "error": "callable returned None",
                "error_type": "NoneReturn",
            })
        return result
    except Exception as exc:
        logger.error({
            "event": "runtime_error",
            "node": node_name,
            "patient_id": patient_id,
            "error": str(exc),
            "error_type": type(exc).__name__,
        })
        raise


def validate_state_keys(state: MultiAgentState, stage: str) -> List[str]:
    """Return list of missing required keys for a pipeline stage (read-only)."""
    issues: List[str] = []

    if stage in ("fusion", "secondary", "report"):
        for key, fields in (("ml_results", REQUIRED_STATE_KEYS["ml_results"]),):
            blob = getattr(state, key, None) or {}
            for field in fields:
                if blob.get(field) is None:
                    issues.append(f"{key}.{field} is missing")

    if stage in ("secondary", "report"):
        fusion = state.fusion_results or {}
        for field in REQUIRED_STATE_KEYS["fusion_results"]:
            if fusion.get(field) is None:
                issues.append(f"fusion_results.{field} is missing")

    if stage == "report":
        if not state.secondary_assessments_complete:
            issues.append("secondary_assessments_complete is False")

    return issues


def validate_node_transition(current: str, proposed: str, allowed: Optional[List[str]] = None) -> Optional[str]:
    """
    Validate a node transition. Returns an error message if invalid, else None.
    """
    if proposed is None:
        return f"transition from {current}: proposed node is None"
    if allowed is not None and proposed not in allowed:
        return f"invalid transition {current} -> {proposed}; allowed={allowed}"
    return None


def assert_pdn_isolated_from_secondary(
    fusion_before: dict,
    fusion_after: dict,
    gestational_results: dict,
    heart_risk_results: dict,
) -> List[str]:
    """Verify secondary assessments did not mutate PDN fusion state."""
    issues: List[str] = []
    if fusion_before != fusion_after:
        issues.append("fusion_results changed after secondary_assessment_node")
    for contaminant in ("gd_score", "hr_score", "gestational", "heart_risk"):
        if contaminant in fusion_after:
            issues.append(f"fusion_results contains non-PDN key: {contaminant}")
    if gestational_results and any(k in (fusion_before or {}) for k in gestational_results):
        issues.append("gestational data leaked into fusion_results keys")
    return issues
