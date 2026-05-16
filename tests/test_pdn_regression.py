"""Regression tests: PDN outputs must remain stable for fixed inputs."""
from core.questionnaire import final_decision, ml_neuropathy_prediction


FIXED_ANSWERS = {
    "ml_bmi": 28,
    "ml_hba1c": 8,
    "ml_heat_right": 35,
    "ml_heat_left": 36,
    "ml_cold_right": 12,
    "ml_cold_left": 14,
}

FIXED_NSS = 6
FIXED_AGE = 52


def test_final_decision_regression_snapshot():
    expected = {
        "final_decision": "PDN Confirmed (اعتلال الأعصاب المحيطية السكري مؤكد)",
        "confidence": "Medium",
        "fusion_score": 2.1,
        "threshold": 1.55,
        "ai_binary": 1,
        "nds_binary": 1,
        "nss_binary": 1,
    }
    result = final_decision("مريض", 8, 6)
    assert result == expected


def test_final_decision_deterministic():
    r1 = final_decision("سليم", 3, 2)
    r2 = final_decision("سليم", 3, 2)
    assert r1 == r2


def test_ml_neuropathy_regression_snapshot():
    result = ml_neuropathy_prediction(FIXED_ANSWERS, FIXED_NSS, FIXED_AGE)
    assert result["predicted_class"] == 1
    assert result["predicted_probability"] == 0.615
    assert result["ai_prediction"] == "مريض"
    assert result["features"]["nss"] == FIXED_NSS
    assert result["features"]["age"] == FIXED_AGE


def test_ml_neuropathy_deterministic():
    r1 = ml_neuropathy_prediction(FIXED_ANSWERS, FIXED_NSS, FIXED_AGE)
    r2 = ml_neuropathy_prediction(FIXED_ANSWERS, FIXED_NSS, FIXED_AGE)
    assert r1 == r2
