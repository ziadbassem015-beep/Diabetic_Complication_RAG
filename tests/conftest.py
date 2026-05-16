"""Shared pytest fixtures and import path setup."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _mock_openai_module():
    """Avoid OpenAI import errors when multi_agent loads rag_engine."""
    mock_openai = MagicMock()
    with patch.dict(sys.modules, {"openai": mock_openai}):
        yield


@pytest.fixture
def male_patient() -> Dict[str, Any]:
    return {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "Test Male",
        "age": 45,
        "gender": "Male",
        "diabetes_duration": 5,
    }


@pytest.fixture
def female_patient() -> Dict[str, Any]:
    return {
        "id": "00000000-0000-0000-0000-000000000002",
        "name": "Test Female",
        "age": 30,
        "gender": "Female",
        "diabetes_duration": 2,
    }


@pytest.fixture
def minimal_answers() -> Dict[str, Any]:
    """Minimal numeric answers for scoring and ML heuristics."""
    return {
        "nss_burning_pain": 1,
        "nss_burning_night": 0,
        "nss_burning_severe": 0,
        "nss_numbness": 1,
        "nss_numbness_night": 0,
        "nss_tingling": 1,
        "nss_fatigue": 0,
        "nss_paresthesia": 0,
        "nss_allodynia": 0,
        "nds_vibration": 2,
        "nds_temp_right_warm": 0,
        "nds_temp_right_cold": 0,
        "nds_pain": 0,
        "nds_touch": 0,
        "nds_reflex": 0,
        "gum_bleeding": 1,
        "gum_swelling": 0,
        "gum_recession": 0,
        "gum_bad_breath": 0,
        "gum_loose_teeth": 0,
        "ulcer_history": 0,
        "ulcer_active": 0,
        "ulcer_healing": 0,
        "ulcer_infection": 0,
        "ml_bmi": 24,
        "ml_hba1c": 7,
        "ml_heat_right": 38,
        "ml_heat_left": 38,
        "ml_cold_right": 20,
        "ml_cold_left": 20,
        "gd_pregnancy_week": 1,
        "gd_fasting_glucose": 0,
        "gd_family_history": 0,
        "gd_bmi": 1,
        "hr_cholesterol": 0,
        "hr_blood_pressure": 0,
        "hr_resting_heart_rate": 0,
        "hr_smoking_status": 0,
        "hr_exercise_frequency": 0,
    }
