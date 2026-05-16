"""
gestational_repo.py — Gestational Diabetes Assessment Data Access Layer
Follows BaseRepository pattern for clean architecture.
"""
import uuid
import logging
from typing import Dict, Any, Optional
from core.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class GestationalRepository(BaseRepository):
    """Manages gestational diabetes assessment data in Supabase."""

    @staticmethod
    def save_gestational_assessment(
        patient_id: str,
        pregnancy_week: int,
        glucose_level: float,
        fasting_glucose: float,
        insulin_resistance: float,
        bmi: float,
        family_history: bool,
        risk_score: float,
        predicted_class: int,
        predicted_probability: float,
    ) -> Dict[str, Any]:
        """Save gestational diabetes assessment to database."""
        client = GestationalRepository._client()
        if not client:
            return {}
        try:
            payload = {
                "id": str(uuid.uuid4()),
                "patient_id": patient_id,
                "pregnancy_week": pregnancy_week,
                "glucose_level": glucose_level,
                "fasting_glucose": fasting_glucose,
                "insulin_resistance": insulin_resistance,
                "bmi": bmi,
                "family_history": family_history,
                "risk_score": risk_score,
                "predicted_class": predicted_class,
                "predicted_probability": predicted_probability,
                "severity": "High Risk" if predicted_class == 1 else "Low Risk",
            }
            res = client.table("gestational_diabetes_assessments").insert(payload).execute()
            result = (res.data or [{}])[0]
            logger.info(f"Gestational diabetes assessment saved for patient {patient_id}")
            return result
        except Exception as e:
            logger.error(f"GestationalRepository.save_gestational_assessment error: {e}")
            return {}

    @staticmethod
    def get_gestational_assessment(patient_id: str) -> Dict[str, Any]:
        """Fetch the latest gestational diabetes assessment for a patient."""
        client = GestationalRepository._client()
        if not client:
            return {}
        try:
            res = (
                client.table("gestational_diabetes_assessments")
                .select("*")
                .eq("patient_id", patient_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            data = res.data or []
            return data[0] if data else {}
        except Exception as e:
            logger.error(f"GestationalRepository.get_gestational_assessment error: {e}")
            return {}

    @staticmethod
    def get_gestational_history(patient_id: str, limit: int = 5) -> list:
        """Fetch gestational diabetes assessment history for a patient."""
        client = GestationalRepository._client()
        if not client:
            return []
        try:
            res = (
                client.table("gestational_diabetes_assessments")
                .select("*")
                .eq("patient_id", patient_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return res.data or []
        except Exception as e:
            logger.error(f"GestationalRepository.get_gestational_history error: {e}")
            return []
