"""
heart_risk_repo.py — Cardiovascular Risk Assessment Data Access Layer
Follows BaseRepository pattern for clean architecture.
"""
import uuid
import logging
from typing import Dict, Any, Optional
from core.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class HeartRiskRepository(BaseRepository):
    """Manages cardiovascular risk assessment data in Supabase."""

    @staticmethod
    def save_heart_risk_assessment(
        patient_id: str,
        cholesterol: float,
        blood_pressure_systolic: float,
        blood_pressure_diastolic: float,
        resting_heart_rate: int,
        smoking_status: str,
        bmi: float,
        diabetes_duration: int,
        exercise_frequency: int,
        risk_score: float,
        predicted_class: int,
        predicted_probability: float,
    ) -> Dict[str, Any]:
        """Save heart risk assessment to database."""
        client = HeartRiskRepository._client()
        if not client:
            return {}
        try:
            # Map predicted_class to severity
            severity_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
            severity = severity_map.get(predicted_class, "Unknown Risk")

            payload = {
                "id": str(uuid.uuid4()),
                "patient_id": patient_id,
                "cholesterol": cholesterol,
                "blood_pressure_systolic": blood_pressure_systolic,
                "blood_pressure_diastolic": blood_pressure_diastolic,
                "resting_heart_rate": resting_heart_rate,
                "smoking_status": smoking_status,
                "bmi": bmi,
                "diabetes_duration": diabetes_duration,
                "exercise_frequency": exercise_frequency,
                "risk_score": risk_score,
                "predicted_class": predicted_class,
                "predicted_probability": predicted_probability,
                "severity": severity,
            }
            res = client.table("heart_risk_assessments").insert(payload).execute()
            result = (res.data or [{}])[0]
            logger.info(f"Heart risk assessment saved for patient {patient_id}")
            return result
        except Exception as e:
            logger.error(f"HeartRiskRepository.save_heart_risk_assessment error: {e}")
            return {}

    @staticmethod
    def get_heart_risk_assessment(patient_id: str) -> Dict[str, Any]:
        """Fetch the latest heart risk assessment for a patient."""
        client = HeartRiskRepository._client()
        if not client:
            return {}
        try:
            res = (
                client.table("heart_risk_assessments")
                .select("*")
                .eq("patient_id", patient_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            data = res.data or []
            return data[0] if data else {}
        except Exception as e:
            logger.error(f"HeartRiskRepository.get_heart_risk_assessment error: {e}")
            return {}

    @staticmethod
    def get_heart_risk_history(patient_id: str, limit: int = 5) -> list:
        """Fetch heart risk assessment history for a patient."""
        client = HeartRiskRepository._client()
        if not client:
            return []
        try:
            res = (
                client.table("heart_risk_assessments")
                .select("*")
                .eq("patient_id", patient_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return res.data or []
        except Exception as e:
            logger.error(f"HeartRiskRepository.get_heart_risk_history error: {e}")
            return []
