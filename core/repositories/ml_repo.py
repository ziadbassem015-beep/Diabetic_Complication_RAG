import logging
from typing import Dict, Any, Optional
from core.database.client import get_supabase

logger = logging.getLogger(__name__)

class MLRepository:

    @staticmethod
    def get_ml_prediction(patient_id: str) -> Dict[str, Any]:
        client = get_supabase()
        if not client:
            return {}
        try:
            res = client.table("ml_neuropathy_predictions").select("*").eq("patient_id", patient_id).order("created_at", desc=True).limit(1).execute()
            return (res.data or [{}])[0]
        except Exception as e:
            logger.error(f"MLRepository.get_ml_prediction error: {e}")
            return {}

    @staticmethod
    def save_ml_prediction(
        patient_id: str,
        nss_score: int,
        bmi_baseline: float,
        age_baseline: int,
        hba1c_baseline: float,
        heat_avg: float,
        cold_avg: float,
        predicted_class: int,
        predicted_probability: float
    ) -> None:
        client = get_supabase()
        if not client: return
        try:
            client.table("ml_neuropathy_predictions").insert({
                "patient_id": patient_id,
                "nss_score": nss_score,
                "bmi_baseline": bmi_baseline,
                "age_baseline": age_baseline,
                "hba1c_baseline": hba1c_baseline,
                "heat_avg": heat_avg,
                "cold_avg": cold_avg,
                "predicted_class": predicted_class,
                "predicted_probability": predicted_probability
            }).execute()
        except Exception as e:
            logger.error(f"MLRepository.save_ml_prediction error: {e}")
