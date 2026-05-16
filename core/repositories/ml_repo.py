import logging
from typing import Dict, Any, Optional
from core.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)

class MLRepository(BaseRepository):

    @staticmethod
    def get_ml_prediction(patient_id: str) -> Dict[str, Any]:
        return MLRepository.select_latest_by_patient("ml_neuropathy_predictions", patient_id)

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
        MLRepository.insert("ml_neuropathy_predictions", {
            "patient_id": patient_id,
            "nss_score": nss_score,
            "bmi_baseline": bmi_baseline,
            "age_baseline": age_baseline,
            "hba1c_baseline": hba1c_baseline,
            "heat_avg": heat_avg,
            "cold_avg": cold_avg,
            "predicted_class": predicted_class,
            "predicted_probability": predicted_probability
        })
