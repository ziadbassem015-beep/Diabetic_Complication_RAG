import logging
from typing import Dict, Any, List
from core.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)

class ClinicalRepository(BaseRepository):

    @staticmethod
    def get_clinical_data(patient_id: str) -> Dict[str, Any]:
        nss = ClinicalRepository.select_latest_by_patient("nss_assessments", patient_id)
        nds = ClinicalRepository.select_latest_by_patient("nds_assessments", patient_id)
        return {
            "nss": nss or {},
            "nds": nds or {}
        }

    @staticmethod
    def save_nss_assessment(patient_id: str, total_score: int) -> None:
        ClinicalRepository.insert("nss_assessments", {"patient_id": patient_id, "total_score": total_score})

    @staticmethod
    def save_nds_assessment(patient_id: str, total_score: int) -> None:
        ClinicalRepository.insert("nds_assessments", {"patient_id": patient_id, "total_score": total_score})

    @staticmethod
    def save_gum_assessment(patient_id: str, total_score: int) -> None:
        ClinicalRepository.insert("gum_assessments", {"patient_id": patient_id, "total_score": total_score})

    @staticmethod
    def save_ulcer_assessment(patient_id: str, ulcer_stage: int) -> None:
        ClinicalRepository.insert("ulcer_assessments", {"patient_id": patient_id, "ulcer_stage": ulcer_stage})
