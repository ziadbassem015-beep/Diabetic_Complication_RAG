import logging
from typing import Dict, Any, List
from core.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)

class DecisionRepository(BaseRepository):

    @staticmethod
    def save_final_decision(
        patient_id: str,
        ai_prediction: str,
        nds_score: int,
        nss_score: int,
        calculated_score: float,
        final_decision: str
    ) -> None:
        DecisionRepository.insert("final_diagnostic_decisions", {
            "patient_id": patient_id,
            "ai_prediction": ai_prediction,
            "nds_score": nds_score,
            "nss_score": nss_score,
            "calculated_score": calculated_score,
            "final_decision": final_decision
        })

    @staticmethod
    def get_latest_decision(patient_id: str) -> Dict[str, Any]:
        return DecisionRepository.select_latest_by_patient("final_diagnostic_decisions", patient_id)
    
    @staticmethod
    def get_decisions(patient_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        return DecisionRepository.select_many_by_patient("final_diagnostic_decisions", patient_id, limit)
