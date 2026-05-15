import logging
from typing import Dict, Any, List
from core.database.client import get_supabase

logger = logging.getLogger(__name__)

class DecisionRepository:

    @staticmethod
    def save_final_decision(
        patient_id: str,
        ai_prediction: str,
        nds_score: int,
        nss_score: int,
        calculated_score: float,
        final_decision: str
    ) -> None:
        client = get_supabase()
        if not client: return
        try:
            client.table("final_diagnostic_decisions").insert({
                "patient_id": patient_id,
                "ai_prediction": ai_prediction,
                "nds_score": nds_score,
                "nss_score": nss_score,
                "calculated_score": calculated_score,
                "final_decision": final_decision
            }).execute()
        except Exception as e:
            logger.error(f"DecisionRepository.save_final_decision error: {e}")

    @staticmethod
    def get_latest_decision(patient_id: str) -> Dict[str, Any]:
        client = get_supabase()
        if not client: return {}
        try:
            res = client.table("final_diagnostic_decisions").select("*").eq("patient_id", patient_id).order("created_at", desc=True).limit(1).execute()
            return (res.data or [{}])[0]
        except Exception as e:
            logger.error(f"DecisionRepository.get_latest_decision error: {e}")
            return {}
    
    @staticmethod
    def get_decisions(patient_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        client = get_supabase()
        if not client: return []
        try:
            res = client.table("final_diagnostic_decisions").select("*").eq("patient_id", patient_id).order("created_at", desc=True).limit(limit).execute()
            return res.data or []
        except Exception as e:
            logger.error(f"DecisionRepository.get_decisions error: {e}")
            return []
