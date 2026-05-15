import logging
from typing import Dict, Any, List
from core.database.client import get_supabase

logger = logging.getLogger(__name__)

class ClinicalRepository:

    @staticmethod
    def get_clinical_data(patient_id: str) -> Dict[str, Any]:
        client = get_supabase()
        if not client:
            return {"nss": {}, "nds": {}}
        try:
            nss = client.table("nss_assessments").select("*").eq("patient_id", patient_id).order("created_at", desc=True).limit(1).execute()
            nds = client.table("nds_assessments").select("*").eq("patient_id", patient_id).order("created_at", desc=True).limit(1).execute()
            return {
                "nss": (nss.data or [{}])[0],
                "nds": (nds.data or [{}])[0]
            }
        except Exception as e:
            logger.error(f"ClinicalRepository.get_clinical_data error: {e}")
            return {"nss": {}, "nds": {}}

    @staticmethod
    def save_nss_assessment(patient_id: str, total_score: int) -> None:
        client = get_supabase()
        if not client: return
        try:
            client.table("nss_assessments").insert({
                "patient_id": patient_id,
                "total_score": total_score
            }).execute()
        except Exception as e:
            logger.error(f"ClinicalRepository.save_nss_assessment error: {e}")

    @staticmethod
    def save_nds_assessment(patient_id: str, total_score: int) -> None:
        client = get_supabase()
        if not client: return
        try:
            client.table("nds_assessments").insert({
                "patient_id": patient_id,
                "total_score": total_score
            }).execute()
        except Exception as e:
            logger.error(f"ClinicalRepository.save_nds_assessment error: {e}")

    @staticmethod
    def save_gum_assessment(patient_id: str, total_score: int) -> None:
        client = get_supabase()
        if not client: return
        try:
            client.table("gum_assessments").insert({
                "patient_id": patient_id,
                "total_score": total_score
            }).execute()
        except Exception as e:
            logger.error(f"ClinicalRepository.save_gum_assessment error: {e}")

    @staticmethod
    def save_ulcer_assessment(patient_id: str, ulcer_stage: int) -> None:
        client = get_supabase()
        if not client: return
        try:
            client.table("ulcer_assessments").insert({
                "patient_id": patient_id,
                "ulcer_stage": ulcer_stage
            }).execute()
        except Exception as e:
            logger.error(f"ClinicalRepository.save_ulcer_assessment error: {e}")
