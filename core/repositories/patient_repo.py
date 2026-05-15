import uuid
import logging
from typing import List, Dict, Any, Optional
from core.database.client import get_supabase

logger = logging.getLogger(__name__)

class PatientRepository:
    
    @staticmethod
    def get_all_patients() -> List[Dict[str, Any]]:
        client = get_supabase()
        if not client:
            return []
        try:
            res = client.table("patients").select("*").execute()
            return res.data or []
        except Exception as e:
            logger.error(f"PatientRepository.get_all_patients error: {e}")
            return []

    @staticmethod
    def create_patient(
        name: str,
        age: int,
        gender: str,
        diabetes_type: Optional[str] = None,
        diabetes_duration: Optional[int] = None
    ) -> Dict[str, Any]:
        client = get_supabase()
        if not client:
            return {}
        try:
            patient_id = str(uuid.uuid4())
            payload = {
                "id": patient_id,
                "name": name.strip(),
                "age": age,
                "gender": gender
            }
            res = client.table("patients").insert(payload).execute()
            patient = (res.data or [{}])[0]
            if not patient:
                return {}
            
            patient["diabetes_type"] = diabetes_type
            patient["diabetes_duration"] = diabetes_duration
            return patient
        except Exception as e:
            logger.error(f"PatientRepository.create_patient error: {e}")
            return {}
