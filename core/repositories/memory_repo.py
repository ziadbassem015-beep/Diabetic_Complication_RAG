import logging
from typing import List, Dict, Any, Optional
from core.database.client import get_supabase

logger = logging.getLogger(__name__)

class MemoryRepository:

    @staticmethod
    def save_conversation_memory(
        patient_id: str,
        session_id: str,
        role: str,
        content: str,
        embedding: Optional[List[float]] = None
    ) -> None:
        client = get_supabase()
        if not client: return
        try:
            payload = {
                "patient_id": patient_id,
                "session_id": session_id,
                "role": role,
                "content": content
            }
            if embedding is not None:
                payload["embedding"] = embedding
            
            client.table("conversation_memory").insert(payload).execute()
        except Exception as e:
            logger.error(f"MemoryRepository.save_conversation_memory error: {e}")

    @staticmethod
    def match_memory(patient_id: str, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        client = get_supabase()
        if not client: return []
        try:
            res = client.rpc("match_memory", {
                "query_embedding": query_embedding,
                "match_threshold": 0.45,
                "match_count": limit,
                "p_patient_id": patient_id
            }).execute()
            return res.data or []
        except Exception as e:
            logger.error(f"MemoryRepository.match_memory error: {e}")
            return []
