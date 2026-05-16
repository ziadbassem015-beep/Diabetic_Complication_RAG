import logging
from typing import List, Dict, Any, Optional
from core.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)

class MemoryRepository(BaseRepository):

    @staticmethod
    def save_conversation_memory(
        patient_id: str,
        session_id: str,
        role: str,
        content: str,
        embedding: Optional[List[float]] = None
    ) -> None:
        payload = {
            "patient_id": patient_id,
            "session_id": session_id,
            "role": role,
            "content": content
        }
        if embedding is not None:
            payload["embedding"] = embedding
        MemoryRepository.insert("conversation_memory", payload)

    @staticmethod
    def match_memory(patient_id: str, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        client = MemoryRepository._client()
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
