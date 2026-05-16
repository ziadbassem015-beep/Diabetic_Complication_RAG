import logging
from typing import Any, Dict, List, Optional
from core.database.client import get_supabase

logger = logging.getLogger(__name__)

class BaseRepository:
    """Common helpers for repositories — centralizes access to the DB client."""

    @classmethod
    def _client(cls) -> Optional[Any]:
        return get_supabase()

    @classmethod
    def select_latest_by_patient(cls, table: str, patient_id: str, order_col: str = 'created_at', limit: int = 1) -> Dict[str, Any]:
        client = cls._client()
        if not client:
            return {}
        try:
            res = client.table(table).select("*").eq("patient_id", patient_id).order(order_col, desc=True).limit(limit).execute()
            return (res.data or [{}])[0]
        except Exception as e:
            logger.error(f"{cls.__name__}.select_latest_by_patient error: {e}")
            return {}

    @classmethod
    def select_many_by_patient(cls, table: str, patient_id: str, limit: int = 5, order_col: str = 'created_at') -> List[Dict[str, Any]]:
        client = cls._client()
        if not client:
            return []
        try:
            res = client.table(table).select("*").eq("patient_id", patient_id).order(order_col, desc=True).limit(limit).execute()
            return res.data or []
        except Exception as e:
            logger.error(f"{cls.__name__}.select_many_by_patient error: {e}")
            return []

    @classmethod
    def insert(cls, table: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        client = cls._client()
        if not client:
            return None
        try:
            res = client.table(table).insert(payload).execute()
            return (res.data or [{}])[0]
        except Exception as e:
            logger.error(f"{cls.__name__}.insert error: {e}")
            return None

    @classmethod
    def select_all(cls, table: str) -> List[Dict[str, Any]]:
        client = cls._client()
        if not client:
            return []
        try:
            res = client.table(table).select("*").execute()
            return res.data or []
        except Exception as e:
            logger.error(f"{cls.__name__}.select_all error: {e}")
            return []
