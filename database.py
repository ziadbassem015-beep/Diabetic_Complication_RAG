"""
database.py — Legacy compatibility shim.

This module is retained for backwards compatibility only and no longer
performs direct Supabase queries. Legacy helpers route through the
repository layer so the application enforces UI → Service → Repository
→ DatabaseClient.
"""

from typing import Optional, List, Dict, Any
from core.repositories.patient_repo import PatientRepository
from core.repositories.clinical_repo import ClinicalRepository
from core.repositories.ml_repo import MLRepository
from core.repositories.memory_repo import MemoryRepository
from core.repositories.decision_repo import DecisionRepository

__all__ = [
    "get_all_patients",
    "get_patient_clinical_data",
    "get_patient_ml_prediction",
    "save_conversation_memory",
    "save_final_decision",
    "create_patient",
]


def get_all_patients() -> List[Dict[str, Any]]:
    return PatientRepository.get_all_patients()


def get_patient_clinical_data(patient_id: str) -> Dict[str, Any]:
    clinical = ClinicalRepository.get_clinical_data(patient_id)
    nss = clinical.get("nss") or {}
    nds = clinical.get("nds") or {}
    return {
        "nss_score": nss.get("total_score", 0),
        "nds_score": nds.get("total_score", 0),
    }


def get_patient_ml_prediction(patient_id: str) -> Dict[str, Any]:
    ml = MLRepository.get_ml_prediction(patient_id)
    return ml if ml else {"predicted_probability": 0.5}


def save_conversation_memory(
    patient_id: str,
    session_id: str,
    role: str,
    content: str,
    embedding: Optional[List[float]] = None,
) -> None:
    MemoryRepository.save_conversation_memory(patient_id, session_id, role, content, embedding)


def save_final_decision(
    patient_id: str,
    ai_prediction: str,
    nds_score: int,
    nss_score: int,
    calculated_score: float,
    final_decision: str,
) -> None:
    DecisionRepository.save_final_decision(
        patient_id=patient_id,
        ai_prediction=ai_prediction,
        nds_score=nds_score,
        nss_score=nss_score,
        calculated_score=calculated_score,
        final_decision=final_decision,
    )


def create_patient(
    name: str,
    age: int,
    gender: str,
    diabetes_type: Optional[str] = None,
    diabetes_duration: Optional[int] = None,
) -> Dict[str, Any]:
    return PatientRepository.create_patient(
        name=name,
        age=age,
        gender=gender,
        diabetes_type=diabetes_type,
        diabetes_duration=diabetes_duration,
    )
