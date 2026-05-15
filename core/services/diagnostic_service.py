import logging
from typing import Dict, Any, List, Optional
from core.repositories.patient_repo import PatientRepository
from core.repositories.clinical_repo import ClinicalRepository
from core.repositories.ml_repo import MLRepository
from core.repositories.memory_repo import MemoryRepository
from core.repositories.decision_repo import DecisionRepository
from core.questionnaire import (
    calculate_section_scores,
    ml_neuropathy_prediction,
    final_decision
)

logger = logging.getLogger(__name__)

class DiagnosticService:
    """
    Business orchestration layer.
    Agents and UI must only call this service, never the database directly.
    """

    @staticmethod
    def load_patient_context(patient_id: str) -> Dict[str, Any]:
        """Loads all existing patient data from DB."""
        clinical = ClinicalRepository.get_clinical_data(patient_id)
        ml_res = MLRepository.get_ml_prediction(patient_id)
        decision = DecisionRepository.get_latest_decision(patient_id)
        
        return {
            "clinical_data": clinical,
            "ml_prediction": ml_res,
            "latest_decision": decision
        }

    @staticmethod
    def get_clinical_data(patient_id: str) -> Dict[str, Any]:
        return ClinicalRepository.get_clinical_data(patient_id)

    @staticmethod
    def get_ml_prediction(patient_id: str) -> Dict[str, Any]:
        return MLRepository.get_ml_prediction(patient_id)

    @staticmethod
    def get_recent_decisions(patient_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        return DecisionRepository.get_decisions(patient_id, limit)

    @staticmethod
    def save_clinical_data(patient_id: str, answers: Dict[str, Any]) -> Dict[str, Any]:
        """Computes section scores from answers and saves them to DB."""
        if not answers:
            return {}
            
        scores = calculate_section_scores(answers)
        
        ClinicalRepository.save_nss_assessment(patient_id, scores.get("nss_score", 0))
        ClinicalRepository.save_nds_assessment(patient_id, scores.get("nds_score", 0))
        ClinicalRepository.save_gum_assessment(patient_id, scores.get("gum_score", 0))
        ClinicalRepository.save_ulcer_assessment(patient_id, scores.get("ulcer_score", 0))
        
        return scores

    @staticmethod
    def run_ml_inference(patient_id: str, answers: Dict[str, Any], nss_score: int, age: int) -> Dict[str, Any]:
        """Computes ML prediction and saves it to DB."""
        result = ml_neuropathy_prediction(answers, nss_score, age)
        
        features = result.get("features", {})
        
        # Fallback values for features not tracked in the current questionnaire
        heat_avg = float((answers.get("ml_heat_right", 38) + answers.get("ml_heat_left", 38)) / 2)
        cold_avg = float((answers.get("ml_cold_right", 20) + answers.get("ml_cold_left", 20)) / 2)
        
        MLRepository.save_ml_prediction(
            patient_id=patient_id,
            nss_score=features.get("nss", nss_score),
            bmi_baseline=float(features.get("bmi", 22)),
            age_baseline=int(features.get("age", age)),
            hba1c_baseline=float(features.get("hba1c", 7)),
            heat_avg=heat_avg,
            cold_avg=cold_avg,
            predicted_class=result["predicted_class"],
            predicted_probability=result["predicted_probability"]
        )
        
        return result

    @staticmethod
    def compute_fusion(patient_id: str, ai_prediction: str, nds_score: int, nss_score: int) -> Dict[str, Any]:
        """Computes final fusion decision and saves it to DB."""
        result = final_decision(ai_prediction, nds_score, nss_score)
        
        DecisionRepository.save_final_decision(
            patient_id=patient_id,
            ai_prediction=ai_prediction,
            nds_score=nds_score,
            nss_score=nss_score,
            calculated_score=result["fusion_score"],
            final_decision=result["final_decision"]
        )
        
        return result

    @staticmethod
    def store_memory(patient_id: str, session_id: str, role: str, content: str, embedding: Optional[List[float]] = None) -> None:
        """Saves interaction to memory RAG layer."""
        MemoryRepository.save_conversation_memory(patient_id, session_id, role, content, embedding)

    @staticmethod
    def retrieve_memory(patient_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieves semantic memory records."""
        from core.rag_engine import generate_embedding

        embedding = generate_embedding(query)
        if not embedding:
            return []
        return MemoryRepository.match_memory(patient_id, embedding, limit)
        
    @staticmethod
    def get_all_patients() -> List[Dict[str, Any]]:
        """Retrieves all patients for UI."""
        return PatientRepository.get_all_patients()
        
    @staticmethod
    def create_new_patient(name: str, age: int, gender: str, diabetes_type: Optional[str] = None, diabetes_duration: Optional[int] = None) -> Dict[str, Any]:
        """Creates a new patient."""
        return PatientRepository.create_patient(name, age, gender, diabetes_type, diabetes_duration)
