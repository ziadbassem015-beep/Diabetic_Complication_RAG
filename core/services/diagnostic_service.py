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
    final_decision,
    ml_gestational_prediction,
    ml_heart_risk_prediction,
    calculate_gestational_score,
    calculate_heart_risk_score,
    get_eligible_questions,
)
from core.repositories.gestational_repo import GestationalRepository
from core.repositories.heart_risk_repo import HeartRiskRepository

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

    # ═══════════════════════════════════════════════════════════════
    # GESTATIONAL DIABETES ASSESSMENT METHODS (NEW)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def save_gestational_assessment(patient_id: str, answers: Dict[str, Any], patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """Computes gestational diabetes risk score and saves it to DB."""
        # Only process for female patients
        if patient_info.get("gender") != "Female":
            return {"error": "Gestational diabetes assessment only applicable for female patients"}
        
        gd_scores = calculate_gestational_score(answers)
        bmi = float(answers.get("gd_bmi", patient_info.get("bmi", 25)))
        pregnancy_week = int(answers.get("gd_pregnancy_week", 0))
        glucose_level = float(answers.get("gd_fasting_glucose", 0) * 30)  # Convert score to mg/dL estimate
        fasting_glucose = float(answers.get("gd_fasting_glucose", 0) * 30)
        insulin_resistance = float(answers.get("gd_bmi", 25) / 10.0)  # Proxy for insulin resistance
        family_history = bool(answers.get("gd_family_history", 0) > 0)
        
        # Run ML prediction
        ml_result = ml_gestational_prediction(answers, gd_scores["gd_score"], bmi, patient_info.get("age", 25))
        
        # Save to database
        db_result = GestationalRepository.save_gestational_assessment(
            patient_id=patient_id,
            pregnancy_week=pregnancy_week,
            glucose_level=glucose_level,
            fasting_glucose=fasting_glucose,
            insulin_resistance=insulin_resistance,
            bmi=bmi,
            family_history=family_history,
            risk_score=ml_result["predicted_probability"],
            predicted_class=ml_result["predicted_class"],
            predicted_probability=ml_result["predicted_probability"],
        )
        
        return {
            "gd_scores": gd_scores,
            "ml_result": ml_result,
            "db_result": db_result,
        }

    @staticmethod
    def get_gestational_assessment(patient_id: str) -> Dict[str, Any]:
        """Fetch the latest gestational diabetes assessment for a patient."""
        return GestationalRepository.get_gestational_assessment(patient_id)

    @staticmethod
    def get_gestational_history(patient_id: str, limit: int = 5) -> list:
        """Fetch gestational diabetes assessment history for a patient."""
        return GestationalRepository.get_gestational_history(patient_id, limit)

    # ═══════════════════════════════════════════════════════════════
    # HEART RISK ASSESSMENT METHODS (NEW)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def save_heart_risk_assessment(patient_id: str, answers: Dict[str, Any], patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """Computes heart risk score and saves it to DB."""
        hr_scores = calculate_heart_risk_score(answers)
        
        # Extract heart risk features
        cholesterol = float(answers.get("hr_cholesterol", 0) * 60 + 150)  # Convert score to mg/dL estimate
        bp_score = int(answers.get("hr_blood_pressure", 0))
        blood_pressure_systolic = float(120 + bp_score * 15)  # Estimate systolic
        blood_pressure_diastolic = float(80 + bp_score * 10)  # Estimate diastolic
        resting_heart_rate = int(60 + answers.get("hr_resting_heart_rate", 0) * 15)
        smoking_status = {0: "Never", 1: "Former (>1y)", 2: "Former (<1y)", 3: "Current"}.get(answers.get("hr_smoking_status", 0), "Unknown")
        bmi = float(answers.get("hr_bmi", patient_info.get("bmi", 25)))
        exercise_frequency = int(answers.get("hr_exercise_frequency", 0))
        diabetes_duration = int(patient_info.get("diabetes_duration", 0))
        
        # Run ML prediction
        ml_result = ml_heart_risk_prediction(
            answers,
            hr_scores["hr_score"],
            patient_info.get("age", 50),
            diabetes_duration
        )
        
        # Save to database
        db_result = HeartRiskRepository.save_heart_risk_assessment(
            patient_id=patient_id,
            cholesterol=cholesterol,
            blood_pressure_systolic=blood_pressure_systolic,
            blood_pressure_diastolic=blood_pressure_diastolic,
            resting_heart_rate=resting_heart_rate,
            smoking_status=smoking_status,
            bmi=bmi,
            diabetes_duration=diabetes_duration,
            exercise_frequency=exercise_frequency,
            risk_score=ml_result["predicted_probability"],
            predicted_class=ml_result["predicted_class"],
            predicted_probability=ml_result["predicted_probability"],
        )
        
        return {
            "hr_scores": hr_scores,
            "ml_result": ml_result,
            "db_result": db_result,
        }

    @staticmethod
    def get_heart_risk_assessment(patient_id: str) -> Dict[str, Any]:
        """Fetch the latest heart risk assessment for a patient."""
        return HeartRiskRepository.get_heart_risk_assessment(patient_id)

    @staticmethod
    def get_heart_risk_history(patient_id: str, limit: int = 5) -> list:
        """Fetch heart risk assessment history for a patient."""
        return HeartRiskRepository.get_heart_risk_history(patient_id, limit)

    @staticmethod
    def run_secondary_assessments(
        patient_id: str,
        answers: Dict[str, Any],
        patient_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run post-fusion secondary assessments (gestational + heart risk).
        Gestational runs only for Female patients; heart risk runs for all.
        """
        gender = patient_info.get("gender", "")
        result: Dict[str, Any] = {
            "gestational": {},
            "heart_risk": {},
            "skipped_assessments": [],
        }

        if gender == "Female":
            result["gestational"] = DiagnosticService.save_gestational_assessment(
                patient_id, answers, patient_info
            )
            logger.info({
                "event": "gestational_saved",
                "patient_id": patient_id,
                "node": "secondary_assessment_node",
            })
        else:
            result["skipped_assessments"].append("gestational_diabetes")
            result["gestational"] = {
                "skipped": True,
                "reason": "Not applicable — gestational screening is for female patients only",
            }

        result["heart_risk"] = DiagnosticService.save_heart_risk_assessment(
            patient_id, answers, patient_info
        )
        logger.info({
            "event": "heart_risk_saved",
            "patient_id": patient_id,
            "node": "secondary_assessment_node",
        })

        return result
