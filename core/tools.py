"""
tools.py — Tool Registry for the Autonomous Medical Diagnostic Agent.
Each tool wraps existing logic and returns a structured Observation dict.
The agent calls these tools dynamically based on its reasoning.
"""
from typing import Any
from core.services.diagnostic_service import DiagnosticService

# ══════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════

def tool_search_memory(state: Any, args: dict) -> dict:
    """
    Vector RAG retrieval: finds semantically similar past records from repository.
    Args: { "query": "text to search for" }
    """
    query = args.get("query", "neuropathy assessment")
    if not state.patient_id:
        return {"status": "error", "message": "No patient selected."}

    try:
        results = DiagnosticService.retrieve_memory(state.patient_id, query)
        state.retrieved_memory.extend(results)
        return {
            "status": "success",
            "count": len(results),
            "results": [{"content": item.get("content", "")[:200]} for item in results]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_get_clinical_data(state: Any, args: dict) -> dict:
    """
    Fetches the latest NSS and NDS clinical scores from repository.
    Args: {}
    """
    if not state.patient_id:
        return {"status": "error", "message": "No patient selected."}

    try:
        data = DiagnosticService.get_clinical_data(state.patient_id)
        nss = data.get("nss")
        nds = data.get("nds")

        result = {
            "status": "success",
            "nss_score": nss["total_score"] if nss else None,
            "nss_severity": nss.get("severity") if nss else None,
            "nds_score": nds["total_score"] if nds else None,
            "nds_severity": nds.get("severity") if nds else None,
            "has_data": bool(nss or nds)
        }
        state.clinical_data = result
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_get_ml_prediction(state: Any, args: dict) -> dict:
    """
    Fetches the stored ML neuropathy prediction from repository.
    Args: {}
    """
    if not state.patient_id:
        return {"status": "error", "message": "No patient selected."}

    try:
        ml = DiagnosticService.get_ml_prediction(state.patient_id)
        if ml:
            result = {
                "status": "success",
                "predicted_class": ml["predicted_class"],
                "predicted_probability": ml["predicted_probability"],
                "ai_prediction": "مريض" if ml["predicted_class"] == 1 else "سليم"
            }
        else:
            result = {
                "status": "no_data",
                "message": "No ML prediction stored. Will compute from answers.",
                "predicted_class": None
            }
        state.ml_results = result
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_compute_ml_prediction(state: Any, args: dict) -> dict:
    """
    Computes ML neuropathy prediction from patient's questionnaire answers.
    Args: {}  (uses state.answers + state.patient_info automatically)
    """
    if not state.patient_id:
        return {"status": "error", "message": "No patient selected."}

    nss_score = 0
    if state.clinical_data:
        nss_score = state.clinical_data.get("nss_score", 0) or 0
    elif state.answers:
        nss_score = DiagnosticService.save_clinical_data(state.patient_id, state.answers).get("nss_score", 0)

    age = state.patient_info.get("age", 50) if hasattr(state, "patient_info") else 50
    result = DiagnosticService.run_ml_inference(
        patient_id=state.patient_id,
        answers=state.answers,
        nss_score=nss_score,
        age=age,
    )
    state.ml_results = result
    return result


def tool_calculate_fusion_score(state: Any, args: dict) -> dict:
    """
    Runs the final_decision weighted formula: 1.0×AI + 1.2×NDS + 0.9×NSS.
    Args: {}  (uses state.ml_results + state.clinical_data automatically)
    """
    if not state.patient_id:
        return {"status": "error", "message": "No patient selected."}

    ai_pred = state.ml_results.get("ai_prediction", "سليم") if state.ml_results else "سليم"
    nds_score = 0
    nss_score = 0
    if state.clinical_data:
        nds_score = state.clinical_data.get("nds_score", 0) or 0
        nss_score = state.clinical_data.get("nss_score", 0) or 0

    result = DiagnosticService.compute_fusion(
        patient_id=state.patient_id,
        ai_prediction=ai_pred,
        nds_score=nds_score,
        nss_score=nss_score,
    )
    state.fusion_results = result
    return result


def tool_ask_patient_question(state: Any, args: dict) -> dict:
    """
    Pauses the agent loop to ask the patient a question.
    Args: { "question": "...", "options": ["opt1", "opt2", ...], "key": "answer_key" }
    The agent MUST use this tool whenever it needs patient input.
    """
    question = args.get("question", "")
    options = args.get("options", [])
    key = args.get("key", f"q_{state.iteration}")

    state.waiting_for_patient = True
    state.pending_question = {
        "question": question,
        "options": options,
        "key": key
    }

    return {
        "status": "waiting",
        "message": f"Waiting for patient to answer: {question}",
        "question": question,
        "options": options,
        "key": key
    }


def tool_save_clinical_scores(state: Any, args: dict) -> dict:
    """
    Saves computed NSS/NDS/Gum/Ulcer scores via DiagnosticService.
    Args: {}  (uses state.answers)
    """
    if not state.answers:
        return {"status": "skip", "message": "No answers collected yet."}

    try:
        result = DiagnosticService.save_clinical_data(
            patient_id=state.patient_id,
            answers=state.answers,
        )
        if not isinstance(state.clinical_data, dict):
            state.clinical_data = {}
        state.clinical_data.update({
            "nss_score": result.get("nss_score"),
            "nds_score": result.get("nds_score"),
            "gum_score": result.get("gum_score"),
            "ulcer_score": result.get("ulcer_score"),
        })
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ══════════════════════════════════════════════════════════════════
# TOOL REGISTRY
# ══════════════════════════════════════════════════════════════════

TOOL_REGISTRY: dict[str, callable] = {
    "search_memory": tool_search_memory,
    "get_clinical_data": tool_get_clinical_data,
    "get_ml_prediction": tool_get_ml_prediction,
    "compute_ml_prediction": tool_compute_ml_prediction,
    "calculate_fusion_score": tool_calculate_fusion_score,
    "ask_patient_question": tool_ask_patient_question,
    "save_clinical_scores": tool_save_clinical_scores,
}

TOOL_DESCRIPTIONS = """
Available tools you can call:

1. search_memory(query)
   → Search patient's long-term memory for similar past conversations/history.
   → Use when: patient history is relevant to current reasoning.

2. get_clinical_data()
   → Fetch stored NSS + NDS clinical scores for this patient.
   → Use when: patient has prior assessments in the database.

3. get_ml_prediction()
   → Fetch stored ML neuropathy prediction.
   → Use when: checking if a prior ML result exists.

4. compute_ml_prediction()
   → Run the ML approximation model using patient's collected answers.
   → Use when: enough BMI/HbA1c/temperature answers are collected.

5. calculate_fusion_score()
   → Run the weighted final decision formula: 1.0×AI + 1.2×NDS + 0.9×NSS.
   → Use when: ML + clinical data are both available.

6. ask_patient_question(question, options, key)
   → Pause and ask the patient a question with clickable options.
   → Use when: you need symptom information from the patient.

7. save_clinical_scores()
   → Save all collected clinical assessment scores.
   → Use when: all sections of questionnaire are answered.
"""

def execute_tool(tool_name: str, state: Any, args: dict) -> dict:
    """Execute a tool from the registry and return its observation."""
    if tool_name not in TOOL_REGISTRY:
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    try:
        result = TOOL_REGISTRY[tool_name](state, args)
        state.add_observation(tool_name, result)
        return result
    except Exception as e:
        err = {"status": "error", "tool": tool_name, "message": str(e)}
        state.add_observation(tool_name, err)
        return err
