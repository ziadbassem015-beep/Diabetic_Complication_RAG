"""
tools.py — Tool Registry for the Autonomous Medical Diagnostic Agent.
Each tool wraps existing logic and returns a structured Observation dict.
The agent calls these tools dynamically based on its reasoning.
"""
from typing import Any
from agent_state import AgentState
from database import (
    get_patient_clinical_data,
    get_patient_ml_prediction,
    save_conversation_memory,
    supabase,
)
from rag_engine import generate_embedding
from questionnaire import (
    calculate_section_scores,
    ml_neuropathy_prediction,
    final_decision as compute_final_decision,
)


# ══════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════

def tool_search_memory(state: AgentState, args: dict) -> dict:
    """
    Vector RAG retrieval: finds semantically similar past records in Supabase.
    Args: { "query": "text to search for" }
    """
    query = args.get("query", "neuropathy assessment")
    if not state.patient_id:
        return {"status": "error", "message": "No patient selected."}

    query_emb = generate_embedding(query)
    if not query_emb:
        return {"status": "no_embedding", "message": "Embeddings unavailable.", "results": []}

    try:
        res = supabase.rpc("match_memory", {
            "query_embedding": query_emb,
            "match_threshold": 0.5,
            "match_count": 5,
            "p_patient_id": state.patient_id
        }).execute()
        results = res.data or []
        state.retrieved_memory.extend(results)
        return {
            "status": "success",
            "count": len(results),
            "results": [{"content": r.get("content", "")[:200]} for r in results]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_get_clinical_data(state: AgentState, args: dict) -> dict:
    """
    Fetches the latest NSS and NDS clinical scores from Supabase.
    Args: {}
    """
    if not state.patient_id:
        return {"status": "error", "message": "No patient selected."}

    try:
        data = get_patient_clinical_data(state.patient_id)
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


def tool_get_ml_prediction(state: AgentState, args: dict) -> dict:
    """
    Fetches the stored ML neuropathy prediction from Supabase.
    Args: {}
    """
    if not state.patient_id:
        return {"status": "error", "message": "No patient selected."}

    try:
        ml = get_patient_ml_prediction(state.patient_id)
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


def tool_compute_ml_prediction(state: AgentState, args: dict) -> dict:
    """
    Computes ML neuropathy prediction from patient's questionnaire answers.
    Args: {}  (uses state.answers + state.patient_info automatically)
    """
    scores = calculate_section_scores(state.answers)
    nss = scores["nss_score"]
    age = state.patient_info.get("age", 50)

    result = ml_neuropathy_prediction(state.answers, nss, age)
    state.ml_results = result

    # Save to Supabase
    try:
        f = result["features"]
        supabase.table("ml_neuropathy_predictions").insert({
            "patient_id": state.patient_id,
            "nss_score": f["nss"],
            "bmi_baseline": f["bmi"],
            "age_baseline": f["age"],
            "hba1c_baseline": f["hba1c"],
            "heat_avg": (f["heat_right"] + f["heat_left"]) / 2,
            "cold_avg": (f["cold_right"] + f["cold_left"]) / 2,
            "predicted_class": result["predicted_class"],
            "predicted_probability": result["predicted_probability"]
        }).execute()
    except Exception:
        pass

    return result


def tool_calculate_fusion_score(state: AgentState, args: dict) -> dict:
    """
    Runs the final_decision weighted formula: 1.0×AI + 1.2×NDS + 0.9×NSS.
    Args: {}  (uses state.ml_results + state.clinical_data automatically)
    """
    ai_pred = state.ml_results.get("ai_prediction", "سليم")

    # Get NDS/NSS from clinical data or compute from answers
    if state.clinical_data.get("nds_score") is not None:
        nds = state.clinical_data["nds_score"]
        nss = state.clinical_data["nss_score"] or 0
    else:
        scores = calculate_section_scores(state.answers)
        nds = scores["nds_score"]
        nss = scores["nss_score"]

    result = compute_final_decision(ai_pred, nds, nss)
    state.fusion_results = result

    # Save final decision to Supabase
    try:
        supabase.table("final_diagnostic_decisions").insert({
            "patient_id": state.patient_id,
            "ai_prediction": ai_pred,
            "nds_score": nds,
            "nss_score": nss,
            "calculated_score": result["fusion_score"],
            "final_decision": result["final_decision"]
        }).execute()
    except Exception:
        pass

    return result


def tool_ask_patient_question(state: AgentState, args: dict) -> dict:
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


def tool_save_clinical_scores(state: AgentState, args: dict) -> dict:
    """
    Saves computed NSS/NDS/Gum/Ulcer scores to Supabase.
    Args: {}  (uses state.answers)
    """
    if not state.answers:
        return {"status": "skip", "message": "No answers collected yet."}

    scores = calculate_section_scores(state.answers)
    nss = scores["nss_score"]
    nds = scores["nds_score"]
    gum = scores["gum_score"]
    ulcer = scores["ulcer_score"]

    nss_cat = "Normal" if nss <= 2 else "Mild" if nss <= 4 else "Moderate" if nss <= 6 else "Severe"
    nds_cat = "Normal" if nds <= 5 else "Mild" if nds <= 10 else "Moderate" if nds <= 16 else "Severe"
    gum_cat = "Healthy" if gum <= 5 else "Mild gingivitis" if gum <= 11 else "Moderate gingivitis" if gum <= 18 else "Severe gingivitis"
    ulcer_cat = "Low risk" if ulcer < 4 else "Moderate risk" if ulcer < 8 else "High risk"

    try:
        supabase.table("nss_assessments").insert({"patient_id": state.patient_id, "total_score": nss, "severity": nss_cat, "symptoms_details": {k: v for k, v in state.answers.items() if k.startswith("nss_")}}).execute()
        supabase.table("nds_assessments").insert({"patient_id": state.patient_id, "total_score": nds, "severity": nds_cat, "test_details": {k: v for k, v in state.answers.items() if k.startswith("nds_")}}).execute()
        supabase.table("gum_assessments").insert({"patient_id": state.patient_id, "total_score": gum, "status": gum_cat, "clinical_signs": {k: v for k, v in state.answers.items() if k.startswith("gum_")}}).execute()
        supabase.table("ulcer_assessments").insert({"patient_id": state.patient_id, "ulcer_stage": ulcer_cat, "infection_signs": {k: v for k, v in state.answers.items() if k.startswith("ulcer_")}}).execute()
    except Exception as e:
        return {"status": "error", "message": str(e)}

    state.clinical_data.update({"nss_score": nss, "nds_score": nds, "gum_score": gum, "ulcer_score": ulcer})
    return {"status": "success", "nss": nss, "nds": nds, "gum": gum, "ulcer": ulcer, "nss_cat": nss_cat, "nds_cat": nds_cat}


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
   → Fetch stored NSS + NDS clinical scores from Supabase for this patient.
   → Use when: patient has prior assessments in the database.

3. get_ml_prediction()
   → Fetch stored ML neuropathy prediction from Supabase.
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
   → Save all collected scores to Supabase.
   → Use when: all sections of questionnaire are answered.
"""


def execute_tool(tool_name: str, state: AgentState, args: dict) -> dict:
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
