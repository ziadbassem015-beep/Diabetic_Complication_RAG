import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    # Use placeholder or warn for now
    print("Warning: SUPABASE_URL or SUPABASE_KEY is missing in .env")

# Initialize Supabase Client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None
    print(f"Error initializing Supabase: {e}")


def get_all_patients():
    """Fetch all patients for the UI dropdown."""
    if not supabase: return []
    response = supabase.table("patients").select("*").execute()
    return response.data


def get_patient_clinical_data(patient_id: str):
    """Fetch the latest NSS, NDS, and Gum assessments for a patient."""
    if not supabase: return {}
    
    # Get latest NSS
    nss_res = supabase.table("nss_assessments").select("*").eq("patient_id", patient_id).order("created_at", desc=True).limit(1).execute()
    # Get latest NDS
    nds_res = supabase.table("nds_assessments").select("*").eq("patient_id", patient_id).order("created_at", desc=True).limit(1).execute()
    
    return {
        "nss": nss_res.data[0] if nss_res.data else None,
        "nds": nds_res.data[0] if nds_res.data else None
    }


def get_patient_ml_prediction(patient_id: str):
    """Fetch the latest Random Forest ML prediction for a patient."""
    if not supabase: return None
    res = supabase.table("ml_neuropathy_predictions").select("*").eq("patient_id", patient_id).order("created_at", desc=True).limit(1).execute()
    return res.data[0] if res.data else None


def save_conversation_memory(patient_id: str, session_id: str, role: str, content: str, embedding: list = None):
    """Store a chat message in the long-term memory."""
    if not supabase: return None
    
    data = {
        "patient_id": patient_id,
        "session_id": session_id,
        "role": role,
        "content": content
    }
    
    # Add embedding if generated
    if embedding:
        data["embedding"] = embedding
        
    return supabase.table("conversation_memory").insert(data).execute()


def save_final_decision(patient_id: str, ai_prediction: str, nds_score: int, nss_score: int, calculated_score: float, final_decision: str):
    """Save the final fused decision."""
    if not supabase: return None
    
    data = {
        "patient_id": patient_id,
        "ai_prediction": ai_prediction,
        "nds_score": nds_score,
        "nss_score": nss_score,
        "calculated_score": calculated_score,
        "final_decision": final_decision
    }
    return supabase.table("final_diagnostic_decisions").insert(data).execute()
