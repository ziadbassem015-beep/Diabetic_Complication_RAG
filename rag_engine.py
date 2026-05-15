import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv
from core.services.diagnostic_service import DiagnosticService

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://models.inference.ai.azure.com")

# Initialize OpenAI Client (GitHub Models)
client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

def generate_embedding(text: str) -> list:
    """Generate vector embedding using GitHub Models (OpenAI compatible)."""
    if not OPENAI_API_KEY:
        return [0.0] * 1536
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding Error: {e}")
        return [0.0] * 1536

def run_diagnostic_pipeline(patient_id: str, chat_history: list) -> dict:
    """The 9-step Medical Diagnostic Pipeline with Decision Fusion."""
    
    # 1. Fetch Patient Context
    clinical_data = DiagnosticService.get_clinical_data(patient_id)
    ml_data = DiagnosticService.get_ml_prediction(patient_id)
    
    # 2. Retrieve Semantic Memory (RAG)
    user_query = chat_history[-1]["content"] if chat_history else ""
    semantic_context = ""
    try:
        rag_results = DiagnosticService.retrieve_memory(patient_id, user_query, limit=3)
        semantic_context = "\n".join([r["content"] for r in rag_results])
    except Exception:
        semantic_context = "No relevant past memory found."

    # 3. Decision Fusion Logic
    ml_prob = ml_data.get("predicted_probability", 0.5)
    nss = clinical_data.get("nss_score", 0)
    nds = clinical_data.get("nds_score", 0)
    
    # Normalize scores for fusion
    norm_nss = nss / 14.0
    norm_nds = nds / 23.0
    fusion_score = (0.4 * ml_prob) + (0.3 * norm_nss) + (0.3 * norm_nds)
    
    # 4. Build Prompt
    prompt = f"""You are an expert AI Medical Decision System specializing in Diabetic Peripheral Neuropathy (DPN).
Your role is dual:
1. Ask the patient structured clinical questions one at a time to assess their condition.
2. Provide a final diagnosis if enough information (NSS, NDS, ML) is gathered.

CONTEXT:
Patient Clinical Scores: NSS={nss}, NDS={nds}
ML Prediction Probability: {ml_prob:.2f}
Decision Fusion Score: {fusion_score:.2f}
Semantic Memory: {semantic_context}

INSTRUCTIONS:
- Be professional and empathetic.
- Output ONLY valid JSON.
- Provide 2-3 'suggested_answers' for the patient to click.
- Provide 2 'suggested_questions' the patient might want to ask YOU.

JSON SCHEMA:
{{
  "answer": "Your response text here",
  "suggested_answers": ["Option 1", "Option 2"],
  "suggested_questions": ["What does my NSS score mean?", "What are the next steps?"]
}}"""

    # 5. Call LLM
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                *chat_history[-5:]
            ],
            model="gpt-4o",
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "answer": f"Error in AI Pipeline: {str(e)}",
            "suggested_answers": ["Retry"],
            "suggested_questions": ["Why did this happen?"]
        }
