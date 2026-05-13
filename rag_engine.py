import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv
from database import get_patient_clinical_data, get_patient_ml_prediction, supabase

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://models.inference.ai.azure.com")

# Initialize OpenAI client pointed at GitHub Models
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)


def call_llm(prompt: str) -> str:
    """Call the LLM safely, returns a raw string response."""
    if not OPENAI_API_KEY:
        return '{"message": "Warning: OPENAI_API_KEY is not set in .env", "suggested_answers": ["OK"], "is_diagnosis_complete": false}'
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        return f'{{"message": "Error communicating with AI: {str(e)[:80]}", "suggested_answers": ["Retry"], "is_diagnosis_complete": false}}'


def generate_embedding(text: str) -> list:
    """Generate vector embedding using OpenAI-compatible endpoint."""
    if not OPENAI_API_KEY:
        return []
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding Error: {e}")
        return []


def search_memory(patient_id: str, query: str, limit: int = 5):
    """Retrieve relevant past conversations using vector similarity."""
    if not supabase:
        return []
    query_embedding = generate_embedding(query)
    if not query_embedding:
        return []
    try:
        response = supabase.rpc(
            "match_memory",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.5,
                "match_count": limit,
                "p_patient_id": patient_id
            }
        ).execute()
        return response.data
    except Exception as e:
        print(f"Memory search error: {e}")
        return []


def calculate_fusion_score(ml_prob: float, nss_score: int, nds_score: int):
    """
    Final Score = 0.4 * ML_prob + 0.3 * NSS_norm + 0.3 * NDS_norm
    """
    nss_norm = min(nss_score / 14.0, 1.0)
    nds_norm = min(nds_score / 23.0, 1.0)
    final_score = (0.4 * ml_prob) + (0.3 * nss_norm) + (0.3 * nds_norm)

    if final_score >= 0.7:
        decision = "PDN confirmed"
    elif final_score >= 0.4:
        decision = "Possible neuropathy"
    else:
        decision = "Likely healthy"

    return final_score, decision


def run_diagnostic_pipeline(patient_id: str, user_query: str, chat_history: list = None):
    """
    Full pipeline: fetch data → fuse scores → RAG retrieval → LLM → parse JSON.
    """
    if chat_history is None:
        chat_history = []

    # 1. Fetch clinical data
    clinical_data = get_patient_clinical_data(patient_id)
    nss_record = clinical_data.get("nss")
    nds_record = clinical_data.get("nds")
    nss_score = nss_record["total_score"] if nss_record else 0
    nds_score = nds_record["total_score"] if nds_record else 0

    # 2. Fetch ML prediction
    ml_record = get_patient_ml_prediction(patient_id)
    ml_prob = ml_record["predicted_probability"] if ml_record else 0.0
    ml_class = ml_record["predicted_class"] if ml_record else 0

    # 3. Fuse scores
    fusion_score, decision = calculate_fusion_score(ml_prob, nss_score, nds_score)

    # 4. RAG retrieval
    memory_context = search_memory(patient_id, user_query)
    context_text = "\n".join([f"- {m['content']}" for m in memory_context]) if memory_context else "No prior history found."

    # 5. Format recent chat history
    history_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history[-6:]])

    # 6. Build prompt
    prompt = f"""You are an expert AI Medical Decision System specializing in Diabetic Peripheral Neuropathy (DPN).
Your role is dual:
1. Ask the patient structured clinical questions one at a time to assess their condition.
2. Answer any questions the patient asks about their health, symptoms, or diagnosis.

You MUST output valid JSON only. No markdown, no extra text — only the JSON object.

Patient's last message: "{user_query}"

--- RECENT CONVERSATION ---
{history_text}

--- CLINICAL DATA ---
ML Prediction Probability: {ml_prob:.2f} (Class: {ml_class})
Clinical NSS Score: {nss_score} / 14
Clinical NDS Score: {nds_score} / 23
Fusion Score: {fusion_score:.2f}
System Decision: {decision}

--- LONG TERM MEMORY (RAG) ---
{context_text}

INSTRUCTIONS:
- Always respond in English.
- If the patient answered your previous question, continue the interview with the NEXT clinical question.
- If the patient asked YOU a question, answer it clearly and briefly, then continue the interview.
- Always include "suggested_answers": 2–4 options for answering your current question.
- Always include "suggested_questions": 2–3 questions the patient might want to ask next (e.g. about their condition, symptoms, or treatment).
- If you have enough information OR the patient requests a final diagnosis, set "is_diagnosis_complete" to true and write the full structured report in "message":

  🧾 Diagnosis: [decision]
  Confidence: [High / Medium / Low]
  🤖 ML Insight: Class {ml_class}, Probability {ml_prob:.0%}
  🧪 Clinical Scores: NSS={nss_score}/14, NDS={nds_score}/23
  📌 Explanation: [simple patient-friendly explanation]
  🚨 Disclaimer: This is AI-assisted analysis only. Always consult a certified physician.

Output ONLY this JSON:
{{
    "message": "...",
    "suggested_answers": ["Yes, I have that", "No, I don't", "Sometimes"],
    "suggested_questions": ["What does this score mean?", "Is my condition serious?", "What should I do next?"],
    "is_diagnosis_complete": false
}}"""

    # 7. Call LLM
    raw_response = call_llm(prompt)

    # 8. Parse JSON
    try:
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
    except Exception as e:
        print(f"JSON parse error: {e}\nRaw: {raw_response}")
        return {
            "message": raw_response,
            "suggested_answers": ["Continue", "Get my diagnosis"],
            "is_diagnosis_complete": False
        }
