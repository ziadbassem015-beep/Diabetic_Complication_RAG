import streamlit as st
import uuid
from database import get_all_patients, save_conversation_memory, supabase
from rag_engine import generate_embedding, call_llm
from questionnaire import QUESTIONNAIRE, calculate_section_scores, ml_neuropathy_prediction, final_decision

st.set_page_config(page_title="Diabetic Diagnostic System", page_icon="🩺", layout="centered")

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 750px; margin: auto; }
    .section-badge {
        background: #1e3a5f; color: white; padding: 6px 14px;
        border-radius: 20px; font-size: 13px; display: inline-block;
        margin-bottom: 10px;
    }
    .question-box {
        background: #f0f4ff; border-left: 5px solid #1e3a5f;
        padding: 18px 20px; border-radius: 10px;
        font-size: 18px; font-weight: 600; margin-bottom: 20px;
        color: #000000 !important;
    }
    .progress-text { color: #888; font-size: 13px; margin-bottom: 6px; }
    .final-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        color: white; padding: 30px; border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────
for key, default in {
    "session_id": str(uuid.uuid4()),
    "step": 0,
    "answers": {},
    "patient": None,
    "analysis": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar — Patient Selection ───────────────────────────────
st.sidebar.title("🩺 Diagnostic System")
st.sidebar.markdown("---")
st.sidebar.subheader("Patient")

patients = get_all_patients()
if patients:
    options = {f"{p['name']} (Age {p.get('age','?')})": p for p in patients}
    choice = st.sidebar.selectbox("Select patient", list(options.keys()))
    if st.session_state.patient != options[choice]:
        st.session_state.patient = options[choice]
        st.session_state.step = 0
        st.session_state.answers = {}
        st.session_state.analysis = None
else:
    st.sidebar.warning("No patients found.")
    if st.sidebar.button("➕ Create Test Patient"):
        new_id = str(uuid.uuid4())
        result = supabase.table("patients").insert({
            "id": new_id, "name": "Test Patient", "age": 50, "gender": "Not specified"
        }).execute()
        st.session_state.patient = result.data[0]
        st.rerun()

if st.sidebar.button("🔄 Restart Assessment"):
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.analysis = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.metric("Progress", f"{st.session_state.step} / {len(QUESTIONNAIRE)}")

# ── Main UI ───────────────────────────────────────────────────
st.title("🩺 Diabetic Complication Assessment")

if not st.session_state.patient:
    st.info("👈 Please select or create a patient from the sidebar to begin.")
    st.stop()

patient = st.session_state.patient
step = st.session_state.step
total = len(QUESTIONNAIRE)

# ── Progress Bar ──────────────────────────────────────────────
st.markdown(f'<div class="progress-text">Question {step} of {total}</div>', unsafe_allow_html=True)
st.progress(step / total)

# ── FINAL ANALYSIS (after all questions answered) ─────────────
if step >= total:
    st.success("✅ All questions answered! Generating your analysis...")

    if st.session_state.analysis is None:
        scores = calculate_section_scores(st.session_state.answers)
        nss = scores["nss_score"]
        nds = scores["nds_score"]
        gum = scores["gum_score"]
        ulcer = scores["ulcer_score"]

        nss_cat = "Normal" if nss <= 2 else "Mild" if nss <= 4 else "Moderate" if nss <= 6 else "Severe"
        nds_cat = "Normal" if nds <= 5 else "Mild" if nds <= 10 else "Moderate" if nds <= 16 else "Severe"
        gum_cat = "Healthy" if gum <= 5 else "Mild gingivitis" if gum <= 11 else "Moderate gingivitis" if gum <= 18 else "Severe gingivitis"
        ulcer_cat = "Low risk" if ulcer < 4 else "Moderate risk" if ulcer < 8 else "High risk"

        # Run ML Neuropathy Prediction (from diabetic neuropathy(90)code notebook)
        age = patient.get("age", 50)
        ml_result = ml_neuropathy_prediction(st.session_state.answers, nss, age)

        # Run Final Decision (from final_decision notebook)
        fd_result = final_decision(ml_result["ai_prediction"], nds, nss)

        prompt = f"""You are an expert AI Medical Decision System for Diabetic Complications.
Generate a clear, structured, patient-friendly medical analysis report in English.

Patient: {patient['name']}, Age: {age}

=== SECTION SCORES ===
NSS (Neuropathy Symptom Score): {nss}/14 → {nss_cat}
NDS (Neuropathy Disability Score): {nds}/23 → {nds_cat}
Gum Health Score: {gum} → {gum_cat}
Ulcer Risk Score: {ulcer} → {ulcer_cat}

=== ML MODEL RESULT (from Neuropathy Prediction model) ===
Predicted Class: {ml_result['predicted_class']} ({'Neuropathy' if ml_result['predicted_class'] == 1 else 'Healthy'})
Probability: {ml_result['predicted_probability']*100:.1f}%
BMI: {ml_result['features']['bmi']}, HbA1c: {ml_result['features']['hba1c']}%

=== FINAL DECISION (1.0×AI + 1.2×NDS + 0.9×NSS formula) ===
Fusion score: {fd_result['fusion_score']} / threshold: {fd_result['threshold']}
🏁 FINAL DECISION: {fd_result['final_decision']}
Confidence: {fd_result['confidence']}

Write a full medical report with:
1. 🏁 Final Diagnosis (one clear sentence with the final decision)
2. 🧠 Neuropathy Assessment (NSS + NDS + ML model)
3. 🦷 Gum Health Assessment
4. 🩹 Foot Ulcer Risk Assessment
5. 📊 How the decision was calculated (brief, simple)
6. 📌 Recommendations (3-5 bullet points)
7. 🚨 Disclaimer: This is AI-assisted analysis only. Always consult a certified physician."""

        with st.spinner("🤖 AI is generating your full analysis..."):
            st.session_state.analysis = call_llm(prompt)

        # Save all results to Supabase
        try:
            supabase.table("nss_assessments").insert({"patient_id": patient["id"], "total_score": nss, "severity": nss_cat, "symptoms_details": {k: v for k, v in st.session_state.answers.items() if k.startswith("nss_")}}).execute()
            supabase.table("nds_assessments").insert({"patient_id": patient["id"], "total_score": nds, "severity": nds_cat, "test_details": {k: v for k, v in st.session_state.answers.items() if k.startswith("nds_")}}).execute()
            supabase.table("gum_assessments").insert({"patient_id": patient["id"], "total_score": gum, "status": gum_cat, "clinical_signs": {k: v for k, v in st.session_state.answers.items() if k.startswith("gum_")}}).execute()
            supabase.table("ulcer_assessments").insert({"patient_id": patient["id"], "ulcer_stage": ulcer_cat, "infection_signs": {k: v for k, v in st.session_state.answers.items() if k.startswith("ulcer_")}}).execute()
            supabase.table("ml_neuropathy_predictions").insert({"patient_id": patient["id"], "nss_score": nss, "bmi_baseline": ml_result["features"]["bmi"], "age_baseline": age, "hba1c_baseline": ml_result["features"]["hba1c"], "heat_avg": (ml_result["features"]["heat_right"] + ml_result["features"]["heat_left"]) / 2, "cold_avg": (ml_result["features"]["cold_right"] + ml_result["features"]["cold_left"]) / 2, "predicted_class": ml_result["predicted_class"], "predicted_probability": ml_result["predicted_probability"]}).execute()
            supabase.table("final_diagnostic_decisions").insert({"patient_id": patient["id"], "ai_prediction": ml_result["ai_prediction"], "nds_score": nds, "nss_score": nss, "calculated_score": fd_result["fusion_score"], "final_decision": fd_result["final_decision"]}).execute()
        except Exception as e:
            st.warning(f"Could not save to database: {e}")

    st.markdown(f"""
    <div class="final-card">
    {st.session_state.analysis.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── CURRENT QUESTION ──────────────────────────────────────────
q = QUESTIONNAIRE[step]

st.markdown(f'<div class="section-badge">{q["section_label"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="question-box">❓ {q["question"]}</div>', unsafe_allow_html=True)

# ── ANSWER BUTTONS ─────────────────────────────────────────────
cols = st.columns(len(q["options"]))
for i, opt in enumerate(q["options"]):
    with cols[i]:
        if st.button(opt["label"], key=f"opt_{step}_{i}", use_container_width=True):
            # Save answer score
            st.session_state.answers[q["key"]] = opt["score"]
            st.session_state.step += 1
            st.rerun()
