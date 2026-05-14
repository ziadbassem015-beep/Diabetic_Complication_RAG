"""
questionnaire.py — All questions extracted from the .ipynb files.
Each question has: id, section, question text, key for DB storage, and options (label + score value).
Sections: NSS → NDS → GUM → ULCER → ML_INPUTS (for neuropathy ML model + final decision)
"""

QUESTIONNAIRE = [

    # ─────────────────────────────────────────────
    # SECTION 1 — NSS (Neuropathy Symptom Score)
    # Max score = 14 (from NSS (1).ipynb)
    # ─────────────────────────────────────────────
    {
        "id": 1, "section": "NSS",
        "section_label": "🧠 Neuropathy Symptom Score (NSS)",
        "question": "Do you experience burning pain in your feet or legs?",
        "key": "nss_burning_pain",
        "options": [
            {"label": "No", "score": 0},
            {"label": "Yes", "score": 1},
        ]
    },
    {
        "id": 2, "section": "NSS",
        "section_label": "🧠 NSS",
        "question": "Does the burning pain get worse at night?",
        "key": "nss_burning_night",
        "options": [
            {"label": "No", "score": 0},
            {"label": "Yes", "score": 1},
        ]
    },
    {
        "id": 3, "section": "NSS",
        "section_label": "🧠 NSS",
        "question": "Is the burning pain severe?",
        "key": "nss_burning_severe",
        "options": [
            {"label": "No", "score": 0},
            {"label": "Yes (Severe)", "score": 1},
        ]
    },
    {
        "id": 4, "section": "NSS",
        "section_label": "🧠 NSS",
        "question": "Do you experience numbness (تنميل) in your feet?",
        "key": "nss_numbness",
        "options": [
            {"label": "No", "score": 0},
            {"label": "Yes", "score": 1},
        ]
    },
    {
        "id": 5, "section": "NSS",
        "section_label": "🧠 NSS",
        "question": "Does the numbness get worse at night?",
        "key": "nss_numbness_night",
        "options": [
            {"label": "No", "score": 0},
            {"label": "Yes", "score": 1},
        ]
    },
    {
        "id": 6, "section": "NSS",
        "section_label": "🧠 NSS",
        "question": "Do you feel tingling / pins & needles (وخز بالإبر) in your feet?",
        "key": "nss_tingling",
        "options": [
            {"label": "No", "score": 0},
            {"label": "Yes", "score": 1},
        ]
    },
    {
        "id": 7, "section": "NSS",
        "section_label": "🧠 NSS",
        "question": "Do you feel fatigue or heaviness in your feet?",
        "key": "nss_fatigue",
        "options": [
            {"label": "No", "score": 0},
            {"label": "Yes", "score": 1},
        ]
    },
    {
        "id": 8, "section": "NSS",
        "section_label": "🧠 NSS",
        "question": "Do you have paresthesia (crawling / ant-like sensation in your skin)?",
        "key": "nss_paresthesia",
        "options": [
            {"label": "No", "score": 0},
            {"label": "Yes", "score": 1},
        ]
    },
    {
        "id": 9, "section": "NSS",
        "section_label": "🧠 NSS",
        "question": "Do you feel pain when something light touches your skin (Allodynia)?",
        "key": "nss_allodynia",
        "options": [
            {"label": "No", "score": 0},
            {"label": "Yes", "score": 1},
        ]
    },

    # ─────────────────────────────────────────────
    # SECTION 2 — NDS (Neuropathy Disability Score)
    # Max score = 23 (from NDS(1) (1).ipynb)
    # ─────────────────────────────────────────────
    {
        "id": 10, "section": "NDS",
        "section_label": "🦶 Neuropathy Disability Score (NDS)",
        "question": "Vibration Sensation Test: Place a vibrating phone or electric shaver on your big toe, then your ankle. How quickly do you feel the vibration?",
        "key": "nds_vibration",
        "options": [
            {"label": "✅ Normal (≤ 2 seconds)", "score": 0},
            {"label": "⚠️ Weak (3–5 seconds)", "score": 2},
            {"label": "❌ Delayed (6–10 seconds)", "score": 3},
            {"label": "❌ Absent (> 10 seconds)", "score": 6},
        ]
    },
    {
        "id": 11, "section": "NDS",
        "section_label": "🦶 NDS",
        "question": "Temperature Sensation Test (Right foot, Warm water): Dip your right foot in warm water. What temperature do you start to feel warmth?",
        "key": "nds_temp_right_warm",
        "options": [
            {"label": "✅ Felt it easily (Normal, ≤ 39°C)", "score": 0},
            {"label": "⚠️ Needed warmer water (40–44°C)", "score": 2},
            {"label": "❌ Only felt it very hot (> 44°C)", "score": 3},
            {"label": "❌ Could not feel the temperature at all", "score": 5},
        ]
    },
    {
        "id": 12, "section": "NDS",
        "section_label": "🦶 NDS",
        "question": "Temperature Sensation Test (Right foot, Cold water): Dip your right foot in cold water. What temperature do you start to feel coldness?",
        "key": "nds_temp_right_cold",
        "options": [
            {"label": "✅ Felt it easily (Normal, ≥ 15°C)", "score": 0},
            {"label": "⚠️ Needed colder water (10–14°C)", "score": 2},
            {"label": "❌ Only felt it ice cold (< 10°C)", "score": 3},
            {"label": "❌ Could not feel the temperature at all", "score": 5},
        ]
    },
    {
        "id": 13, "section": "NDS",
        "section_label": "🦶 NDS",
        "question": "Pain Sensation Test: Using a pin or safety pin, lightly press 5 spots on your foot. What do you feel?",
        "key": "nds_pain",
        "options": [
            {"label": "✅ Normal pain sensation", "score": 0},
            {"label": "⚠️ Weak / dull pain", "score": 2},
            {"label": "❌ Felt only touch, no pain", "score": 4},
            {"label": "❌ No sensation at all", "score": 6},
        ]
    },
    {
        "id": 14, "section": "NDS",
        "section_label": "🦶 NDS",
        "question": "Light Touch Test: Use a cotton ball and lightly touch 5 spots on your foot. What do you feel?",
        "key": "nds_touch",
        "options": [
            {"label": "✅ Normal — felt everything", "score": 0},
            {"label": "⚠️ Weak — missed some spots", "score": 1},
            {"label": "❌ Intermittent — felt it sometimes", "score": 2},
            {"label": "❌ Absent — felt nothing", "score": 3},
        ]
    },
    {
        "id": 15, "section": "NDS",
        "section_label": "🦶 NDS",
        "question": "Balance / Reflex Test: Try standing on your toes for 10 seconds, then walk on your heels. How is your balance?",
        "key": "nds_reflex",
        "options": [
            {"label": "✅ Normal — no problem", "score": 0},
            {"label": "⚠️ Slight difficulty", "score": 1},
            {"label": "❌ Clear difficulty", "score": 2},
            {"label": "❌ Unable to do it", "score": 3},
        ]
    },

    # ─────────────────────────────────────────────
    # SECTION 3 — Gum Health / الحنك (الحنك.ipynb)
    # Max score = 23
    # ─────────────────────────────────────────────
    {
        "id": 16, "section": "GUM",
        "section_label": "🦷 Gum Health Assessment",
        "question": "What is the color of your gums?",
        "key": "gum_color",
        "options": [
            {"label": "🩷 Normal pink (Healthy)", "score": 0},
            {"label": "❤️ Bright red (Inflamed)", "score": 1},
            {"label": "💜 Dark red / purple (Severe)", "score": 2},
        ]
    },
    {
        "id": 17, "section": "GUM",
        "section_label": "🦷 Gum Health",
        "question": "Do your gums bleed when you brush or spontaneously?",
        "key": "gum_bleeding",
        "options": [
            {"label": "No bleeding", "score": 0},
            {"label": "Slight (only when brushing)", "score": 1},
            {"label": "Obvious (frequent bleeding)", "score": 2},
            {"label": "Spontaneous (bleeds on its own)", "score": 3},
        ]
    },
    {
        "id": 18, "section": "GUM",
        "section_label": "🦷 Gum Health",
        "question": "Are your gums swollen?",
        "key": "gum_swelling",
        "options": [
            {"label": "No swelling", "score": 0},
            {"label": "Mild swelling", "score": 1},
            {"label": "Severe swelling", "score": 2},
        ]
    },
    {
        "id": 19, "section": "GUM",
        "section_label": "🦷 Gum Health",
        "question": "Do you have tartar / calculus buildup on your teeth?",
        "key": "gum_calculus",
        "options": [
            {"label": "None", "score": 0},
            {"label": "Mild", "score": 1},
            {"label": "Heavy", "score": 2},
        ]
    },
    {
        "id": 20, "section": "GUM",
        "section_label": "🦷 Gum Health",
        "question": "Do you have bad breath (halitosis)?",
        "key": "gum_breath",
        "options": [
            {"label": "No", "score": 0},
            {"label": "Yes", "score": 1},
        ]
    },
    {
        "id": 21, "section": "GUM",
        "section_label": "🦷 Gum Health",
        "question": "Are you a smoker?",
        "key": "gum_smoking",
        "options": [
            {"label": "No", "score": 0},
            {"label": "Yes", "score": 1},
        ]
    },

    # ─────────────────────────────────────────────
    # SECTION 4 — Ulcer Screening (ulcer (1).ipynb)
    # ─────────────────────────────────────────────
    {
        "id": 22, "section": "ULCER",
        "section_label": "🩹 Diabetic Foot Ulcer Screening",
        "question": "Do you have any wounds on your feet that do not heal?",
        "key": "ulcer_wound",
        "options": [
            {"label": "No", "score": 0},
            {"label": "Yes", "score": 3},
        ]
    },
    {
        "id": 23, "section": "ULCER",
        "section_label": "🩹 Ulcer Screening",
        "question": "Do you feel numbness or loss of sensation in your foot?",
        "key": "ulcer_numbness",
        "options": [
            {"label": "No", "score": 0},
            {"label": "Yes", "score": 2},
        ]
    },
    {
        "id": 24, "section": "ULCER",
        "section_label": "🩹 Ulcer Screening",
        "question": "Is there pain in the foot?",
        "key": "ulcer_pain",
        "options": [
            {"label": "Yes, I feel pain", "score": 0},
            {"label": "No pain (painless wound)", "score": 2},
        ]
    },
    {
        "id": 25, "section": "ULCER",
        "section_label": "🩹 Ulcer Screening",
        "question": "Is there any change in skin color on your foot?",
        "key": "ulcer_color",
        "options": [
            {"label": "No change", "score": 0},
            {"label": "Yes (redness, darkening, or paleness)", "score": 2},
        ]
    },
    {
        "id": 26, "section": "ULCER",
        "section_label": "🩹 Ulcer Screening",
        "question": "Is there any swelling or redness on your foot?",
        "key": "ulcer_swelling",
        "options": [
            {"label": "No", "score": 0},
            {"label": "Yes", "score": 2},
        ]
    },
    {
        "id": 27, "section": "ULCER",
        "section_label": "🩹 Ulcer Screening",
        "question": "Is there any discharge or bad odor from your foot?",
        "key": "ulcer_discharge",
        "options": [
            {"label": "No", "score": 0},
            {"label": "Yes", "score": 3},
        ]
    },
    {
        "id": 28, "section": "ULCER",
        "section_label": "🩹 Ulcer Screening",
        "question": "Do you wear tight shoes regularly?",
        "key": "ulcer_shoes",
        "options": [
            {"label": "No", "score": 0},
            {"label": "Yes", "score": 1},
        ]
    },
    # ─────────────────────────────────────────────
    # SECTION 5 — ML Clinical Inputs
    # (for diabetic neuropathy(90)code + final_decision notebooks)
    # Features: BMI, HbA1c, heat/cold sensitivity
    # ─────────────────────────────────────────────
    {
        "id": 29, "section": "ML",
        "section_label": "🤖 ML Clinical Inputs (Neuropathy Prediction)",
        "question": "What is your approximate BMI (Body Mass Index)?",
        "key": "ml_bmi",
        "options": [
            {"label": "Underweight (< 18.5)", "score": 17},
            {"label": "Normal (18.5–24.9)", "score": 22},
            {"label": "Overweight (25–29.9)", "score": 27},
            {"label": "Obese (≥ 30)", "score": 33},
        ]
    },
    {
        "id": 30, "section": "ML",
        "section_label": "🤖 ML Clinical Inputs",
        "question": "What is your most recent HbA1c level (blood sugar control test)?",
        "key": "ml_hba1c",
        "options": [
            {"label": "✅ Well controlled (< 6.5%)", "score": 6},
            {"label": "⚠️ Borderline (6.5–7.5%)", "score": 7},
            {"label": "❌ Poorly controlled (7.6–9%)", "score": 8},
            {"label": "❌ Very high (> 9%)", "score": 10},
        ]
    },
    {
        "id": 31, "section": "ML",
        "section_label": "🤖 ML Clinical Inputs",
        "question": "Heat sensitivity — your RIGHT foot: At what temperature does warm water feel warm to you?",
        "key": "ml_heat_right",
        "options": [
            {"label": "Normal (≤ 39°C)", "score": 38},
            {"label": "Slightly reduced (40–42°C)", "score": 41},
            {"label": "Clearly reduced (43–45°C)", "score": 44},
            {"label": "Very reduced / absent (> 45°C)", "score": 47},
        ]
    },
    {
        "id": 32, "section": "ML",
        "section_label": "🤖 ML Clinical Inputs",
        "question": "Heat sensitivity — your LEFT foot: At what temperature does warm water feel warm to you?",
        "key": "ml_heat_left",
        "options": [
            {"label": "Normal (≤ 39°C)", "score": 38},
            {"label": "Slightly reduced (40–42°C)", "score": 41},
            {"label": "Clearly reduced (43–45°C)", "score": 44},
            {"label": "Very reduced / absent (> 45°C)", "score": 47},
        ]
    },
    {
        "id": 33, "section": "ML",
        "section_label": "🤖 ML Clinical Inputs",
        "question": "Cold sensitivity — your RIGHT foot: At what temperature does cold water feel cold to you?",
        "key": "ml_cold_right",
        "options": [
            {"label": "Normal (≥ 15°C)", "score": 20},
            {"label": "Slightly reduced (10–14°C)", "score": 12},
            {"label": "Clearly reduced (5–9°C)", "score": 7},
            {"label": "Very reduced / absent (< 5°C)", "score": 3},
        ]
    },
    {
        "id": 34, "section": "ML",
        "section_label": "🤖 ML Clinical Inputs",
        "question": "Cold sensitivity — your LEFT foot: At what temperature does cold water feel cold to you?",
        "key": "ml_cold_left",
        "options": [
            {"label": "Normal (≥ 15°C)", "score": 20},
            {"label": "Slightly reduced (10–14°C)", "score": 12},
            {"label": "Clearly reduced (5–9°C)", "score": 7},
            {"label": "Very reduced / absent (< 5°C)", "score": 3},
        ]
    },
]


def calculate_section_scores(answers: dict) -> dict:
    """Calculate total score for each section."""
    nss_keys = [q["key"] for q in QUESTIONNAIRE if q["section"] == "NSS"]
    nds_keys = [q["key"] for q in QUESTIONNAIRE if q["section"] == "NDS"]
    gum_keys = [q["key"] for q in QUESTIONNAIRE if q["section"] == "GUM"]
    ulcer_keys = [q["key"] for q in QUESTIONNAIRE if q["section"] == "ULCER"]

    nss_score = sum(answers.get(k, 0) for k in nss_keys)
    nds_score = sum(answers.get(k, 0) for k in nds_keys)
    gum_score = sum(answers.get(k, 0) for k in gum_keys)
    ulcer_score = sum(answers.get(k, 0) for k in ulcer_keys)

    return {
        "nss_score": min(nss_score, 14),
        "nds_score": min(nds_score, 23),
        "gum_score": gum_score,
        "ulcer_score": ulcer_score,
    }


def ml_neuropathy_prediction(answers: dict, nss_score: int, age: int) -> dict:
    """
    Approximates the Random Forest ML model from diabetic neuropathy(90)code (1).ipynb
    using the clinical inputs provided by the patient.
    Features used: NSS, heat_right, heat_left, cold_right, cold_left, BMI, Age, HbA1c
    """
    bmi = answers.get("ml_bmi", 22)
    hba1c = answers.get("ml_hba1c", 7)
    heat_right = answers.get("ml_heat_right", 38)
    heat_left = answers.get("ml_heat_left", 38)
    cold_right = answers.get("ml_cold_right", 20)
    cold_left = answers.get("ml_cold_left", 20)

    # Approximate feature importance scoring (based on notebook feature importance ranking)
    # NSS (most important: 15.4%)
    nss_risk = min(nss_score / 14.0, 1.0)
    # BMI/Age ratio
    bmi_age_ratio = bmi / max(age, 1)
    bmi_risk = min((bmi - 18.5) / 20.0, 1.0) if bmi > 18.5 else 0
    # Temperature sensitivity
    heat_avg = (heat_right + heat_left) / 2
    cold_avg = (cold_right + cold_left) / 2
    heat_risk = min((heat_avg - 37) / 15.0, 1.0)
    cold_risk = min((37 - cold_avg) / 20.0, 1.0)
    # HbA1c risk
    hba1c_risk = min((hba1c - 6.5) / 5.0, 1.0) if hba1c > 6.5 else 0

    # Weighted probability (based on feature importances from the notebook)
    probability = (
        0.154 * nss_risk +
        0.131 * min(bmi_age_ratio / 0.5, 1.0) +
        0.110 * bmi_risk +
        0.108 * heat_risk +
        0.093 * hba1c_risk +
        0.077 * heat_risk +  # heat_avg feature
        0.079 * cold_risk +  # cold_right
        0.082 * cold_risk +  # cold_left
        0.067 * cold_risk +  # cold_avg
        0.059 * heat_risk    # heat_left
    )
    probability = min(max(probability, 0.0), 1.0)
    predicted_class = 1 if probability >= 0.5 else 0

    return {
        "predicted_class": predicted_class,
        "predicted_probability": round(probability, 3),
        "ai_prediction": "مريض" if predicted_class == 1 else "سليم",
        "features": {
            "nss": nss_score, "bmi": bmi, "age": age,
            "hba1c": hba1c, "heat_right": heat_right, "heat_left": heat_left,
            "cold_right": cold_right, "cold_left": cold_left,
        }
    }


def final_decision(ai_prediction: str, nds_score: int, nss_score: int) -> dict:
    """
    Exact implementation of final_decision (1).ipynb weighted scoring formula.
    weight_nds = 1.2, weight_ai = 1.0, weight_nss = 0.9
    """
    ai_binary = 1 if ai_prediction == "مريض" else 0
    nds_binary = 1 if nds_score >= 6 else 0
    nss_binary = 1 if nss_score >= 5 else 0

    weight_nds = 1.2
    weight_ai = 1.0
    weight_nss = 0.9

    score = (ai_binary * weight_ai) + (nds_binary * weight_nds) + (nss_binary * weight_nss)
    threshold = (weight_ai + weight_nds + weight_nss) / 2  # = 1.55

    if score >= threshold:
        decision = "PDN Confirmed (اعتلال الأعصاب المحيطية السكري مؤكد)"
        confidence = "High" if score >= 2.5 else "Medium"
    else:
        decision = "Likely Healthy (سليم على الأغلب)"
        confidence = "High" if score == 0 else "Medium"

    return {
        "final_decision": decision,
        "confidence": confidence,
        "fusion_score": round(score, 2),
        "threshold": threshold,
        "ai_binary": ai_binary,
        "nds_binary": nds_binary,
        "nss_binary": nss_binary,
    }
