-- Enable the pgvector extension for vector search (RAG)
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Patients Table
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. NSS (Neuropathy Symptom Score) Table
CREATE TABLE IF NOT EXISTS nss_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    total_score INTEGER,
    severity TEXT,
    symptoms_details JSONB, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. NDS (Neuropathy Disability Score) Table
CREATE TABLE IF NOT EXISTS nds_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    total_score INTEGER,
    severity TEXT,
    test_details JSONB, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Gum Health / الحنك Table
CREATE TABLE IF NOT EXISTS gum_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    total_score INTEGER,
    status TEXT,
    clinical_signs JSONB, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. Ulcer Table
CREATE TABLE IF NOT EXISTS ulcer_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    ulcer_stage TEXT,
    location TEXT,
    size_cm FLOAT,
    infection_signs JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 6. ML Neuropathy Predictions (From Random Forest Model)
CREATE TABLE IF NOT EXISTS ml_neuropathy_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    nss_score INTEGER,
    bmi_baseline FLOAT,
    age_baseline FLOAT,
    hba1c_baseline FLOAT,
    heat_avg FLOAT,
    cold_avg FLOAT,
    predicted_class INTEGER, -- 0 or 1
    predicted_probability FLOAT, -- Percentage
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 7. Final Diagnostic Decisions (Combines AI + NDS + NSS)
CREATE TABLE IF NOT EXISTS final_diagnostic_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    ai_prediction TEXT, -- "مريض" أو "سليم"
    nds_score INTEGER,
    nss_score INTEGER,
    calculated_score FLOAT, -- The weighted score
    final_decision TEXT, -- "PDN مؤكد" أو "سليم على الأغلب"
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 8. Conversation Memory Table (The Long Term Memory for the Chatbot RAG)
CREATE TABLE IF NOT EXISTS conversation_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL, -- The actual message text
    embedding VECTOR(1536), -- Vector representation of the content for RAG
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 9. Knowledge Base Table (For static medical documents/guidelines)
CREATE TABLE IF NOT EXISTS knowledge_base (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 10. Gestational Diabetes Assessment (Female patients only)
CREATE TABLE IF NOT EXISTS gestational_diabetes_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    pregnancy_week INTEGER,
    glucose_level FLOAT,
    fasting_glucose FLOAT,
    insulin_resistance FLOAT,
    bmi FLOAT,
    family_history BOOLEAN,
    risk_score FLOAT,
    predicted_class INTEGER, -- 0 = low risk, 1 = high risk
    predicted_probability FLOAT,
    severity TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 11. Heart Risk Assessment (All patients)
CREATE TABLE IF NOT EXISTS heart_risk_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    cholesterol FLOAT,
    blood_pressure_systolic FLOAT,
    blood_pressure_diastolic FLOAT,
    resting_heart_rate INTEGER,
    smoking_status TEXT, -- 'never', 'former', 'current'
    bmi FLOAT,
    diabetes_duration INTEGER,
    exercise_frequency INTEGER, -- times per week
    risk_score FLOAT,
    predicted_class INTEGER, -- 0 = low risk, 1 = medium risk, 2 = high risk
    predicted_probability FLOAT,
    severity TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_gestational_patient_id ON gestational_diabetes_assessments(patient_id);
CREATE INDEX IF NOT EXISTS idx_gestational_created_at ON gestational_diabetes_assessments(created_at);
CREATE INDEX IF NOT EXISTS idx_heart_risk_patient_id ON heart_risk_assessments(patient_id);
CREATE INDEX IF NOT EXISTS idx_heart_risk_created_at ON heart_risk_assessments(created_at);

-- Create a function to search for relevant knowledge/memory (for RAG)
CREATE OR REPLACE FUNCTION match_memory (
    query_embedding VECTOR(1536),
    match_threshold FLOAT,
    match_count INT,
    p_patient_id UUID DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        cm.id,
        cm.content,
        1 - (cm.embedding <=> query_embedding) AS similarity
    FROM conversation_memory cm
    WHERE 
        (p_patient_id IS NULL OR cm.patient_id = p_patient_id)
        AND 1 - (cm.embedding <=> query_embedding) > match_threshold
    ORDER BY cm.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
