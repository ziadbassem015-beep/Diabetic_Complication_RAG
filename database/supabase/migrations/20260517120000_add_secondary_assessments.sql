-- Additive migration: gestational diabetes and heart risk assessment tables
-- Safe to run on existing databases (IF NOT EXISTS)

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
    predicted_class INTEGER,
    predicted_probability FLOAT,
    severity TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS heart_risk_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    cholesterol FLOAT,
    blood_pressure_systolic FLOAT,
    blood_pressure_diastolic FLOAT,
    resting_heart_rate INTEGER,
    smoking_status TEXT,
    bmi FLOAT,
    diabetes_duration INTEGER,
    exercise_frequency INTEGER,
    risk_score FLOAT,
    predicted_class INTEGER,
    predicted_probability FLOAT,
    severity TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gestational_patient_id ON gestational_diabetes_assessments(patient_id);
CREATE INDEX IF NOT EXISTS idx_gestational_created_at ON gestational_diabetes_assessments(created_at);
CREATE INDEX IF NOT EXISTS idx_heart_risk_patient_id ON heart_risk_assessments(patient_id);
CREATE INDEX IF NOT EXISTS idx_heart_risk_created_at ON heart_risk_assessments(created_at);
