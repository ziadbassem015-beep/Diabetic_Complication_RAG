"""
test_db.py — Repository-based connectivity and schema validation test.
Run with: python test_db.py
"""

from dotenv import load_dotenv
load_dotenv()

from core.repositories.patient_repo import PatientRepository
from core.repositories.clinical_repo import ClinicalRepository
from core.repositories.ml_repo import MLRepository
from core.repositories.decision_repo import DecisionRepository

print("=" * 45)
print("   Repository Connectivity Test")
print("=" * 45)

# Test 1: Patients
try:
    patients = PatientRepository.get_all_patients()
    print(f"\nPatients in DB: {len(patients)}")
    for p in patients[:10]:
        print(
            f"  - {p.get('name', 'Unknown')} | "
            f"age: {p.get('age', '?')} | "
            f"id: {str(p.get('id', ''))[:8]}..."
        )
except Exception as e:
    print(f"[ERROR] PatientRepository failure: {e}")
    exit(1)

# Test 2: Repository access checks
try:
    clinical = ClinicalRepository.get_clinical_data("00000000-0000-0000-0000-000000000000")
    print(f"\nClinical data keys: {list(clinical.keys())}")
    ml = MLRepository.get_ml_prediction("00000000-0000-0000-0000-000000000000")
    print(f"ML prediction keys: {list(ml.keys()) if isinstance(ml, dict) else type(ml)}")
    decisions = DecisionRepository.get_decisions("00000000-0000-0000-0000-000000000000")
    print(f"Decision records returned: {len(decisions)}")
except Exception as e:
    print(f"[ERROR] Repository access failed: {e}")
    exit(1)

print("\n" + "=" * 45)
print("Repository connectivity check completed.")
print("=" * 45)
