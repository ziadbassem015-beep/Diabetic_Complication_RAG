"""
test_db.py — Quick test to verify Supabase connection and table row counts.
Run with: uv run python test_db.py
"""
from dotenv import load_dotenv
load_dotenv()
from database import supabase, get_all_patients

print("=" * 45)
print("   Supabase Connection Test")
print("=" * 45)

# Test 1: Patients
patients = get_all_patients()
print(f"\nPatients in DB: {len(patients)}")
for p in patients:
    print(f"  - {p['name']} | age: {p.get('age')} | id: {p['id'][:8]}...")

# Test 2: Table row counts
tables = [
    "nss_assessments",
    "nds_assessments",
    "gum_assessments",
    "ulcer_assessments",
    "ml_neuropathy_predictions",
    "final_diagnostic_decisions",
    "conversation_memory",
]

print("\nTable Row Counts:")
for t in tables:
    try:
        res = supabase.table(t).select("id", count="exact").execute()
        print(f"  [OK] {t}: {res.count} rows")
    except Exception as e:
        print(f"  [ERR] {t}: {e}")

print("\n" + "=" * 45)
print("If all tables show [OK], your DB is connected!")
print("Results are saved automatically after answering")
print("all 34 questions in the Streamlit app.")
print("=" * 45)
