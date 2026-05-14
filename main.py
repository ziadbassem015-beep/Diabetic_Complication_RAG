import uuid
import sys
from dotenv import load_dotenv
from core.database import get_all_patients, save_conversation_memory, supabase
from core.rag_engine import run_diagnostic_pipeline, generate_embedding

load_dotenv()

def select_patient():
    """Let user pick or create a patient."""
    patients = get_all_patients()
    
    print("\n" + "="*55)
    print("   🩺 AI Diagnostic System - Diabetic Neuropathy")
    print("="*55)

    if patients:
        print("\n📋 Available Patients:")
        for i, p in enumerate(patients, 1):
            print(f"  {i}. {p['name']} (Age: {p.get('age', '?')}, Gender: {p.get('gender', '?')})")
        print(f"  {len(patients)+1}. ➕ Create New Patient")
        
        while True:
            try:
                choice = int(input("\nSelect patient number: "))
                if 1 <= choice <= len(patients):
                    return patients[choice - 1]
                elif choice == len(patients) + 1:
                    return create_patient()
                else:
                    print("⚠️  Please enter a valid number.")
            except ValueError:
                print("⚠️  Please enter a valid number.")
    else:
        print("\n⚠️  No patients found in the database.")
        return create_patient()


def create_patient():
    """Create a new patient in the database."""
    print("\n--- Create New Patient ---")
    name = input("Patient Name: ").strip() or "New Patient"
    try:
        age = int(input("Age: "))
    except ValueError:
        age = 0
    gender = input("Gender (Male/Female): ").strip() or "Not specified"
    
    new_id = str(uuid.uuid4())
    result = supabase.table("patients").insert({
        "id": new_id,
        "name": name,
        "age": age,
        "gender": gender
    }).execute()
    
    patient = result.data[0]
    print(f"\n✅ Patient created: {patient['name']}")
    return patient


def display_options(options: list):
    """Display numbered answer options."""
    print("\n  Your options:")
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")
    print()


def get_choice(options: list) -> str:
    """Get the user's choice from numbered options."""
    while True:
        try:
            raw = input("Enter your choice (number): ").strip()
            choice = int(raw)
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print(f"⚠️  Enter a number between 1 and {len(options)}.")
        except ValueError:
            print("⚠️  Please enter a valid number.")


def run_chat(patient: dict):
    """Main chat loop with the AI diagnostic system."""
    patient_id = patient["id"]
    session_id = str(uuid.uuid4())
    chat_history = []
    
    print(f"\n✅ Welcome, {patient['name']}! I will help you with your diagnosis.")
    print("─" * 55)
    
    current_input = "Hello, I want to start the diagnosis."
    
    while True:
        # Save user message
        chat_history.append({"role": "user", "content": current_input})
        user_emb = generate_embedding(current_input)
        save_conversation_memory(patient_id, session_id, "user", current_input, user_emb if user_emb else None)
        
        print("\n⏳ Analyzing...")
        
        # Get AI response
        result = run_diagnostic_pipeline(patient_id, current_input, chat_history)
        
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
            break
        
        message = result.get("message", "")
        options = result.get("suggested_answers", [])
        is_done = result.get("is_diagnosis_complete", False)
        
        # Save assistant message
        chat_history.append({"role": "assistant", "content": message})
        asst_emb = generate_embedding(message)
        save_conversation_memory(patient_id, session_id, "assistant", message, asst_emb if asst_emb else None)
        
        # Display AI message
        print("\n🤖 AI System:")
        print(f"\n{message}")
        
        if is_done:
            print("\n" + "="*55)
            print("✅ Diagnosis complete. Thank you!")
            print("="*55)
            break
        
        if options:
            display_options(options)
            current_input = get_choice(options)
            print(f"\n👤 Your answer: {current_input}")
            print("─" * 55)
        else:
            current_input = input("\n👤 Your response: ").strip()
            if not current_input:
                break


def main():
    try:
        patient = select_patient()
        run_chat(patient)
    except KeyboardInterrupt:
        print("\n\n👋 Session ended.")
        sys.exit(0)


if __name__ == "__main__":
    main()
