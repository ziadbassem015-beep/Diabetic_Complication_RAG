"""
app.py — Streamlit UI for the Autonomous Medical Diagnostic Agent.
The agent drives the conversation; the UI displays it.
"""
import streamlit as st
import uuid
from database import get_all_patients, supabase
from agent import create_agent_session, MedicalDiagnosticAgent
from agent_state import AgentState

st.set_page_config(
    page_title="Diabetic RAG Diagnostic Agent",
    page_icon="🤖",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; }
.question-box {
    background: #f0f4ff;
    border-left: 5px solid #1e3a5f;
    padding: 18px 20px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 600;
    color: #000000 !important;
    margin-bottom: 16px;
}
.thought-box {
    background: #fff8e1;
    border-left: 4px solid #f59e0b;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 13px;
    color: #444;
    margin: 6px 0;
    font-style: italic;
}
.tool-badge {
    background: #e0e7ff;
    color: #3730a3;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    display: inline-block;
    margin: 4px 2px;
}
.plan-item {
    padding: 4px 0;
    color: #333;
    font-size: 14px;
}
.final-card {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
    color: white;
    padding: 28px;
    border-radius: 14px;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ── Session State Init ─────────────────────────────────────────────
for key, default in {
    "session_id": str(uuid.uuid4()),
    "agent": None,
    "agent_state": None,
    "messages": [],
    "patient": None,
    "started": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────────
st.sidebar.title("🤖 Agentic Diagnostic System")
st.sidebar.markdown("*ReAct-style Autonomous Medical Agent*")
st.sidebar.markdown("---")
st.sidebar.subheader("Select Patient")

patients = get_all_patients()
if patients:
    options = {f"{p['name']} (Age {p.get('age','?')})": p for p in patients}
    choice = st.sidebar.selectbox("Patient", list(options.keys()))
    selected_patient = options[choice]
else:
    st.sidebar.warning("No patients found.")
    if st.sidebar.button("➕ Create Test Patient"):
        new_id = str(uuid.uuid4())
        result = supabase.table("patients").insert({
            "id": new_id, "name": "Test Patient", "age": 50, "gender": "Not specified"
        }).execute()
        st.rerun()
    st.stop()

# Reset if patient changed
if st.session_state.patient != selected_patient:
    st.session_state.patient = selected_patient
    st.session_state.agent = None
    st.session_state.agent_state = None
    st.session_state.messages = []
    st.session_state.started = False

if st.sidebar.button("🔄 Reset Session"):
    st.session_state.agent = None
    st.session_state.agent_state = None
    st.session_state.messages = []
    st.session_state.started = False
    st.rerun()

# ── Agent Status Sidebar ───────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Agent Status")
state: AgentState = st.session_state.agent_state

if state:
    st.sidebar.metric("Iteration", f"{state.iteration} / {state.max_iterations}")
    st.sidebar.metric("Tools Called", len(state.observations))
    st.sidebar.metric("Answers Collected", len(state.answers))

    if state.plan:
        st.sidebar.markdown("**Plan:**")
        for i, step in enumerate(state.plan, 1):
            st.sidebar.markdown(f'<div class="plan-item">{i}. {step}</div>', unsafe_allow_html=True)

# ── Main UI ────────────────────────────────────────────────────────
st.title("🤖 Autonomous Medical Diagnostic Agent")
st.caption(f"Patient: **{selected_patient['name']}** | ReAct reasoning loop | Max 10 iterations")
st.markdown("---")

# ── Display Chat History ───────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    msg_type = msg.get("type", "message")

    if role == "agent_thought":
        with st.expander(f"🧠 Agent Reasoning (Iteration {msg.get('iteration', '?')})", expanded=False):
            st.markdown(f'<div class="thought-box">{content}</div>', unsafe_allow_html=True)
            if msg.get("tool"):
                st.markdown(f'<span class="tool-badge">🔧 {msg["tool"]}</span>', unsafe_allow_html=True)
    elif role == "assistant" and msg_type == "final":
        st.markdown(f'<div class="final-card">{content.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(content)
    elif role == "user":
        with st.chat_message("user"):
            st.markdown(content)


def run_agent_step(answer: str = None):
    """Advance the agent by one step and update messages."""
    agent: MedicalDiagnosticAgent = st.session_state.agent
    state: AgentState = st.session_state.agent_state

    result = agent.step(state, patient_answer=answer)

    # Log agent's reasoning
    if state.intermediate_reasoning:
        last = state.intermediate_reasoning[-1]
        st.session_state.messages.append({
            "role": "agent_thought",
            "content": last["thought"],
            "iteration": last["iteration"],
            "tool": last["action"].get("name", "")
        })

    r_type = result.get("type")
    content = result.get("content", "")

    if r_type == "final":
        st.session_state.messages.append({
            "role": "assistant",
            "content": content,
            "type": "final"
        })
    elif r_type == "question":
        st.session_state.messages.append({
            "role": "assistant",
            "content": content,
            "type": "question",
            "options": result.get("options", []),
            "key": result.get("key", "")
        })
    elif r_type == "message" and content:
        st.session_state.messages.append({
            "role": "assistant",
            "content": content,
            "type": "message"
        })


# ── Start Agent ────────────────────────────────────────────────────
if not st.session_state.started:
    if st.button("🚀 Start Diagnostic Session", type="primary", use_container_width=True):
        with st.spinner("Initializing agent and generating plan..."):
            agent, state = create_agent_session(
                selected_patient,
                st.session_state.session_id
            )
            st.session_state.agent = agent
            st.session_state.agent_state = state
            st.session_state.started = True

            # Show plan
            if state.plan:
                plan_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(state.plan)])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Agent Plan Generated:**\n{plan_text}",
                    "type": "message"
                })

            # Run first agent step
            run_agent_step()
        st.rerun()

# ── Active Agent Interaction ───────────────────────────────────────
elif state and not state.is_complete:
    # Find last question in messages
    last_q = None
    for msg in reversed(st.session_state.messages):
        if msg.get("type") == "question":
            last_q = msg
            break

    if last_q and state.waiting_for_patient:
        st.markdown(f'<div class="question-box">❓ {last_q["content"]}</div>', unsafe_allow_html=True)
        options = last_q.get("options", [])

        if options:
            st.markdown("**Select your answer:**")
            cols = st.columns(min(len(options), 4))
            for i, opt in enumerate(options):
                with cols[i % 4]:
                    if st.button(opt, key=f"opt_{state.iteration}_{i}", use_container_width=True):
                        with st.spinner("Agent is reasoning..."):
                            # Record user choice
                            st.session_state.messages.append({
                                "role": "user", "content": opt, "type": "answer"
                            })
                            run_agent_step(answer=opt)
                            # Auto-advance if not waiting for another question
                            while (not st.session_state.agent_state.waiting_for_patient
                                   and not st.session_state.agent_state.is_complete
                                   and st.session_state.agent_state.iteration < st.session_state.agent_state.max_iterations):
                                run_agent_step()
                                if st.session_state.agent_state.waiting_for_patient:
                                    break
                        st.rerun()

        # Free-text fallback
        st.markdown("**Or type a custom response:**")
        with st.form("free_text_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                custom = st.text_input("Type here...", label_visibility="collapsed")
            with col2:
                if st.form_submit_button("Send ➤", use_container_width=True):
                    if custom.strip():
                        with st.spinner("Agent is reasoning..."):
                            st.session_state.messages.append({
                                "role": "user", "content": custom, "type": "answer"
                            })
                            run_agent_step(answer=custom)
                            while (not st.session_state.agent_state.waiting_for_patient
                                   and not st.session_state.agent_state.is_complete
                                   and st.session_state.agent_state.iteration < st.session_state.agent_state.max_iterations):
                                run_agent_step()
                                if st.session_state.agent_state.waiting_for_patient:
                                    break
                        st.rerun()

    elif not state.waiting_for_patient:
        # Agent is thinking, auto-advance
        with st.spinner("🤖 Agent is reasoning autonomously..."):
            run_agent_step()
        st.rerun()

elif state and state.is_complete:
    st.success("✅ Diagnosis complete! All results have been saved to Supabase.")
    if st.button("🔄 Start New Session", use_container_width=True):
        st.session_state.agent = None
        st.session_state.agent_state = None
        st.session_state.messages = []
        st.session_state.started = False
        st.rerun()
