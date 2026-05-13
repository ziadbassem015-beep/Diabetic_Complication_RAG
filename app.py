"""
app.py — Streamlit UI for the Multi-Agent Diagnostic System.
Drives the DiagnosticGraph and streams intermediate agent events to the UI.
"""
import streamlit as st
import uuid
from database import get_all_patients, supabase
from multi_agent import DiagnosticGraph, MultiAgentState

st.set_page_config(
    page_title="Multi-Agent Medical AI",
    page_icon="🧠",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.agent-badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 600; margin: 2px;
}
.badge-planner     { background:#dbeafe; color:#1e40af; }
.badge-memory      { background:#f3e8ff; color:#7c3aed; }
.badge-reasoning   { background:#dcfce7; color:#166534; }
.badge-ml          { background:#fef9c3; color:#854d0e; }
.badge-fusion      { background:#ffedd5; color:#9a3412; }
.badge-reflection  { background:#fce7f3; color:#9d174d; }
.badge-report      { background:#cffafe; color:#0e7490; }
.badge-tool        { background:#e0e7ff; color:#3730a3; }

.thought-box {
    background: #fffbeb; border-left: 4px solid #f59e0b;
    padding: 10px 14px; border-radius: 8px;
    font-size: 13px; color: #444; font-style: italic;
    margin: 4px 0;
}
.question-box {
    background: #f0f4ff; border-left: 5px solid #1e3a5f;
    padding: 18px 20px; border-radius: 10px;
    font-size: 18px; font-weight: 600; color: #000 !important;
    margin-bottom: 16px;
}
.final-card {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
    color: white; padding: 28px; border-radius: 14px; line-height: 1.8;
}
.plan-step { padding: 3px 0; font-size: 14px; color: #333; }
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; }
.metric-box {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 10px 16px; text-align: center; min-width: 90px;
}
.metric-label { font-size: 11px; color: #888; }
.metric-value { font-size: 22px; font-weight: 700; color: #1e3a5f; }
.node-flow {
    font-size: 12px; color: #666; letter-spacing: 1px; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Badge Helper ───────────────────────────────────────────────────
AGENT_BADGE_MAP = {
    "PlannerAgent":          ("planner",   "🗂 Planner"),
    "MemoryRAGAgent":        ("memory",    "🔍 Memory"),
    "ClinicalReasoningAgent":("reasoning", "🧠 Reasoning"),
    "MLInferenceAgent":      ("ml",        "🧪 ML"),
    "FusionDecisionAgent":   ("fusion",    "📊 Fusion"),
    "ReflectionAgent":       ("reflection","🧭 Reflection"),
    "ReportGeneratorAgent":  ("report",    "🧾 Report"),
    "ToolNode":              ("tool",      "🔧 Tool"),
    "graph":                 ("tool",      "⚙ Graph"),
    "system":                ("tool",      "⚙ System"),
}

def agent_badge(agent: str) -> str:
    cls, label = AGENT_BADGE_MAP.get(agent, ("tool", agent))
    return f'<span class="agent-badge badge-{cls}">{label}</span>'

# ── Session State ──────────────────────────────────────────────────
for k, v in {
    "graph": None,
    "session_id": str(uuid.uuid4()),
    "patient": None,
    "ui_events": [],    # rendered events
    "started": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Multi-Agent Medical AI")
    st.caption("Autonomous · Self-Reflecting · RAG-Powered")
    st.markdown("---")

    patients = get_all_patients()
    if not patients:
        st.warning("No patients found.")
        if st.button("➕ Create Test Patient"):
            pid = str(uuid.uuid4())
            supabase.table("patients").insert(
                {"id": pid, "name": "Test Patient", "age": 50, "gender": "Not specified"}
            ).execute()
            st.rerun()
        st.stop()

    patient_map = {f"{p['name']} (Age {p.get('age','?')})": p for p in patients}
    sel = st.selectbox("Patient", list(patient_map.keys()))
    selected_patient = patient_map[sel]

    if st.session_state.patient != selected_patient:
        st.session_state.patient = selected_patient
        st.session_state.graph = None
        st.session_state.ui_events = []
        st.session_state.started = False

    if st.button("🔄 Reset Session"):
        st.session_state.graph = None
        st.session_state.ui_events = []
        st.session_state.started = False
        st.rerun()

    st.markdown("---")
    st.subheader("Agent Status")

    g: DiagnosticGraph = st.session_state.graph
    if g:
        state = g.state
        st.markdown(f'<div class="node-flow">Current: <b>{state.current_node}</b></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Iteration", f"{state.iteration}/{state.max_iterations}")
        col2.metric("Confidence", f"{g.confidence:.0%}")
        col1.metric("Answers", len(state.answers))
        col2.metric("Tools Called", len([e for e in state.audit_log if "Executing" in e.action]))

        if state.plan:
            st.markdown("**Diagnostic Plan:**")
            for i, step in enumerate(state.plan, 1):
                icon = "✅" if i <= state.reflection_count else "◻"
                st.markdown(f'<div class="plan-step">{icon} {i}. {step}</div>', unsafe_allow_html=True)

    st.markdown("---")
    # Agent graph visualization
    st.subheader("Agent Graph")
    st.markdown("""
```
[Planner] → [Memory]
    ↓
[Reasoning] ←─────────┐
    ↓                  │
 [Tool] → [Wait]       │
    ↓                  │
[Reflection] ──────────┘
    ↓
  [ML] → [Fusion]
    ↓
 [Report] → END
```
""")

# ── Main UI ────────────────────────────────────────────────────────
st.title("🧠 Autonomous Multi-Agent Medical Diagnostic System")
st.caption(
    f"Patient: **{selected_patient['name']}** | "
    f"ReAct + Self-Reflection + RAG | 8 Specialized Agents"
)
st.markdown("---")

# ── Render UI Events ───────────────────────────────────────────────
def render_event(ev: dict):
    ev_type = ev.get("type", "")
    agent = ev.get("agent", "system")
    content = ev.get("content", ev.get("details", ""))
    iteration = ev.get("iteration", "?")

    badge = agent_badge(agent)

    if ev_type == "agent_start":
        st.markdown(f'{badge} <small>iter {iteration}</small> — {content}', unsafe_allow_html=True)

    elif ev_type == "plan":
        with st.expander("📋 Diagnostic Plan Generated", expanded=True):
            st.markdown(f'{badge}', unsafe_allow_html=True)
            for line in content.split("\n"):
                st.markdown(f"- {line}")

    elif ev_type == "thought":
        with st.expander(f"🧠 Reasoning — iteration {iteration}", expanded=False):
            st.markdown(f'<div class="thought-box">{content}</div>', unsafe_allow_html=True)

    elif ev_type == "tool_call":
        st.markdown(f'{badge} {content}', unsafe_allow_html=True)

    elif ev_type == "observation":
        st.info(f"📡 Observation: {content}")

    elif ev_type == "memory":
        st.markdown(f'{badge} {content}', unsafe_allow_html=True)

    elif ev_type == "ml_result":
        st.success(f"🧪 {content}")

    elif ev_type == "fusion":
        st.success(f"📊 {content}")

    elif ev_type == "reflection":
        st.markdown(f'{badge} {content}', unsafe_allow_html=True)

    elif ev_type == "replan":
        st.warning(f"🔄 Re-planning: {content}")

    elif ev_type == "warning":
        st.warning(content)

    elif ev_type == "final_report":
        pass  # rendered separately below

    elif ev_type == "audit":
        pass  # shown in audit tab


for ev in st.session_state.ui_events:
    render_event(ev)


# ── Current Question (if waiting) ─────────────────────────────────
g: DiagnosticGraph = st.session_state.graph

if g and g.is_waiting:
    pq = g.pending_question
    question = pq.get("question", "")
    options = pq.get("options", [])
    key = pq.get("key", "")

    st.markdown("---")
    st.markdown(f'<div class="question-box">❓ {question}</div>', unsafe_allow_html=True)
    st.markdown("**Select your answer:**")

    cols = st.columns(min(len(options), 4))
    for i, opt in enumerate(options):
        with cols[i % 4]:
            if st.button(opt, key=f"ans_{key}_{i}", use_container_width=True):
                with st.spinner("🤖 Agents reasoning..."):
                    new_events = g.submit_patient_answer(key, opt)
                    st.session_state.ui_events.extend(new_events)
                st.rerun()

    st.markdown("**Or type a custom response:**")
    with st.form("custom_ans", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            custom = st.text_input("Your answer", label_visibility="collapsed")
        with c2:
            if st.form_submit_button("Send ➤", use_container_width=True) and custom.strip():
                with st.spinner("🤖 Agents reasoning..."):
                    new_events = g.submit_patient_answer(key, custom.strip())
                    st.session_state.ui_events.extend(new_events)
                st.rerun()


# ── Final Report ───────────────────────────────────────────────────
if g and g.is_complete and g.final_report:
    st.markdown("---")
    st.markdown(f'<div class="final-card">{g.final_report.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True)

    # Audit log tab
    with st.expander("🔍 Full Agent Audit Log", expanded=False):
        audit = g.get_audit_log()
        for entry in audit:
            st.markdown(
                f"**[{entry['iteration']}]** `{entry['agent']}` → {entry['action']} — {entry['details']}"
            )

    if st.button("🔄 Start New Session", use_container_width=True):
        st.session_state.graph = None
        st.session_state.ui_events = []
        st.session_state.started = False
        st.rerun()


# ── Auto-advance (non-waiting, non-complete) ───────────────────────
elif g and not g.is_waiting and not g.is_complete and st.session_state.started:
    with st.spinner("🤖 Agents working autonomously..."):
        new_events = g.run_until_pause()
        st.session_state.ui_events.extend(new_events)
    st.rerun()


# ── Start Button ───────────────────────────────────────────────────
elif not st.session_state.started:
    st.markdown("### Ready to begin the diagnostic session?")
    st.markdown("The agent system will autonomously plan, reason, collect data, and generate a medical report.")

    if st.button("🚀 Launch Multi-Agent Diagnostic Session", type="primary", use_container_width=True):
        with st.spinner("🧠 Initializing agents and planning session..."):
            g = DiagnosticGraph(selected_patient, st.session_state.session_id)
            st.session_state.graph = g
            st.session_state.started = True
            init_events = g.initialize()
            st.session_state.ui_events.extend(init_events)
        st.rerun()
