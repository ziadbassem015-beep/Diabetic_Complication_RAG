ARCHITECTURE_RULES.md
# SYSTEM ARCHITECTURE RULES

This is a production-style medical multi-agent diagnostic system.

STRICT RULES:
- Preserve all existing functionality.
- Do NOT break old flows.
- All changes must be additive only.
- Maintain clean architecture.
- Keep repository pattern.
- No direct DB access outside repositories.
- Keep LangGraph-style orchestration.
- Preserve existing agents and evaluations.
- Maintain backward compatibility.
- Avoid rewriting existing modules unless necessary.
- Reuse existing abstractions whenever possible.
- Add new features using extension, not replacement.

Current system contains:
- Multi-agent orchestration
- RAG memory
- Streamlit UI
- Supabase pgvector
- ML inference pipeline
- Fusion decision logic
- Evaluation framework

New features must integrate cleanly into:
- questionnaire pipeline
- repositories
- graph routing
- report generation
- evaluation system