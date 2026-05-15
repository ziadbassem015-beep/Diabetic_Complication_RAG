"""
multi_agent/memory.py — Hybrid Memory System.
Manages short-term (conversation), long-term (RAG/Supabase), and episodic (past sessions).
"""
from typing import Any
from core.rag_engine import generate_embedding
from core.services.diagnostic_service import DiagnosticService


class HybridMemory:
    """Unified interface for all 3 memory layers."""

    def __init__(self, patient_id: str, session_id: str):
        self.patient_id = patient_id
        self.session_id = session_id
        self._embedding_cache: dict[str, list] = {}

    # ── Short-term Memory ──────────────────────────────────────────
    def save_short_term(self, role: str, content: str) -> None:
        """Persist a chat message to the memory repository."""
        try:
            emb = self._get_embedding(content)
            DiagnosticService.store_memory(
                self.patient_id,
                self.session_id,
                role,
                content,
                emb if emb else None
            )
        except Exception as e:
            print(f"[Memory] Failed to save short-term: {e}")

    # ── Long-term RAG Memory ───────────────────────────────────────
    def search_long_term(self, query: str, limit: int = 5) -> list[dict]:
        """Vector similarity search through the service layer."""
        try:
            return DiagnosticService.retrieve_memory(self.patient_id, query, limit)
        except Exception as e:
            print(f"[Memory] RAG search failed: {e}")
            return []

    # ── Episodic Memory ────────────────────────────────────────────
    def load_episodic(self) -> list[dict]:
        """Load past diagnostic decisions for this patient."""
        try:
            return DiagnosticService.get_recent_decisions(self.patient_id)
        except Exception as e:
            print(f"[Memory] Episodic load failed: {e}")
            return []

    # ── Embedding Cache ────────────────────────────────────────────
    def _get_embedding(self, text: str) -> list:
        """Cache embeddings to avoid duplicate API calls."""
        key = text[:100]
        if key not in self._embedding_cache:
            self._embedding_cache[key] = generate_embedding(text)
        return self._embedding_cache[key]

    def format_long_term_context(self, results: list[dict]) -> str:
        """Format RAG results into readable context string."""
        if not results:
            return "No relevant history found."
        lines = []
        for r in results:
            role = r.get("role", "?")
            content = r.get("content", "")[:150]
            lines.append(f"[{role.upper()}]: {content}")
        return "\n".join(lines)
