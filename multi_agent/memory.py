"""
multi_agent/memory.py — Hybrid Memory System.
Manages short-term (conversation), long-term (RAG/Supabase), and episodic (past sessions).
"""
from typing import Any
from rag_engine import generate_embedding
from database import save_conversation_memory, supabase


class HybridMemory:
    """Unified interface for all 3 memory layers."""

    def __init__(self, patient_id: str, session_id: str):
        self.patient_id = patient_id
        self.session_id = session_id
        self._embedding_cache: dict[str, list] = {}

    # ── Short-term Memory ──────────────────────────────────────────
    def save_short_term(self, role: str, content: str) -> None:
        """Persist a chat message to Supabase conversation_memory."""
        try:
            emb = self._get_embedding(content)
            save_conversation_memory(
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
        """Vector similarity search in Supabase conversation_memory."""
        emb = self._get_embedding(query)
        if not emb:
            return []
        try:
            res = supabase.rpc("match_memory", {
                "query_embedding": emb,
                "match_threshold": 0.45,
                "match_count": limit,
                "p_patient_id": self.patient_id
            }).execute()
            return res.data or []
        except Exception as e:
            print(f"[Memory] RAG search failed: {e}")
            return []

    # ── Episodic Memory ────────────────────────────────────────────
    def load_episodic(self) -> list[dict]:
        """Load past diagnostic decisions for this patient."""
        try:
            res = supabase.table("final_diagnostic_decisions") \
                .select("*") \
                .eq("patient_id", self.patient_id) \
                .order("created_at", desc=True) \
                .limit(5) \
                .execute()
            return res.data or []
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
