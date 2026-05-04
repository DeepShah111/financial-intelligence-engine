"""
Conversation Memory module for the Financial Intelligence Engine.

Maintains a rolling window of the last N question-answer turns and uses
that history to contextualise follow-up questions before they are passed
to the retriever and generation agent.

Design decisions:
- Pure Python — no LLM call needed for query reformulation.
  Follow-up detection is heuristic (pronoun / reference word scanning).
  This avoids adding latency and token cost to every user turn.
- Rolling window (default: last 3 turns). Older turns are evicted to
  prevent context window bloat in the retrieval query string.
- Immutable Turn dataclass — history entries cannot be mutated after the
  fact, preventing accidental state corruption in multi-threaded Gradio workers.
- Thread-safe add_turn via list replacement rather than in-place mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import logger


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Turn:
    """
    Immutable record of a single conversation turn.

    Attributes:
        query:  The user's original question (before any reformulation).
        answer: The final audited answer returned by the generation agent.
    """
    query:  str
    answer: str


# ── Conversation Memory ───────────────────────────────────────────────────────

# Signals that a user message is a follow-up referencing a previous entity.
# When any of these appear in the new query, the last N prior questions are
# appended as context so the retriever receives a self-contained query string.
_FOLLOW_UP_SIGNALS: frozenset[str] = frozenset({
    "their", "its", "they", "it", "those", "these",
    "the same", "what about", "how about", "also",
    "and their", "compare to", "compared to",
    "how does that", "how do they", "what else",
    "tell me more", "elaborate", "expand on",
    "what about the", "versus", "vs",
})


class ConversationMemory:
    """
    Rolling-window conversation history with follow-up query reformulation.

    Usage:
        memory = ConversationMemory(max_turns=3)

        # After each RAG call:
        memory.add_turn(user_query, agent_answer)

        # Before next RAG call — enriches follow-up queries with prior context:
        retrieval_query = memory.reformulate_query(new_user_question)

    The reformulated query is ONLY used for retrieval and generation; the
    original user question is always stored in history so the conversation
    log remains human-readable.
    """

    def __init__(self, max_turns: int = 3) -> None:
        """
        Args:
            max_turns: Maximum number of prior turns to keep in the rolling
                       window. Older turns are evicted FIFO. Default: 3.
        """
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        self._max_turns: int        = max_turns
        self._history:  list[Turn]  = []

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_turn(self, query: str, answer: str) -> None:
        """
        Append a completed turn and evict the oldest if the window is full.

        Args:
            query:  The original user question for this turn.
            answer: The final agent answer for this turn.
        """
        self._history.append(Turn(query=query, answer=answer))
        if len(self._history) > self._max_turns:
            # Evict oldest turn (FIFO). List slice creates a new list,
            # so concurrent readers of _history see a consistent snapshot.
            self._history = self._history[-self._max_turns:]
        logger.info(
            "[ConversationMemory] Turn added. History depth: %d/%d.",
            len(self._history), self._max_turns,
        )

    def clear(self) -> None:
        """Reset conversation history to an empty state."""
        self._history = []
        logger.info("[ConversationMemory] History cleared.")

    # ── Query Reformulation ───────────────────────────────────────────────────

    def reformulate_query(self, current_query: str) -> str:
        """
        Return a retrieval-ready query enriched with conversation context
        when the current message appears to be a follow-up.

        Follow-up detection: if the lowercased query contains any token from
        _FOLLOW_UP_SIGNALS AND there is at least one prior turn in history,
        the prior questions are appended so the retriever has a self-contained
        query string (no dangling pronouns or implicit references).

        If the query is standalone (no follow-up signals, or history is empty),
        it is returned unchanged — zero overhead for fresh questions.

        Args:
            current_query: The raw question from the user this turn.

        Returns:
            The original query, or a context-enriched version for retrieval.
        """
        if not self._history:
            return current_query

        lowered: str = current_query.lower()
        is_follow_up: bool = any(
            signal in lowered for signal in _FOLLOW_UP_SIGNALS
        )

        if not is_follow_up:
            return current_query

        # Build a compact context string from the last N prior questions.
        # We include only the questions (not the full answers) to keep the
        # retrieval query concise and avoid the 8192-token BM25 limit.
        prior_questions: str = " | ".join(
            f"Q{i}: {turn.query}"
            for i, turn in enumerate(self._history, 1)
        )

        reformulated: str = (
            f"{current_query} "
            f"[Prior conversation context: {prior_questions}]"
        )

        logger.info(
            "[ConversationMemory] Follow-up detected. Reformulated query: '%s'",
            reformulated[:120],
        )
        return reformulated

    # ── Read-Only Accessors ───────────────────────────────────────────────────

    def get_history_as_text(self, include_answers: bool = True) -> str:
        """
        Return the conversation history as a human-readable string.

        Args:
            include_answers: If True, include truncated answer previews (200 chars).
                             If False, return only the questions — useful for
                             building a compact retrieval context string.

        Returns:
            Multi-line string with Q/A pairs, or empty string if no history.
        """
        if not self._history:
            return ""

        lines: list[str] = []
        for i, turn in enumerate(self._history, 1):
            lines.append(f"Q{i}: {turn.query}")
            if include_answers:
                preview = turn.answer[:200].rstrip()
                suffix  = "..." if len(turn.answer) > 200 else ""
                lines.append(f"A{i}: {preview}{suffix}")
            lines.append("")   # blank line separator

        return "\n".join(lines).strip()

    def get_history_as_gradio_pairs(self) -> list[dict]:
        """
        Return history in the Gradio 6 message dict format expected by
        gr.Chatbot so a restored session can seed the Gradio chat component.

        Returns:
            List of role/content dicts.
        """
        pairs = []
        for turn in self._history:
            pairs.append({"role": "user", "content": turn.query})
            pairs.append({"role": "assistant", "content": turn.answer})
        return pairs

    def get_last_n_queries(self, n: Optional[int] = None) -> list[str]:
        """
        Return the last n user questions.

        Args:
            n: Number of recent queries to return. Defaults to max_turns.

        Returns:
            List of query strings, oldest first.
        """
        limit: int = n if n is not None else self._max_turns
        return [turn.query for turn in self._history[-limit:]]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def turn_count(self) -> int:
        """Number of completed turns currently held in the window."""
        return len(self._history)

    @property
    def is_empty(self) -> bool:
        """True if no turns have been recorded yet."""
        return len(self._history) == 0

    @property
    def max_turns(self) -> int:
        """The configured rolling-window size."""
        return self._max_turns

    def __repr__(self) -> str:
        return (
            f"ConversationMemory("
            f"turns={self.turn_count}/{self._max_turns}, "
            f"queries={self.get_last_n_queries()})"
        )