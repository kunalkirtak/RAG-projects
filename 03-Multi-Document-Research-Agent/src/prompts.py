"""
prompts.py
----------
Prompt templates for the RAG pipeline: the main answer-generation prompt,
and a smaller prompt used for conversational query rewriting (resolving
pronouns / follow-up references against chat history).
"""

from __future__ import annotations

from typing import Any, Dict, List

SYSTEM_PROMPT = """You are a meticulous research assistant. Answer the user's \
question using ONLY the information in the provided context excerpts. \
Cite sources inline using the bracketed numbers, e.g. [1], [2]. \
If the context does not contain enough information to answer confidently, \
say so plainly instead of guessing. Keep answers concise and well-structured."""


QUERY_REWRITE_SYSTEM_PROMPT = """You rewrite follow-up questions into fully \
self-contained search queries. Given the recent conversation and a new user \
message, produce ONE rewritten query that resolves pronouns and implicit \
references (e.g. "it", "that", "the second one") using the conversation \
context. If the message is already self-contained, return it unchanged. \
Respond with ONLY the rewritten query, no explanation."""


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Render retrieved chunks as a numbered context block for the prompt."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        filename = chunk.get("metadata", {}).get("filename", "unknown source")
        lines.append(f"[{i}] (source: {filename})\n{chunk['text']}")
    return "\n\n".join(lines)


def build_rag_prompt(query: str, chunks: List[Dict[str, Any]], history: str = "") -> str:
    """Assemble the full RAG generation prompt from context chunks and history.

    Args:
        query: The (possibly rewritten) user question.
        chunks: Retrieved context chunk records.
        history: Optional formatted conversation history.

    Returns:
        A single prompt string, ready to send to the Gemini API.
    """
    context_block = build_context_block(chunks)
    history_block = f"\nConversation so far:\n{history}\n" if history else ""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_block}\n"
        f"{history_block}\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


def build_query_rewrite_prompt(query: str, history: str) -> str:
    """Assemble the prompt used to rewrite a follow-up query into a standalone one."""
    return (
        f"{QUERY_REWRITE_SYSTEM_PROMPT}\n\n"
        f"Conversation:\n{history or '(no prior turns)'}\n\n"
        f"New message: {query}\n\n"
        f"Rewritten query:"
    )
