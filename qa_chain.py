"""
RAG QA: retrieve relevant chunks, then call Ollama with a strict context-only prompt.

Answers must be grounded in the retrieved text; if context is insufficient,
the model is instructed to say so.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.embeddings import Embeddings

from vector_store import build_faiss_index, similarity_search_with_scores


SYSTEM_PROMPT = """You are a study assistant. Answer ONLY using the CONTEXT below.
- If the answer is not contained in the context, say you cannot find it in the textbook and suggest what topic the user might check.
- Do not use outside knowledge, guesses, or general facts not stated in the context.
- Be concise and clear. If quoting, keep quotes short."""


def format_context(docs: List[Document]) -> str:
    """Turn retrieved chunks into a single string with page labels."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "?")
        src = doc.metadata.get("source", "PDF")
        parts.append(f"[Excerpt {i} | page {page} | source: {src}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def create_ollama_llm(model: str = "mistral", base_url: str = "http://127.0.0.1:11434") -> ChatOllama:
    """Local chat model via Ollama (no API key)."""
    return ChatOllama(model=model, base_url=base_url, temperature=0.1)


def build_vector_store_from_chunks(chunks: List[Document], embeddings: Embeddings):
    """Thin wrapper so qa_chain can own the index if you prefer one-call setup."""
    return build_faiss_index(chunks, embeddings)


def answer_question(
    question: str,
    store,
    llm: ChatOllama,
    k: int = 4,
) -> Dict[str, Any]:
    """
    Retrieve top-k chunks, run the local LLM, return answer + source documents.

    Returns dict keys: answer (str), sources (list of dict with page, preview).
    """
    pairs: List[Tuple[Document, float]] = similarity_search_with_scores(store, question, k=k)
    docs = [d for d, _ in pairs]

    context = format_context(docs)
    user_content = f"""CONTEXT:
{context}

QUESTION:
{question}

Answer using only the context above."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    answer_text = response.content if hasattr(response, "content") else str(response)

    sources = []
    for doc, score in pairs:
        preview = doc.page_content.strip().replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:217] + "..."
        sources.append(
            {
                "page": doc.metadata.get("page"),
                "score": float(score),
                "preview": preview,
            }
        )

    return {"answer": answer_text.strip(), "sources": sources}
