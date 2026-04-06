"""
FAISS vector store: embed document chunks and retrieve by similarity.

FAISS is in-memory (fast for a single book). For huge corpora, persist with
store.save_local / load_local.
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def build_faiss_index(documents: List[Document], embeddings: Embeddings) -> FAISS:
    """Embed all chunks and build an in-memory FAISS index."""
    if not documents:
        raise ValueError("No documents to index.")
    return FAISS.from_documents(documents, embeddings)


def similarity_search_with_scores(
    store: FAISS,
    query: str,
    k: int = 4,
) -> List[Tuple[Document, float]]:
    """
    Return top-k chunks with distance scores (lower is more similar for L2;
    with normalized embeddings, interpret relatively).
    """
    if not query.strip():
        raise ValueError("Search query cannot be empty.")
    return store.similarity_search_with_score(query, k=k)


def get_retriever(store: FAISS, k: int = 4):
    """LangChain retriever for optional use with higher-level chains."""
    return store.as_retriever(search_kwargs={"k": k})
