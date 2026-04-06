"""
Local embedding model via Hugging Face (downloads once, runs on CPU/GPU).

Model: sentence-transformers/all-MiniLM-L6-v2 — small, fast, no API key.
"""

from __future__ import annotations

from langchain_community.embeddings import HuggingFaceEmbeddings

# Keep name explicit so it matches the project spec.
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def get_local_embeddings(model_name: str = EMBEDDING_MODEL_ID) -> HuggingFaceEmbeddings:
    """
    Build a HuggingFaceEmbeddings instance (runs locally).

    First run downloads weights from Hugging Face Hub (free, no key required
    for this public model).
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # set to "cuda" if you have GPU + CUDA
        encode_kwargs={"normalize_embeddings": True},
    )
