"""
Utility helpers: text splitting and lightweight validation.

Keeps chunking policy in one place so you can tune overlap and size
without touching PDF or vector-store code.
"""

from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Default chunk size tuned for textbooks; smaller chunks = finer retrieval but more vectors.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def get_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Create a splitter that breaks text on headings, paragraphs, then sentences."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_documents(documents: List[Document], **splitter_kwargs) -> List[Document]:
    """Split LangChain Documents into smaller chunks; preserves metadata (e.g. page)."""
    splitter = get_text_splitter(**splitter_kwargs)
    return splitter.split_documents(documents)


def validate_pdf_path(path: str) -> str:
    """Ensure the PDF path exists and is a file; return absolute path."""
    if not path or not str(path).strip():
        raise ValueError("PDF path is empty.")
    path = os.path.abspath(os.path.expanduser(path.strip()))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    if not path.lower().endswith(".pdf"):
        raise ValueError("File must be a .pdf")
    return path


def validate_question(text: str) -> str:
    """Strip and reject empty questions."""
    if text is None:
        raise ValueError("Question cannot be empty.")
    q = text.strip()
    if not q:
        raise ValueError("Question cannot be empty.")
    return q
