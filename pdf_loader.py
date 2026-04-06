"""
Load textbook PDFs and return LangChain Documents with page metadata.

Uses pypdf only (no cloud). Each page becomes at least one document so
sources can cite page numbers after chunking.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader

from utils import validate_pdf_path


def load_pdf_as_documents(pdf_path: str) -> List[Document]:
    """
    Read all pages from a PDF and return one Document per page.

    Page text may be empty on scanned PDFs without OCR; those pages are skipped
    with a warning-style empty skip (caller can check len).
    """
    path = validate_pdf_path(pdf_path)
    reader = PdfReader(path)
    if len(reader.pages) == 0:
        raise ValueError("PDF has no pages.")

    documents: List[Document] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from page {i + 1}: {e}") from e

        text = text.strip()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={"source": path, "page": i + 1},
            )
        )

    if not documents:
        raise ValueError(
            "No text extracted from PDF. It may be image-only; try a text-based PDF or OCR."
        )
    return documents
