"""
Entry point: load PDF → chunk → embed → FAISS → question loop (type 'exit' to quit).

Requires Ollama running locally with a pulled model (e.g. mistral).
"""

from __future__ import annotations

import argparse
import sys

from langchain_core.messages import HumanMessage

from embeddings import get_local_embeddings
from pdf_loader import load_pdf_as_documents
from qa_chain import answer_question, build_vector_store_from_chunks, create_ollama_llm
from utils import split_documents, validate_question


def parse_args():
    p = argparse.ArgumentParser(description="PDF RAG study assistant (local Ollama + FAISS).")
    p.add_argument("pdf_path", nargs="?", help="Path to the textbook PDF")
    p.add_argument(
        "--model",
        default="mistral",
        help='Ollama model name (default: mistral). Try "llama3" if installed.',
    )
    p.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Ollama API base URL",
    )
    p.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    return p.parse_args()


def print_sources(sources: list) -> None:
    """Pretty-print which PDF excerpts were used."""
    print("\n--- Sources (retrieved chunks) ---")
    for i, s in enumerate(sources, start=1):
        page = s.get("page", "?")
        score = s.get("score", 0.0)
        preview = s.get("preview", "")
        print(f"  [{i}] Page {page} | distance {score:.4f}")
        print(f"      {preview}")
    print("--- End sources ---\n")


def main() -> int:
    args = parse_args()
    pdf_path = args.pdf_path
    if not pdf_path:
        pdf_path = input("Enter path to your PDF: ").strip()
    if not pdf_path:
        print("Error: No PDF path provided.", file=sys.stderr)
        return 1

    try:
        print("Loading PDF...")
        docs = load_pdf_as_documents(pdf_path)
        print(f"Loaded {len(docs)} page(s) with text.")

        print("Splitting into chunks...")
        chunks = split_documents(docs)
        print(f"Created {len(chunks)} chunk(s).")

        print("Loading embedding model (first run may download weights)...")
        embeddings = get_local_embeddings()

        print("Building FAISS index...")
        store = build_vector_store_from_chunks(chunks, embeddings)

        print(f"Connecting to Ollama model '{args.model}' at {args.ollama_url}...")
        llm = create_ollama_llm(model=args.model, base_url=args.ollama_url)
        # Probes the server; raises if Ollama is down or model is missing.
        _ = llm.invoke([HumanMessage(content="Reply with exactly: ok")])

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Setup failed: {e}", file=sys.stderr)
        print(
            "Hint: Start Ollama, run: ollama pull " + args.model,
            file=sys.stderr,
        )
        return 1

    print("\nReady. Ask questions about the PDF. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            q = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not q:
            print("Please enter a question or type 'exit'.")
            continue
        lowered = q.lower()
        if lowered in ("exit", "quit", "q"):
            print("Bye.")
            return 0

        try:
            q = validate_question(q)
        except ValueError as e:
            print(f"Error: {e}")
            continue

        try:
            result = answer_question(q, store, llm, k=args.top_k)
        except Exception as e:
            print(f"Query failed: {e}")
            continue

        print("\nAnswer:")
        print(result["answer"])
        print_sources(result["sources"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
