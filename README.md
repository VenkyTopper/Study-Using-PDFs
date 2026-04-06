# PDF RAG Study Assistant (100% Local)

Ask questions about a textbook PDF. Answers are grounded in retrieved chunks only, using **Ollama** (local LLM), **FAISS** (vector search), and **Hugging Face** embeddings (local, no API key).

## Prerequisites

1. **Python 3.10+**
2. **Ollama** — [Install Ollama](https://ollama.com/download) for Windows, then start it (it runs a local server, usually `http://127.0.0.1:11434`).

### Pull a chat model (pick one)

```bash
ollama pull mistral
```

Or:

```bash
ollama pull llama3
```

Smoke-test:

```bash
ollama run mistral
```

Type a message; if it replies, the daemon and model are working. Exit with `Ctrl+D` or `/bye` depending on your terminal.

## Python dependencies

From this folder:

```bash
python -m pip install -r requirements.txt
```

The first run will download the embedding model `sentence-transformers/all-MiniLM-L6-v2` (free, public on Hugging Face Hub).

## Run

```bash
python main.py path\to\your\textbook.pdf
```

Or run without arguments and paste the path when prompted.

- Ask multiple questions in a loop.
- Type `exit`, `quit`, or `q` to stop.
- Each answer lists **sources** (page number + snippet preview).

### Options

```bash
python main.py book.pdf --model llama3 --top-k 6
```

- `--model` — Ollama model name (default `mistral`).
- `--ollama-url` — if Ollama listens elsewhere.
- `--top-k` — how many chunks to retrieve.

## Project layout

| File | Role |
|------|------|
| `main.py` | CLI entry: load PDF, index, question loop |
| `pdf_loader.py` | PDF → text + page metadata (`pypdf`) |
| `utils.py` | Chunking + path/question validation |
| `embeddings.py` | Local Hugging Face embeddings |
| `vector_store.py` | Build/query FAISS |
| `qa_chain.py` | Retrieve → prompt → Ollama → answer + sources |

## Notes

- **Scanned PDFs** without OCR may yield no text; use a text-based PDF or run OCR first.
- For **GPU** embeddings, edit `embeddings.py` and set `device` to `"cuda"` if you have a suitable GPU.
