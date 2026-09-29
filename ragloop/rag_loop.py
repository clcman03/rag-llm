#!/usr/bin/env python3
"""
rag_loop.py

Runs an interactive loop against a locally-hosted LLM (via Ollama), using
retrieval-augmented generation over a folder of .txt/.md files. After each
answer, the same LLM is asked to grade its own response on a 0-100 scale.

Requirements:
    - Ollama installed and running locally (https://ollama.com), with a model
      pulled, e.g.:  `ollama pull llama3`
    - pip install requests sentence-transformers numpy

Usage:
    python rag_loop.py --docs-dir ./knowledge_base --model llama3

Note on the self-evaluation score: this is the model grading its own answer,
not an independent ground-truth check. Treat it as a rough self-consistency
signal, not a guarantee of correctness.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests


# --------------------------------------------------------------------------
# Document loading and chunking
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def load_documents(docs_dir: Path) -> List[Tuple[str, str]]:
    """Returns list of (filename, raw_text) for every .txt/.md file in docs_dir."""
    paths = sorted(list(docs_dir.rglob("*.txt")) + list(docs_dir.rglob("*.md")))
    if not paths:
        raise FileNotFoundError(f"No .txt or .md files found under {docs_dir}")

    docs = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[warn] skipping {path}: {e}", file=sys.stderr)
            continue
        if text.strip():
            docs.append((str(path.relative_to(docs_dir)), text))
    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Splits on paragraph boundaries, then packs paragraphs into chunks of
    roughly `chunk_size` characters, with `overlap` characters repeated
    between consecutive chunks to preserve context across chunk boundaries.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 1 > chunk_size:
            chunks.append(current.strip())
            # carry the tail of the previous chunk forward for overlap
            current = current[-overlap:] + "\n" + para
        else:
            current = (current + "\n" + para) if current else para
    if current.strip():
        chunks.append(current.strip())

    return chunks


def build_chunks(docs_dir: Path, chunk_size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for source, text in load_documents(docs_dir):
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(Chunk(text=piece, source=source, chunk_id=i))
    if not chunks:
        raise ValueError("No chunks produced from documents — check file contents.")
    return chunks


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

class Retriever:
    def __init__(self, chunks: List[Chunk], embed_model_name: str):
        from sentence_transformers import SentenceTransformer  # local import: slow to load

        self.chunks = chunks
        print(f"[setup] loading embedding model '{embed_model_name}' ...")
        self.model = SentenceTransformer(embed_model_name)

        texts = [c.text for c in chunks]
        print(f"[setup] embedding {len(texts)} chunks ...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # normalize so dot product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        self.embeddings = embeddings / norms

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        q_emb = self.model.encode([query], convert_to_numpy=True)[0]
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            q_norm = 1e-8
        q_emb = q_emb / q_norm

        scores = self.embeddings @ q_emb
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]


# --------------------------------------------------------------------------
# Local LLM client (Ollama)
# --------------------------------------------------------------------------

class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434", temperature: float = 0.2):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            },  
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def build_rag_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    context_blocks = []
    for chunk, score in retrieved:
        context_blocks.append(
            f"[Source: {chunk.source} | chunk {chunk.chunk_id} | similarity {score:.3f}]\n{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    return (
        "You are a careful assistant answering questions using ONLY the provided context. "
        "If the context does not contain enough information to answer, say so explicitly "
        "rather than guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def build_eval_prompt(query: str, retrieved: List[Tuple[Chunk, float]], answer: str) -> str:
    context = "\n\n---\n\n".join(chunk.text for chunk, _ in retrieved)
    return (
        "You are grading an AI-generated answer for factual correctness and grounding "
        "in the provided context, on a scale from 0 to 100 (100 = fully correct and "
        "fully supported by the context; 0 = wrong or unsupported).\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer to grade: {answer}\n\n"
        "Respond ONLY with a JSON object in this exact form, no other text:\n"
        '{"score": <integer 0-100>, "justification": "<one or two sentences>"}'
    )


def parse_score(raw: str) -> Tuple[int, str]:
    """Extracts {"score": int, "justification": str} from the model's raw output,
    falling back gracefully if the model didn't return clean JSON."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            score = int(obj.get("score", -1))
            justification = str(obj.get("justification", "")).strip()
            if 0 <= score <= 100:
                return score, justification
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return -1, f"[could not parse score from model output]: {raw[:200]}"


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local RAG loop with self-evaluation.")
    parser.add_argument("--docs-dir", type=Path, required=True, help="Folder of .txt/.md files")
    parser.add_argument("--model", default="llama3", help="Ollama model tag (default: llama3)")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2",
                         help="sentence-transformers model name for retrieval embeddings")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve per query")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--log-file", type=Path, default=None,
                         help="Optional path to append JSON-lines records of each turn")
    args = parser.parse_args()

    print(f"[setup] loading documents from {args.docs_dir} ...")
    chunks = build_chunks(args.docs_dir, args.chunk_size, args.chunk_overlap)
    print(f"[setup] built {len(chunks)} chunks from the knowledge base")

    retriever = Retriever(chunks, args.embed_model)
    llm = OllamaClient(args.model, args.ollama_host, args.temperature)

    log_fh = open(args.log_file, "a", encoding="utf-8") if args.log_file else None

    print("\nReady. Type a question, or 'exit' / 'quit' to stop.\n")

    try:
        while True:
            try:
                query = input("Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                break

            retrieved = retriever.search(query, top_k=args.top_k)

            rag_prompt = build_rag_prompt(query, retrieved)
            print("\n[retrieving + generating ...]")
            try:
                answer = llm.generate(rag_prompt)
            except requests.RequestException as e:
                print(f"[error] could not reach Ollama at {args.ollama_host}: {e}", file=sys.stderr)
                continue

            print(f"\nAnswer:\n{answer}\n")

            eval_prompt = build_eval_prompt(query, retrieved, answer)
            try:
                eval_raw = llm.generate(eval_prompt)
            except requests.RequestException as e:
                print(f"[error] self-evaluation request failed: {e}", file=sys.stderr)
                eval_raw = ""

            score, justification = parse_score(eval_raw)
            if score >= 0:
                print(f"Self-evaluated correctness: {score}/100")
                print(f"Justification: {justification}\n")
            else:
                print(f"Self-evaluation unavailable. {justification}\n")

            print("Sources used:")
            for chunk, sim in retrieved:
                print(f"  - {chunk.source} (chunk {chunk.chunk_id}, similarity {sim:.3f})")
            print()

            if log_fh:
                record = {
                    "query": query,
                    "answer": answer,
                    "score": score,
                    "justification": justification,
                    "sources": [
                        {"source": c.source, "chunk_id": c.chunk_id, "similarity": sim}
                        for c, sim in retrieved
                    ],
                }
                log_fh.write(json.dumps(record) + "\n")
                log_fh.flush()

    finally:
        if log_fh:
            log_fh.close()

    print("Goodbye.")


if __name__ == "__main__":
    main()
