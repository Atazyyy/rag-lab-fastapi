"""RAG Lab - Step 2: API tanya-jawab (FastAPI)

Retrieval: cari chunk paling relevan dari ChromaDB (embedding makna, bukan kata kunci)
Generation: potongan teratas disisipkan ke prompt, then dikirim ke LLM via env:
    GENERATOR_BASE_URL  (default: http://127.0.0.1:20128/v1 -> 9Router)
    GENERATOR_API_KEY
    GENERATOR_MODEL     (default: oc/person...)
Kalau env tidak diisi, endpoint tetap jalan dalam mode retrieval-only.
"""

import os
import pathlib

import chromadb
import httpx
from fastembed import TextEmbedding
from fastapi import FastAPI
from pydantic import BaseModel

DB_DIR = pathlib.Path(__file__).parent / "db"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

GENERATOR_BASE_URL = os.environ.get(
    "GENERATOR_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)
GENERATOR_API_KEY = os.environ.get("GENERATOR_API_KEY", "")
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "gemini-3.5-flash")

app = FastAPI(title="RAG Lab", version="0.1.0")
embedder = TextEmbedding(model_name=EMBED_MODEL)
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_or_create_collection("rag_lab")


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


def retrieve(question: str, top_k: int) -> list[dict]:
    q_vec = list(embedder.embed([question]))[0].tolist()
    res = collection.query(query_embeddings=[q_vec], n_results=top_k)
    hits = []
    for doc, dist in zip(res["documents"][0], res["distances"][0]):
        hits.append({"text": doc, "distance": round(dist, 3)})
    return hits


def generate(question: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(contexts))
    system = (
        "Kamu asisten yang menjawab HANYA dari konteks yang diberikan. "
        "Jika konteks tidak memuat jawaban, jawab: 'Tidak ada di dokumen saya.'"
    )
    user = f"Konteks:\n{context_block}\n\nPertanyaan: {question}"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {"maxOutputTokens": 800},
    }
    headers = {"x-goog-api-key": GENERATOR_API_KEY}
    model = GENERATOR_MODEL.lstrip("/")
    r = httpx.post(
        f"{GENERATOR_BASE_URL}/models/{model}:generateContent",
        json=body,
        headers=headers,
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


@app.get("/health")
def health():
    return {"chunks": collection.count()}


@app.post("/ask")
def ask(req: AskRequest):
    hits = retrieve(req.question, max(req.top_k, 1))
    answer = None
    if GENERATOR_API_KEY and GENERATOR_MODEL:
        try:
            answer = generate(req.question, hits)
        except Exception as e:
            answer = f"(generator gagal: {e}) — ini jawaban mode retrieval-only."
    else:
        answer = "(mode retrieval-only: set GENERATOR_MODEL env untuk jawaban LLM)"
    return {
        "question": req.question,
        "answer": answer,
        "sources": hits,
    }
