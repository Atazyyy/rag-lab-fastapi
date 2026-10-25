"""RAG Lab - Step 1: Ingest dokumen -> chunking -> embedding -> ChromaDB"""

import pathlib
from fastembed import TextEmbedding
import chromadb

DOCS_DIR = pathlib.Path(__file__).parent / "docs"
DB_DIR = pathlib.Path(__file__).parent / "db"
CHUNK_CHARS = 400
CHUNK_OVERLAP = 80
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_documents() -> list[str]:
    texts = []
    for path in DOCS_DIR.glob("*.txt"):
        texts.append(path.read_text(encoding="utf-8"))
    return texts


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def main() -> None:
    print("1/5 Load dokument dari", DOCS_DIR)
    docs = load_documents()
    if not docs:
        print("!! Tidak ada file .txt di docs/ — taruh dulu dokumennya")
        return

    print("2/5 Chunking (potong", CHUNK_CHARS, "karakter, overlap", CHUNK_OVERLAP, ")")
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))
    print("   ->", len(chunks), "chunk siap di-embed")

    print("3/5 Load model embedding:", EMBED_MODEL)
    embedder = TextEmbedding(model_name=EMBED_MODEL)

    print("4/5 Embedding semua chunk (lokal, gratis, tanpa API key)")
    vectors = list(embedder.embed(chunks))

    print("5/5 Simpan ke ChromaDB di", DB_DIR)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection("rag_lab")
    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=[v.tolist() for v in vectors],
    )

    print()
    print("SELESAI. Isi database:")
    print("  - chunk    :", collection.count())
    print("  - dimensi  :", len(vectors[0]))


if __name__ == "__main__":
    main()
