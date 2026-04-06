"""
ingest.py — Run once to load the knowledge base into ChromaDB.

What this script does:
  1. Reads all .md files from the knowledge_base/ folder
  2. Splits each file into overlapping chunks (~500 characters)
  3. Stores the chunks in ChromaDB — ChromaDB generates embeddings automatically
     using a local sentence-transformer model (all-MiniLM-L6-v2, ~80MB download on first run)

Run this before starting app.py (local dev only — app.py auto-ingests on Railway):
  python ingest.py
"""

import os
import glob
import chromadb

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "acme_corp"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def read_markdown_files(directory):
    files = []
    for path in glob.glob(os.path.join(directory, "*.md")):
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        files.append((filename, content))
        print(f"  Read: {filename} ({len(content)} characters)")
    return files


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    stride = chunk_size - overlap
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += stride
    return chunks


def main():
    print("=== Acme Corp Knowledge Base Ingestion ===\n")

    print("Reading knowledge base files...")
    files = read_markdown_files(KNOWLEDGE_BASE_DIR)
    if not files:
        raise FileNotFoundError(f"No .md files found in {KNOWLEDGE_BASE_DIR}")

    print("\nChunking documents...")
    all_chunks, all_ids, all_metadata = [], [], []

    for filename, content in files:
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}::chunk{i}")
            all_metadata.append({"source": filename})
        print(f"  {filename}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"\nStoring in ChromaDB at: {CHROMA_DB_DIR}")
    print("(First run will download the embedding model — this may take a minute)")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("  Deleted existing collection (re-ingesting fresh)")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(ids=all_ids, documents=all_chunks, metadatas=all_metadata)

    print(f"\nDone. Ingested {len(all_chunks)} chunks from {len(files)} files.")
    print("You can now run: python app.py")


if __name__ == "__main__":
    main()
