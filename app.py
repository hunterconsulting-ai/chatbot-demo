"""
app.py — RAG Chatbot Demo for Acme Corp
Hunter Consulting — Portfolio Demo

What this app does:
  1. At startup, auto-ingests the knowledge base into ChromaDB if not already loaded
  2. Serves a chat UI at http://localhost:8080 (or PORT env var)
  3. On each user question:
     a. Queries ChromaDB for the 3 most semantically similar chunks
        (ChromaDB handles embedding the query automatically)
     b. Sends retrieved context + question to Claude (Haiku)
     c. Returns Claude's answer + which source files were used

Run:
  python app.py
"""

import os
import glob
import anthropic
import chromadb
from flask import Flask, send_from_directory, request, jsonify
from dotenv import load_dotenv

# Load .env from workspace root when running locally
# On Railway, ANTHROPIC_API_KEY is set as an environment variable directly
_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", ".env"))
load_dotenv(_env_path)

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "acme_corp"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

app = Flask(__name__, static_folder="public")

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError("ANTHROPIC_API_KEY not set.")

anthopic_client = anthropic.Anthropic(api_key=api_key)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)


# ── Ingestion (runs at startup if collection is missing) ────────────────────

def chunk_text(text):
    chunks = []
    stride = CHUNK_SIZE - CHUNK_OVERLAP
    start = 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += stride
    return chunks


def run_ingest():
    print("Knowledge base not found — running ingestion...")
    files = []
    for path in glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.md")):
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        files.append((filename, content))
        print(f"  Read: {filename}")

    all_chunks, all_ids, all_metadata = [], [], []
    for filename, content in files:
        for i, chunk in enumerate(chunk_text(content)):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}::chunk{i}")
            all_metadata.append({"source": filename})

    print(f"  Chunked: {len(all_chunks)} total chunks — generating embeddings...")
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(ids=all_ids, documents=all_chunks, metadatas=all_metadata)
    print(f"  Done. Ingested {len(all_chunks)} chunks from {len(files)} files.")
    return collection


# Load or create collection at startup
try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"Loaded collection '{COLLECTION_NAME}' ({collection.count()} chunks)")
except Exception:
    collection = run_ingest()


# ── RAG helpers ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful customer service assistant for Acme Corp, a home goods retailer.

Answer the customer's question using ONLY the context provided below. Be friendly, clear, and conversational — write the way a knowledgeable customer service representative would speak, not like a formatted document.

Formatting rules:
- Do not use markdown formatting of any kind. No hashtags, no asterisks for bold or italics, no dash bullet points.
- Write in plain prose sentences.
- Use numbered steps (1. 2. 3.) only when walking through a sequence of actions, written naturally in the flow of your response.

If the answer is not contained in the provided context, say exactly:
"I don't have information about that in my knowledge base. For further help, please contact our customer service team at 1-800-555-2674 or support@acmecorp.example.com."

Do not make up information. Do not answer from your general knowledge.

Smart navigation: When the customer's question is clearly focused on one specific topic area, end your response with this exact sentence: "More information about this can be found on the [Page Name] page. Would you like me to take you there now?" — replacing [Page Name] with the most relevant page from this list:
- Products (questions about what Acme Corp sells, specific items, or best sellers)
- Shipping (delivery times, costs, tracking, or carriers)
- Returns (return window, how to start a return, refunds, or exchanges)
- FAQ (account questions, orders, warranty, gift cards, or password reset)
- Pricing (price match, payment methods, Rewards program, or financing)
- About (company background, contact info, or business hours)

Only add this sentence when the question maps clearly to one page. Do not add it for general or multi-topic questions."""


def retrieve_context(question):
    results = collection.query(
        query_texts=[question],
        n_results=TOP_K,
        include=["documents", "metadatas"],
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return chunks, sources


def build_context_block(chunks, sources):
    parts = []
    for i, (chunk, source) in enumerate(zip(chunks, sources), 1):
        parts.append(f"[Source {i}: {source}]\n{chunk}")
    return "\n\n---\n\n".join(parts)


def ask_claude(user_question, context_block):
    user_message = f"""Context from Acme Corp knowledge base:

{context_block}

---

Customer question: {user_question}"""

    response = anthopic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


# ── Routes ────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("public", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or not data.get("question", "").strip():
        return jsonify({"error": "No question provided."}), 400

    question = data["question"].strip()

    try:
        chunks, sources = retrieve_context(question)
        context_block = build_context_block(chunks, sources)
        answer = ask_claude(question, context_block)
        unique_sources = list(dict.fromkeys(sources))
        return jsonify({"answer": answer, "sources": unique_sources})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\nAcme Corp RAG Chatbot running at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
