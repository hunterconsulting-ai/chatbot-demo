# Acme Corp RAG Chatbot — Hunter Consulting Demo

A RAG (Retrieval-Augmented Generation) chatbot demo using:

- **ChromaDB** — local vector database
- **ChromaDB default embeddings** — sentence-transformer model (all-MiniLM-L6-v2, no extra API key needed)
- **Claude Haiku** — answer generation via Anthropic API
- **Flask** — web server, deployed on Railway

## Live Demo

Deployed on Railway — see the live URL in the repo description.

## Local Setup

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Set your API key

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your-key-here
```

### Step 3 — Ingest the knowledge base (first run only)

```bash
python ingest.py
```

### Step 4 — Start the app

```bash
python app.py
```

Open http://localhost:8080

## How to Demo

1. Ask: *"What is your return policy?"* → answers from `returns.md`, shows source tag
2. Ask: *"What is the capital of France?"* → declines honestly, no hallucination
3. Point out the **Sources** tags — demonstrates RAG transparency

## Customizing for a Real Client

1. Replace markdown files in `knowledge_base/` with the client's content
2. Update the system prompt in `app.py` (company name + tone)
3. Update the header in `public/index.html` (company name + avatar)
4. Redeploy — `app.py` auto-ingests on startup

## Key Design Decisions

| Decision | Reason |
|---|---|
| Local ChromaDB | No external vector DB account; zero cost; simple |
| Local embeddings | No embedding API needed; runs offline after first download |
| Claude Haiku | Fast + low cost for demos; easy to swap for Sonnet |
| Auto-ingest on startup | Survives Railway redeploys without manual step |
| Sources shown in UI | RAG transparency — key demo talking point |
