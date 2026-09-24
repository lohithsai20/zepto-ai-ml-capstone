# Support Assistant

## Baseline
The graded path is offline/mock LLM mode. `MOCK_LLM` defaults to `1`; no LLM provider is contacted. Embeddings are generated locally with `all-MiniLM-L6-v2` and persisted in ChromaDB.

## Architecture
```text
8 policy TXT files
      |
      v
ingest.py: per-document chunks
      |
      v
SentenceTransformer(all-MiniLM-L6-v2)
      |
      v
ChromaDB collection: zepto_policy
      |
      v
FastAPI /ask -> LangGraph classify_intent
                    |
             +------+------+
             |             |
       policy_question  general_question
             |             |
             v             v
 retrieve_and_answer   direct_answer
             |             |
             +------+------+
                    v
        Pydantic AskResponse JSON
```

Ingestion reads the eight files from `docs/`, makes one chunk per document, embeds each chunk in `ingest.py`, and stores vectors plus document/chunk IDs in the `zepto_policy` ChromaDB collection. `retrieve_and_answer` embeds the query and performs top-3 cosine retrieval. In mock mode it creates the deterministic `Based on the retrieved context: ...` answer from the top chunk. In optional real-LLM mode only the generation step changes: the structured role/context/task/format/length prompt in `prompts.py` is sent to the LLM. `classify_intent` also has an optional real-LLM branch; the required default uses the exact keyword heuristic. `direct_answer` is fixed in mock mode and optionally uses the LLM in real mode. Pydantic validates the final response.

## Run locally
From the repository root:
```bash
python -m support_assistant.ingest
MOCK_LLM=1 uvicorn support_assistant.main:app --reload --port 7860
```

Example calls with mock mode:
```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d "{\"query\":\"What is the delivery fee?\"}"
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d "{\"query\":\"What is the capital of France?\"}"
```

Record the exact JSON returned by your local run in this README before submission. The first query must route to `retrieve_and_answer`; the second must route to `direct_answer`.

## Docker
From the repository root:
```bash
docker build -f support_assistant/Dockerfile -t zepto-support-assistant .
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
```
Then POST to `http://127.0.0.1:7860/ask`.

## Optional real LLM
Set `MOCK_LLM=0` and provide `GROQ_API_KEY` without committing it. The real path uses the structured prompt and retries invalid JSON up to two additional times. This extension is not required for grading.

## Example API Calls

### 1. Policy retrieval example

Request:

```json
{
  "query": "How can I track my order?"
}
```
Response:

```json
{
  "answer": "Based on the retrieved context: Every Zepto order shows a live rider-tracking map from the moment it is packed until delivery, accessible from the 'Track Order' screen. Estimated delivery time updates automatically as the rider move",
  "sources": [
    "doc_04_chunk_0",
    "doc_06_chunk_0",
    "doc_01_chunk_0"
  ],
  "confidence": 1.0
}
```
This query is classified as `policy_question` and routed to `retrieve_and_answer`.

### 2. General-question example

Request:

```json
{
  "query": "What is the capital of France?"
}
```
Response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```
This query is classified as `general_question` and routed to `direct_answer`.

