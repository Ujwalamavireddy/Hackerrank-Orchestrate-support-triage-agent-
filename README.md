# AI Support Triage Agent

A terminal-based Python support triage system for HackerRank Support, Claude Help Center, and Visa Support.

## Architecture Overview

The system consists of the following components:

- **Data Ingestion**: Loads support documents from `support_issues/` directory (claude.txt, hackerrank.txt, visa.txt).
- **Retrieval Pipeline**: Uses FAISS vector search with sentence-transformers (all-MiniLM-L6-v2) for semantic similarity retrieval.
- **Classification**: Hybrid rule-based and retrieval-assisted classification of request type (Security, Billing, Account Update, Bug Report, Feature Request) and product area (Authentication, Billing, Account, Security, Technical, Documentation).
- **Escalation Logic**: Automatically escalates sensitive topics (security, fraud, billing disputes) or uncertain classifications.
- **Response Generation**: Grounded responses using Gemini API, restricted to retrieved documents only.
- **Hallucination Prevention**: Strict prompting ensures responses rely solely on corpus; escalates if insufficient information.

## Retrieval Pipeline

1. Embed support documents using sentence-transformers.
2. Build normalized FAISS index for cosine similarity search.
3. For each ticket, retrieve top-3 most relevant documents.
4. Use retrieved documents to classify and generate responses.

## Escalation Logic

Tickets are escalated if:
- Keywords indicate security/fraud/billing issues.
- Request type is Security or Billing Question.
- Product area is Security or Billing.
- No relevant documents found.
- Classification is uncertain (General Help + Other).

## Hallucination Prevention Strategy

- Prompt explicitly forbids unsupported claims.
- Responses must cite sources and escalate if unclear.
- Gemini temperature set to 0.0 for deterministic output.
- Fallback to escalation on generation failure.

## Setup Instructions

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Gemini API key in `.env`:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```bash
   python src/main.py
   ```

## Design Decisions

- **Keyword Classification**: Simple and fast, avoids ML complexity for hackathon.
- **FAISS Retrieval**: Efficient for small corpus, normalized embeddings for better similarity.
- **Gemini Grounding**: API supports long contexts; prompt enforces document-only responses.
- **Escalation First**: Prioritizes safety over automation for sensitive topics.
- **Logging**: Detailed logs for debugging and AI Judge review.

## What this project does

- Reads incoming support tickets from `data.zip/support_tickets/sample_support_tickets.csv`
- Loads support corpus documents from `support_issues/`
- Builds a FAISS vector index with `sentence-transformers`
- Classifies request type and product area
- Checks safety and escalation rules
- Generates grounded responses using Gemini API
- Writes results to `output.csv`
- Logs activity to `log.txt`

## Output Schema

The generated `output.csv` contains:

- issue
- subject
- company
- response
- product_area
- status
- request_type
- justification
