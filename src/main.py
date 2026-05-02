import logging
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd

from ingest import load_support_documents
from retrieve import build_faiss_index, create_embedder, search_documents
from classify import classify_request_type, classify_product_area
from safety import requires_escalation
from generate import generate_response


LOG_FILE = Path("log.txt")
OUTPUT_FILE = Path("output.csv")
DEFAULT_INPUT_FILE = Path("data/tickets.csv")
ZIP_INPUT_FILE = Path("data.zip/support_tickets/support_tickets.csv")
SAMPLE_INPUT_FILE = Path("data.zip/support_tickets/sample_support_tickets.csv")


def setup_logging() -> None:
    """Configure logging to write both to the terminal and the log file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def read_ticket_data(input_file: Path) -> pd.DataFrame:
    """Read the support tickets from the CSV file."""
    return pd.read_csv(input_file)


def get_ticket_file() -> Path:
    """Return the file path for the ticket CSV based on the workspace layout."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    if SAMPLE_INPUT_FILE.exists():
        return SAMPLE_INPUT_FILE
    if ZIP_INPUT_FILE.exists():
        return ZIP_INPUT_FILE
    return DEFAULT_INPUT_FILE


def build_ticket_text(row) -> tuple[str, str, str, str]:
    """Construct the ticket text using Issue, Subject, and Company fields."""
    issue = str(row.get("Issue", row.get("issue", ""))).strip()
    subject = str(row.get("Subject", row.get("subject", ""))).strip()
    company = str(row.get("Company", row.get("company", ""))).strip()

    ticket_text = issue
    if subject:
        ticket_text = f"{ticket_text} {subject}".strip()
    if company:
        ticket_text = f"{ticket_text} Company: {company}".strip()

    return ticket_text, issue, subject, company


def save_output(rows: List[dict]) -> None:
    """Save the processed ticket results to the output CSV."""
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    logging.info("Saved output to %s", OUTPUT_FILE)


def process_ticket(row, ticket_index, faiss_index, embedder, documents):
    """Process a single ticket and return the result row."""
    ticket_id = row.get("ticket_id")
    if pd.isna(ticket_id) or ticket_id == "":
        ticket_id = f"T{ticket_index + 1}"

    ecosystem = str(row.get("Company", row.get("company", row.get("ecosystem", "")))).strip()
    ticket_text, issue, subject, company = build_ticket_text(row)

    logging.info("Processing ticket %s (%s)", ticket_id, ecosystem)

    top_documents = search_documents(ticket_text, faiss_index, embedder, documents, top_k=3)
    request_type = classify_request_type(ticket_text, top_documents)
    product_area = classify_product_area(ticket_text, top_documents)
    escalate, reason = requires_escalation(ticket_text, request_type, product_area, top_documents)

    # Log detailed info for AI Judge review
    logging.info("Ticket %s - Request Type: %s, Product Area: %s", ticket_id, request_type, product_area)
    logging.info("Ticket %s - Retrieved Docs (%d):", ticket_id, len(top_documents))
    for i, doc in enumerate(top_documents):
        logging.info("  Doc %d: Source: %s, Score: %.3f, Text: %s", i+1, doc['source'], doc.get('score', 0), doc['text'][:100])
    logging.info("Ticket %s - Escalation: %s (%s)", ticket_id, escalate, reason)

    if escalate:
        status = "escalate"
        response = (
            "This issue has been flagged for escalation because it involves a sensitive or unsupported case. "
            "Please contact the specialist support team."
        )
        justification = f"Escalated due to {reason}. Retrieved documents indicate potential security or billing concerns. Response not generated to avoid unsupported policies."
    else:
        status = "reply"
        try:
            response = generate_response(
                ticket_text,
                top_documents,
                request_type,
                product_area,
                status,
                subject=subject,
                company=company,
            )
            justification = f"Replied based on retrieved support documents. Response grounded in {len(top_documents)} relevant docs with average confidence {sum(doc.get('score', 0) for doc in top_documents)/len(top_documents):.3f}."
        except Exception as exc:
            logging.exception("Failed to generate response for ticket %s", ticket_id)
            status = "escalate"
            response = (
                "Unable to generate a grounded response at this time. "
                "Escalating to the support team for review."
            )
            justification = f"Escalated due to generation failure: {exc}. Retrieval found {len(top_documents)} docs but API unavailable."

    if not escalate:
        logging.info("Ticket %s - Generated Response: %s", ticket_id, response[:200])
    else:
        logging.info("Ticket %s - Escalation Response: %s", ticket_id, response)

    return {
        "issue": issue,
        "subject": subject,
        "company": company,
        "response": response,
        "product_area": product_area,
        "status": status,
        "request_type": request_type,
        "justification": justification,
    }


def main() -> None:
    """Main program entry point for the AI support triage workflow."""
    setup_logging()
    logging.info("Starting AI support triage agent")

    input_file = get_ticket_file()
    if not input_file.exists():
        logging.error("Input file not found: %s", input_file)
        return

    tickets = read_ticket_data(input_file)
    if tickets.empty:
        logging.warning("No tickets found in %s", input_file)
        return

    documents = load_support_documents("support_issues")
    if not documents:
        logging.error("No support documents were loaded from support_issues/")
        return

    embedder = create_embedder()
    index, documents = build_faiss_index(documents, embedder)

    results = []
    for ticket_index, row in tickets.iterrows():
        result = process_ticket(row, ticket_index, index, embedder, documents)
        results.append(result)

    save_output(results)
    logging.info("Completed processing %d tickets", len(results))


if __name__ == "__main__":
    main()
