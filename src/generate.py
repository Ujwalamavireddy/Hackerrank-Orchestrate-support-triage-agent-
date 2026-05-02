import os
import logging
from pathlib import Path
from typing import Dict, List

import google.generativeai as genai
from dotenv import load_dotenv


def configure_gemini() -> bool:
    """Configure the Gemini API client if an API key is available."""
    repo_root = Path(__file__).resolve().parents[1]
    dotenv_path = repo_root / ".env"
    logging.info("Attempting to load .env from %s", dotenv_path)
    logging.info("dotenv_path exists: %s", dotenv_path.exists())

    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
    else:
        logging.warning(".env file not found at %s", dotenv_path)

    api_key = os.getenv("GOOGLE_API_KEY")
    logging.info("after load_dotenv, GOOGLE_API_KEY: %s", api_key is not None)
    if not api_key:
        logging.warning(
            "GOOGLE_API_KEY not found in environment. cwd=%s. "
            "Ensure .env is present and the script is running from the repo root.",
            Path.cwd(),
        )
        return False

    try:
        genai.configure(api_key=api_key)
    except Exception as exc:
        logging.exception("Failed to configure Gemini API client")
        raise RuntimeError("Gemini configuration failed") from exc

    logging.info("Gemini client configured successfully with GOOGLE_API_KEY from %s", dotenv_path)
    return True


def build_prompt(
    ticket_text: str,
    top_documents: List[Dict[str, str]],
    request_type: str,
    product_area: str,
    action: str,
    subject: str = "",
    company: str = "",
) -> str:
    """Build a grounded prompt for Gemini that uses only retrieved support documents."""
    source_lines = []
    for document in top_documents:
        source_lines.append(f"SOURCE: {document['source']}\n{document['text']}\n")

    document_section = "\n---\n".join(source_lines)

    subject_line = f"Subject: {subject}\n" if subject else ""
    company_line = f"Company/Ecosystem: {company}\n" if company else ""

    prompt = (
        "You are an AI support assistant. Answer only using the information from the provided support documents. "
        "Do not add any facts that are not directly supported by the documents. If the answer is not clear from the documents, "
        "say you are unable to answer from the provided corpus and recommend escalation.\n\n"
        f"Ticket text: {ticket_text}\n"
        f"{subject_line}"
        f"{company_line}"
        f"Request type: {request_type}\n"
        f"Product area: {product_area}\n"
        f"Action: {action}\n\n"
        "Support documents:\n"
        f"{document_section}\n\n"
        "Provide a short, clear support answer and mention the source names in the response."
    )
    return prompt


def generate_response(
    ticket_text: str,
    top_documents: List[Dict[str, str]],
    request_type: str,
    product_area: str,
    action: str,
    subject: str = "",
    company: str = "",
) -> str:
    """Generate a reply from Gemini using the retrieved documents."""
    has_gemini_key = configure_gemini()

    prompt = build_prompt(
        ticket_text,
        top_documents,
        request_type,
        product_area,
        action,
        subject=subject,
        company=company,
    )
    logging.debug("Gemini prompt:\n%s", prompt)

    if not has_gemini_key:
        sources = ", ".join(doc["source"] for doc in top_documents)
        return (
            f"Based on the retrieved support documents ({sources}), this response is grounded in the corpus. "
            "Configure GOOGLE_API_KEY to use Gemini for full generation."
        )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=256,
            ),
        )
        text = getattr(response, "text", None)
        if text is None:
            raise RuntimeError("Gemini returned empty response")
        return text.strip()
    except Exception as exc:
        error_msg = str(exc).lower()
        if "429" in error_msg or "quota" in error_msg or "rate limit" in error_msg:
            logging.warning("Rate limit hit, retrying after delay")
            import time
            time.sleep(60)  # Wait 60 seconds
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=256,
                    ),
                )
                text = getattr(response, "text", None)
                if text is None:
                    raise RuntimeError("Gemini returned empty response on retry")
                return text.strip()
            except Exception as retry_exc:
                logging.exception("Gemini retry failed")
                raise RuntimeError(f"Gemini generation failed after retry: {retry_exc}") from retry_exc
        logging.exception("Gemini API request failed")
        raise RuntimeError(f"Gemini generation failed: {exc}") from exc

    if text is None:
        text = str(response)

    return text.strip()
