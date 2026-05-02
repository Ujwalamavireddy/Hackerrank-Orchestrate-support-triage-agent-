from typing import Dict, List, Tuple

ESCALATION_KEYWORDS = [
    "fraud",
    "scam",
    "billing dispute",
    "unauthorized access",
    "identity theft",
    "account compromise",
    "security breach",
    "hacked",
    "privacy",
    "compromised",
    "threat",
]

ESCALATION_REQUEST_TYPES = [
    "Fraud Report",
    "Refund Request",
    "Unauthorized Access",
]

ESCALATION_PRODUCT_AREAS = [
    "Security",
    "Fraud",
    "Billing",
]


def requires_escalation(
    ticket_text: str,
    request_type: str,
    product_area: str,
    top_documents: List[Dict[str, str]],
) -> Tuple[bool, str]:
    """Decide whether a ticket should be escalated based on safety rules."""
    text = ticket_text.lower()

    if any(keyword in text for keyword in ESCALATION_KEYWORDS):
        return True, "escalation keyword detected"

    if request_type in ESCALATION_REQUEST_TYPES:
        return True, f"request type requires escalation: {request_type}"

    if product_area in ESCALATION_PRODUCT_AREAS:
        return True, f"product area requires escalation: {product_area}"

    for doc in top_documents:
        doc_text = doc["text"].lower()
        if any(keyword in doc_text for keyword in ESCALATION_KEYWORDS):
            return True, "retrieved document indicates escalation needed"

    return False, "no escalation needed"
