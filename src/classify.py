from typing import Dict, List

REQUEST_TYPE_KEYWORDS = {
    "Fraud Report": [
        "fraud",
        "scam",
        "suspicious charge",
        "suspicious activity",
        "unauthorized charge",
        "identity theft",
        "compromised account",
    ],
    "Unauthorized Access": [
        "unauthorized access",
        "unauthorized login",
        "access denied",
        "login failed",
        "can\'t log in",
        "cannot log in",
        "account compromise",
    ],
    "Password Reset": [
        "forgot password",
        "reset password",
        "change password",
        "password reset",
        "password help",
    ],
    "Refund Request": [
        "refund",
        "chargeback",
        "money back",
        "billing dispute",
        "double charged",
        "overcharged",
    ],
    "Billing Question": [
        "billing question",
        "invoice",
        "payment due",
        "billing issue",
        "charge failed",
        "payment failed",
        "charged twice",
    ],
    "Technical Bug": [
        "error",
        "bug",
        "crash",
        "not working",
        "technical issue",
        "page broken",
        "screen freeze",
    ],
    "Feature Request": [
        "feature request",
        "request new",
        "suggestion",
        "enhancement",
        "new feature",
        "improve",
    ],
    "Account Management": [
        "update profile",
        "change settings",
        "profile settings",
        "account settings",
        "close account",
        "delete account",
        "manage account",
        "account details",
    ],
    "Assessment Question": [
        "assessment",
        "coding assessment",
        "test",
        "challenge",
        "exam",
        "evaluation",
    ],
    "Subscription Issue": [
        "subscription",
        "renewal",
        "plan",
        "membership",
        "cancel subscription",
        "upgrade plan",
    ],
}

PRODUCT_AREA_KEYWORDS = {
    "Security": [
        "security",
        "breach",
        "compromise",
        "threat",
        "hacked",
        "privacy",
        "identity theft",
    ],
    "Fraud": [
        "fraud",
        "scam",
        "suspicious",
        "unauthorized charge",
        "identity theft",
    ],
    "Authentication": [
        "log in",
        "login",
        "sign in",
        "sign-in",
        "password",
        "two-factor",
        "2fa",
        "access denied",
        "account access",
    ],
    "Billing": [
        "billing",
        "charge",
        "invoice",
        "refund",
        "payment",
        "overcharged",
        "chargeback",
    ],
    "Assessments": [
        "assessment",
        "coding assessment",
        "test",
        "challenge",
        "exam",
        "evaluation",
    ],
    "Account Management": [
        "profile",
        "settings",
        "account settings",
        "update profile",
        "change settings",
        "account details",
        "manage account",
    ],
    "Technical Issues": [
        "error",
        "bug",
        "crash",
        "not working",
        "technical issue",
        "broken",
        "fail",
    ],
    "Subscription": [
        "subscription",
        "renewal",
        "plan",
        "membership",
        "cancel",
        "upgrade",
    ],
    "Product Feedback": [
        "feature",
        "suggestion",
        "request",
        "enhancement",
        "feedback",
    ],
}


def classify_request_type(ticket_text: str, top_documents: List[Dict[str, str]]) -> str:
    """Classify the intent of the support ticket using keyword matching."""
    lower_text = ticket_text.lower()

    for request_type, keywords in REQUEST_TYPE_KEYWORDS.items():
        if any(keyword in lower_text for keyword in keywords):
            return request_type

    # If the support documents mention a stronger category, use that as a fallback.
    for document in top_documents:
        text = document["text"].lower()
        for request_type, keywords in REQUEST_TYPE_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return request_type

    return "General Help"


def classify_product_area(ticket_text: str, top_documents: List[Dict[str, str]]) -> str:
    """Assign a product area based on ticket content and retrieved support documents."""
    lower_text = ticket_text.lower()

    for product_area, keywords in PRODUCT_AREA_KEYWORDS.items():
        if any(keyword in lower_text for keyword in keywords):
            return product_area

    for document in top_documents:
        text = document["text"].lower()
        for product_area, keywords in PRODUCT_AREA_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return product_area

    return "Other"
