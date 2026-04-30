"""
crime_mapper.py – Multi-label crime classification (phishing, scam, fraud, etc.)
"""
import re

CRIME_KEYWORDS = {
    "phishing":   ["phishing", "click here", "verify your account", "update your info",
                   "login link", "confirm password", "account suspended", "unusual activity"],
    "scam":       ["scam", "fake", "win prize", "you won", "lottery", "free gift",
                   "limited offer", "act now", "claim reward", "too good to be true"],
    "fraud":      ["fraud", "identity theft", "stolen card", "unauthorized charge",
                   "bank fraud", "wire transfer", "payment fraud", "false claim"],
    "hate_speech":["terrorist", "extremist", "jihad", "infidel", "kafir", "bomb threat",
                   "kill all", "ethnic cleansing", "genocide", "white supremacy"],
    "harassment": ["harass", "bully", "stalk", "threaten", "doxx", "blackmail",
                   "revenge porn", "swat", "cyberbullying"],
    "misinformation": ["fake news", "hoax", "conspiracy", "plandemic", "deep state",
                       "crisis actor", "false flag", "rigged election"],
}

def classify_crime(text: str) -> str:
    """
    Returns the first matched crime category or 'none'.
    Multi-label: returns the highest-priority match.
    """
    text_lower = text.lower()
    for category, keywords in CRIME_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return category
    return "none"

def classify_crime_multilabel(text: str) -> list:
    """Returns all matched crime categories."""
    text_lower = text.lower()
    matched = []
    for category, keywords in CRIME_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(category)
    return matched if matched else ["none"]
