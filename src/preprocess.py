"""
preprocess.py – Tweet cleaning utilities
"""
import re


def clean_tweet(text: str) -> str:
    """Return a cleaned, normalised version of a raw tweet string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"^rt\s+", "", text)
    text = re.sub(r"&amp;|&lt;|&gt;|&quot;|&#39;", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
