"""Text preprocessor for financial news."""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Clean financial news text for NLP processing."""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove HTML entities
    text = re.sub(r'&\w+;', ' ', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special chars but keep basic punctuation
    text = re.sub(r"[^\w\s.,!?;:\'\-]", '', text)
    return text.strip()


def truncate_for_model(text: str, max_length: int = 512) -> str:
    """Truncate text to roughly fit within token limit.

    FinBERT has a 512 token limit. Rough heuristic: ~4 chars per token.
    """
    max_chars = max_length * 4
    if len(text) <= max_chars:
        return text
    # Try to cut at a sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > max_chars * 0.5:
        return truncated[:last_period + 1]
    return truncated
