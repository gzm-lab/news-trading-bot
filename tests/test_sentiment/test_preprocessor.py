"""Tests for text preprocessor."""

import pytest

from src.sentiment.preprocessor import clean_text, truncate_for_model


class TestCleanText:
    def test_removes_urls(self):
        text = "Check this out https://example.com/article and http://test.org"
        result = clean_text(text)
        assert "https://" not in result
        assert "http://" not in result

    def test_removes_html_entities(self):
        text = "Apple &amp; Microsoft &lt;both&gt; rise"
        result = clean_text(text)
        assert "&amp;" not in result
        assert "&lt;" not in result

    def test_normalizes_whitespace(self):
        text = "Too   many    spaces\n\nand\nnewlines"
        result = clean_text(text)
        assert "  " not in result
        assert "\n" not in result

    def test_preserves_punctuation(self):
        text = "Apple's revenue grew 15%, beating expectations!"
        result = clean_text(text)
        assert "'" in result
        assert "," in result
        assert "!" in result

    def test_strips_surrounding_whitespace(self):
        text = "  leading and trailing  "
        result = clean_text(text)
        assert result == "leading and trailing"

    def test_empty_string(self):
        assert clean_text("") == ""


class TestTruncateForModel:
    def test_short_text_unchanged(self):
        text = "Short text that fits easily."
        result = truncate_for_model(text, max_length=512)
        assert result == text

    def test_long_text_truncated(self):
        text = "Word " * 1000
        result = truncate_for_model(text, max_length=100)
        assert len(result) <= 100 * 4 + 10  # Some buffer for sentence boundary

    def test_cuts_at_sentence_boundary(self):
        # First sentence at position ~100, second at ~200
        text = "A" * 95 + ". This is the second sentence. " + "B" * 2000
        result = truncate_for_model(text, max_length=50)
        # Should cut at the period
        assert result.endswith(".")

    def test_custom_max_length(self):
        text = "X " * 500
        result = truncate_for_model(text, max_length=50)
        assert len(result) <= 50 * 4 + 10
