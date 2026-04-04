"""Tests for FinBERT analyzer — mocked model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.sentiment.finbert import FinBERTAnalyzer, SentimentResult


@pytest.fixture
def mock_analyzer():
    """Create analyzer with mocked model & tokenizer."""
    analyzer = FinBERTAnalyzer.__new__(FinBERTAnalyzer)
    analyzer._model_name = "ProsusAI/finbert"
    analyzer._batch_size = 16
    analyzer._max_length = 512
    analyzer._device = torch.device("cpu")
    analyzer._tokenizer = MagicMock()
    analyzer._model = MagicMock()
    return analyzer


class TestFinBERTAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_empty(self, mock_analyzer):
        results = await mock_analyzer.analyze([])
        assert results == []

    @pytest.mark.asyncio
    async def test_analyze_positive(self, mock_analyzer):
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_analyzer._tokenizer.return_value = mock_inputs

        # Positive: high prob for index 0
        mock_logits = torch.tensor([[2.0, -1.0, 0.0]])
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_analyzer._model.return_value = mock_output

        results = await mock_analyzer.analyze(["Great earnings report"])

        assert len(results) == 1
        assert results[0].label == "positive"
        assert results[0].score > 0

    @pytest.mark.asyncio
    async def test_analyze_negative(self, mock_analyzer):
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_analyzer._tokenizer.return_value = mock_inputs

        # Negative: high prob for index 1
        mock_logits = torch.tensor([[-1.0, 2.0, 0.0]])
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_analyzer._model.return_value = mock_output

        results = await mock_analyzer.analyze(["Company faces bankruptcy"])

        assert results[0].label == "negative"
        assert results[0].score < 0

    @pytest.mark.asyncio
    async def test_analyze_batch(self, mock_analyzer):
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_analyzer._tokenizer.return_value = mock_inputs

        # Two items: positive and negative
        mock_logits = torch.tensor([
            [2.0, -1.0, 0.0],   # positive
            [-1.0, 2.0, -0.5],  # negative
        ])
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_analyzer._model.return_value = mock_output

        results = await mock_analyzer.analyze([
            "Apple beats earnings expectations",
            "Tesla crashes after recall announcement",
        ])

        assert len(results) == 2
        assert all(isinstance(r, SentimentResult) for r in results)

        # First text should be positive
        assert results[0].label == "positive"
        assert results[0].score > 0

        # Second text should be negative
        assert results[1].label == "negative"
        assert results[1].score < 0

    @pytest.mark.asyncio
    async def test_analyze_single(self, mock_analyzer):
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_analyzer._tokenizer.return_value = mock_inputs

        mock_logits = torch.tensor([[1.0, -1.0, 0.0]])
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_analyzer._model.return_value = mock_output

        result = await mock_analyzer.analyze_single("Good news for markets")

        assert isinstance(result, SentimentResult)
        assert result.label == "positive"

    @pytest.mark.asyncio
    async def test_score_range(self, mock_analyzer):
        """Score should always be between -1 and +1."""
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_analyzer._tokenizer.return_value = mock_inputs

        # Extreme logits
        mock_logits = torch.tensor([[10.0, -10.0, 0.0]])
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_analyzer._model.return_value = mock_output

        result = await mock_analyzer.analyze_single("Extreme sentiment")
        assert -1.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_probabilities_sum_to_one(self, mock_analyzer):
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_analyzer._tokenizer.return_value = mock_inputs

        mock_logits = torch.tensor([[1.0, 0.5, -0.3]])
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_analyzer._model.return_value = mock_output

        result = await mock_analyzer.analyze_single("Test")
        total = sum(result.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-5)
