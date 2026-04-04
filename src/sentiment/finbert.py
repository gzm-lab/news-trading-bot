"""FinBERT sentiment analysis wrapper."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.sentiment.preprocessor import clean_text, truncate_for_model

log = structlog.get_logger()


@dataclass
class SentimentResult:
    """Sentiment analysis result for a single text."""

    text: str
    label: str  # "positive", "negative", "neutral"
    score: float  # -1.0 to +1.0 (negative to positive)
    confidence: float  # 0.0 to 1.0
    probabilities: dict[str, float]  # raw probabilities per class


class FinBERTAnalyzer:
    """FinBERT-based financial sentiment analyzer.

    Model: ProsusAI/finbert (~500MB, runs on CPU in ~200ms per batch of 16)
    Labels: positive, negative, neutral
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        batch_size: int = 16,
        max_length: int = 512,
    ):
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_length = max_length
        self._tokenizer = None
        self._model = None
        self._device = None

    async def load(self) -> None:
        """Load the model (downloads on first run, ~500MB)."""
        log.info("finbert.loading", model=self._model_name)

        def _load():
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            self._model.eval()

        await asyncio.to_thread(_load)
        log.info("finbert.loaded", device=str(self._device))

    async def analyze(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze a list of texts. Returns one SentimentResult per text."""
        if not texts:
            return []

        assert self._model is not None, "Model not loaded. Call load() first."

        # Preprocess
        cleaned = [truncate_for_model(clean_text(t), self._max_length) for t in texts]

        # Process in batches
        all_results: list[SentimentResult] = []
        for i in range(0, len(cleaned), self._batch_size):
            batch = cleaned[i : i + self._batch_size]
            original_batch = texts[i : i + self._batch_size]
            results = await asyncio.to_thread(self._predict_batch, batch, original_batch)
            all_results.extend(results)

        return all_results

    async def analyze_single(self, text: str) -> SentimentResult:
        """Analyze a single text."""
        results = await self.analyze([text])
        return results[0]

    def _predict_batch(
        self, texts: list[str], original_texts: list[str]
    ) -> list[SentimentResult]:
        """Run inference on a batch (called in thread)."""
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # FinBERT labels: positive (0), negative (1), neutral (2)
        label_map = {0: "positive", 1: "negative", 2: "neutral"}

        results = []
        for idx in range(len(texts)):
            probs = probabilities[idx].cpu().numpy()
            prob_dict = {
                "positive": float(probs[0]),
                "negative": float(probs[1]),
                "neutral": float(probs[2]),
            }

            # Predicted label = highest probability
            pred_idx = probs.argmax()
            label = label_map[int(pred_idx)]
            confidence = float(probs[pred_idx])

            # Composite score: positive - negative (range: -1 to +1)
            score = float(probs[0]) - float(probs[1])

            results.append(
                SentimentResult(
                    text=original_texts[idx],
                    label=label,
                    score=score,
                    confidence=confidence,
                    probabilities=prob_dict,
                )
            )

        return results
