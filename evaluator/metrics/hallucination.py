"""Hallucination detection via NLI cross-encoder.

Uses cross-encoder/nli-deberta-v3-small to classify each (context, answer)
pair as entailment, contradiction, or neutral. Answers that contradict the
context are flagged as potential hallucinations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# NLI label order for cross-encoder/nli-deberta-v3-small
# Scores index: 0=contradiction, 1=entailment, 2=neutral
_LABELS = ["contradiction", "entailment", "neutral"]
_HALLUCINATION_THRESHOLD = 0.5  # contradiction score above this → hallucination


class NLIHallucinationChecker:
    """Lazy-loading wrapper around sentence-transformers CrossEncoder NLI model.

    The model is only loaded on first use to avoid startup latency.
    """

    MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or self.MODEL_NAME
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading NLI cross-encoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def predict(self, premise: str, hypothesis: str) -> tuple[str, float]:
        """Run NLI inference on a (premise, hypothesis) pair.

        Args:
            premise: The reference context passage.
            hypothesis: The model's answer to check.

        Returns:
            Tuple of (label, confidence_score) where label is one of
            'contradiction', 'entailment', 'neutral'.
        """
        model = self._load()
        import numpy as np

        scores = model.predict([(premise, hypothesis)])
        # scores shape: (1, 3) — [contradiction, entailment, neutral]
        probs = _softmax(scores[0])
        label_idx = int(np.argmax(probs))
        return _LABELS[label_idx], float(probs[label_idx])

    def is_hallucination(self, context: str, answer: str) -> tuple[bool, str, float]:
        """Determine if an answer contradicts its context.

        Returns:
            (is_hallucination, nli_label, nli_confidence)
        """
        label, score = self.predict(context, answer)
        flagged = label == "contradiction" and score >= _HALLUCINATION_THRESHOLD
        return flagged, label, score


def check_nli_hallucination(
    checker: NLIHallucinationChecker,
    context: str,
    answer: str,
) -> tuple[Optional[str], Optional[float], bool]:
    """Convenience wrapper for use inside the Evaluator.

    Returns:
        (nli_label, nli_score, is_hallucination)
    """
    if not context or not answer:
        return None, None, False
    try:
        is_hall, label, score = checker.is_hallucination(context, answer)
        return label, score, is_hall
    except Exception as exc:
        logger.warning("NLI check failed: %s", exc)
        return None, None, False


def _softmax(logits) -> list[float]:
    """Numerically stable softmax."""
    import numpy as np

    arr = np.array(logits, dtype=float)
    arr -= arr.max()
    exp_arr = np.exp(arr)
    return (exp_arr / exp_arr.sum()).tolist()


def batch_check_hallucination(
    checker: NLIHallucinationChecker,
    pairs: list[tuple[str, str]],
) -> list[tuple[str, float, bool]]:
    """Batch NLI inference for efficiency.

    Args:
        checker: Initialized NLIHallucinationChecker.
        pairs: List of (context, answer) tuples.

    Returns:
        List of (label, score, is_hallucination) per pair.
    """
    if not pairs:
        return []

    model = checker._load()

    import numpy as np

    raw_scores = model.predict(list(pairs))
    results = []
    for scores in raw_scores:
        probs = _softmax(scores)
        label_idx = int(np.argmax(probs))
        label = _LABELS[label_idx]
        score = float(probs[label_idx])
        is_hall = label == "contradiction" and score >= _HALLUCINATION_THRESHOLD
        results.append((label, score, is_hall))
    return results
