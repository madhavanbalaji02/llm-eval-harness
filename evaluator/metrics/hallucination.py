"""Hallucination detection via Groq LLM zero-shot NLI.

Uses llama-3.1-8b-instant on Groq to classify each (context, answer) pair as
ENTAILMENT, CONTRADICTION, or NEUTRAL — equivalent to NLI but without loading
a local cross-encoder model (which deadlocks on macOS due to MPS mutex issues).

The NLIHallucinationChecker class is retained for unit-test compatibility; the
production path uses check_nli_groq() which is fully async and has no local
GPU dependencies.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LABELS = ["contradiction", "entailment", "neutral"]
_HALLUCINATION_THRESHOLD = 0.5

_NLI_PROMPT = """\
Classify the relationship between the context and the answer.

Context: {context}

Answer: {answer}

Does the answer ENTAIL the context (consistent/supported), CONTRADICT it \
(factually inconsistent), or is it NEUTRAL (unrelated)?

Respond with EXACTLY one word: ENTAILMENT, CONTRADICTION, or NEUTRAL."""


async def check_nli_groq(
    context: str,
    answer: str,
    model: str = "llama-3.1-8b-instant",
) -> tuple[Optional[str], Optional[float], bool]:
    """LLM-based NLI classification via Groq (no local model required).

    Uses llama-3.1-8b-instant for low cost and fast response.

    Returns:
        (nli_label, confidence, is_hallucination)
    """
    if not context or not answer:
        return None, None, False
    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        prompt = _NLI_PROMPT.format(
            context=context[:800],
            answer=answer[:400],
        )
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip().upper()

        if "CONTRADICTION" in raw:
            return "contradiction", 0.85, True
        elif "ENTAILMENT" in raw:
            return "entailment", 0.90, False
        else:
            return "neutral", 0.75, False

    except Exception as exc:
        logger.warning("NLI Groq check failed: %s", exc)
        return None, None, False


# ── Sync wrapper kept for legacy / unit-test call sites ──────────────────────


def check_nli_hallucination(
    checker: "NLIHallucinationChecker",
    context: str,
    answer: str,
) -> tuple[Optional[str], Optional[float], bool]:
    """Synchronous wrapper used in tests (checker is mocked)."""
    if not context or not answer:
        return None, None, False
    try:
        is_hall, label, score = checker.is_hallucination(context, answer)
        return label, score, is_hall
    except Exception as exc:
        logger.warning("NLI check failed: %s", exc)
        return None, None, False


class NLIHallucinationChecker:
    """Stub retained for unit-test compatibility.

    In production, check_nli_groq() is used instead.
    """

    MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or self.MODEL_NAME
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            raise RuntimeError(
                "Local NLI model loading is disabled on macOS (MPS deadlock). "
                "Use check_nli_groq() for async NLI inference via Groq API."
            )
        return self._model

    def predict(self, premise: str, hypothesis: str) -> tuple[str, float]:
        raise NotImplementedError("Use check_nli_groq() instead.")

    def is_hallucination(self, context: str, answer: str) -> tuple[bool, str, float]:
        raise NotImplementedError("Use check_nli_groq() instead.")


def _softmax(logits) -> list[float]:
    import numpy as np
    arr = np.array(logits, dtype=float)
    arr -= arr.max()
    exp_arr = np.exp(arr)
    return (exp_arr / exp_arr.sum()).tolist()
