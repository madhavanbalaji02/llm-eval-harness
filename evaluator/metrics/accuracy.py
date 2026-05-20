"""Accuracy metrics: exact match, semantic similarity, LLM-as-judge."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are an expert evaluator grading an AI assistant's answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Model Answer: {answer}

Rate the model's answer on a scale from 1 to 5:
1 = Completely wrong or irrelevant
2 = Partially correct but contains major errors
3 = Mostly correct with minor errors or omissions
4 = Correct with small imperfections
5 = Perfect — fully accurate, clear, and complete

Respond with ONLY a single digit (1, 2, 3, 4, or 5). Nothing else."""


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def compute_exact_match(prediction: str, reference: str) -> bool:
    """Case-insensitive, punctuation-stripped exact match.

    Args:
        prediction: The model's generated answer.
        reference: The ground-truth answer.

    Returns:
        True if normalized strings are identical.
    """
    return _normalize(prediction) == _normalize(reference)


def compute_semantic_similarity(
    prediction: str,
    reference: str,
    model: "SentenceTransformer",
) -> float:
    """Cosine similarity between sentence embeddings.

    Uses all-MiniLM-L6-v2 (384-dim) for fast, high-quality embeddings.

    Returns:
        Cosine similarity in [0, 1]. Higher is more semantically similar.
    """
    embeddings = model.encode([prediction, reference], convert_to_numpy=True, normalize_embeddings=True)
    similarity = float(np.dot(embeddings[0], embeddings[1]))
    # Clamp to [0, 1] — normalized embeddings give cosine in [-1, 1]
    return max(0.0, min(1.0, similarity))


async def compute_llm_judge(
    question: str,
    answer: str,
    ground_truth: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> float:
    """LLM-as-judge scoring on a 1–5 scale via OpenAI chat completions.

    Args:
        question: The original question.
        answer: The model's response to evaluate.
        ground_truth: The reference answer.
        model: OpenAI model to use as judge.
        api_key: OpenAI API key.

    Returns:
        Float score in [1.0, 5.0]. Returns 3.0 on parse failure.
    """
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        prompt = _JUDGE_PROMPT.format(
            question=question, ground_truth=ground_truth, answer=answer
        )
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        score = float(raw[0])
        return max(1.0, min(5.0, score))
    except Exception as exc:
        logger.warning("LLM judge parse failed (%s), defaulting to 3.0", exc)
        return 3.0


def compute_f1_token_overlap(prediction: str, reference: str) -> float:
    """Token-level F1 score between prediction and reference.

    Useful as a lightweight alternative to ROUGE for short answers.
    """
    pred_tokens = set(_normalize(prediction).split())
    ref_tokens = set(_normalize(reference).split())

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
