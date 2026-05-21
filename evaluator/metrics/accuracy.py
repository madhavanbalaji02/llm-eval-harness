"""Accuracy metrics: exact match, semantic similarity (HF Inference API), LLM-as-judge."""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_HF_EMBED_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction"
    "/sentence-transformers/all-MiniLM-L6-v2"
)

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
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def compute_exact_match(prediction: str, reference: str) -> bool:
    """Case-insensitive, punctuation-stripped exact match."""
    return _normalize(prediction) == _normalize(reference)


# ── Async API-based semantic similarity (no local torch/MPS) ──────────────────


async def compute_semantic_similarity_api(
    prediction: str,
    reference: str,
    hf_token: Optional[str] = None,
) -> float:
    """Semantic similarity via LLM scoring (Groq) with TF-IDF cosine fallback.

    Primary: asks llama-3.1-8b-instant to rate similarity on 0.0–1.0.
    Fallback: sklearn TF-IDF cosine — reliable, no network/GPU required.

    The LLM approach captures paraphrase and synonym relationships that
    pure lexical methods miss, equivalent to dense embedding similarity.
    """
    # Try LLM-based semantic scoring via Groq
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=groq_key)
            prompt = (
                "Rate the semantic similarity between these two texts on a scale from 0.0 to 1.0.\n"
                "1.0 = identical meaning, 0.0 = completely unrelated.\n"
                "Consider paraphrases and synonyms as high similarity.\n\n"
                f"Text A: {prediction[:400]}\n"
                f"Text B: {reference[:400]}\n\n"
                "Respond with ONLY a float between 0.0 and 1.0. Nothing else."
            )
            resp = await client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            score = float(raw.split()[0])
            return max(0.0, min(1.0, score))
        except Exception as exc:
            logger.warning("LLM semantic scoring failed: %s — using TF-IDF fallback", exc)

    # Fallback: TF-IDF cosine similarity (pure Python, no GPU)
    return _tfidf_cosine(prediction, reference)


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity — runs anywhere, no model required."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer().fit_transform([text_a, text_b])
        return float(cosine_similarity(vec[0:1], vec[1:2])[0][0])
    except Exception:
        # Last-resort: token overlap
        a_tokens = set(_normalize(text_a).split())
        b_tokens = set(_normalize(text_b).split())
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


# ── Sync model-based version (kept for unit tests that pass a mock model) ────


def compute_semantic_similarity(
    prediction: str,
    reference: str,
    model: "SentenceTransformer",
) -> float:
    """Cosine similarity using a pre-loaded SentenceTransformer (test usage)."""
    embeddings = model.encode(
        [prediction, reference], convert_to_numpy=True, normalize_embeddings=True
    )
    return max(0.0, min(1.0, float(np.dot(embeddings[0], embeddings[1]))))


# ── LLM judge ────────────────────────────────────────────────────────────────


async def compute_llm_judge(
    question: str,
    answer: str,
    ground_truth: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> float:
    """LLM-as-judge scoring on a 1–5 scale.

    Uses OpenAI if OPENAI_API_KEY is set; otherwise falls back to Groq
    (llama-3.3-70b-versatile) which is also a strong judge model.
    """
    prompt = _JUDGE_PROMPT.format(
        question=question, ground_truth=ground_truth, answer=answer
    )

    openai_key = os.getenv("OPENAI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")

    # Try OpenAI if a real key is configured
    if openai_key and not openai_key.startswith("sk-..."):
        try:
            from openai import AsyncOpenAI
            response = await AsyncOpenAI(api_key=openai_key).chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            return max(1.0, min(5.0, float(raw[0])))
        except Exception as exc:
            logger.warning("OpenAI judge failed: %s — trying Groq", exc)

    # Fallback: Groq judge (8b has higher TPM limits than 70b)
    if groq_key:
        try:
            from groq import AsyncGroq
            response = await AsyncGroq(api_key=groq_key).chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            return max(1.0, min(5.0, float(raw[0])))
        except Exception as exc:
            logger.warning("Groq judge failed: %s — defaulting to 3.0", exc)

    return 3.0


def compute_f1_token_overlap(prediction: str, reference: str) -> float:
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
