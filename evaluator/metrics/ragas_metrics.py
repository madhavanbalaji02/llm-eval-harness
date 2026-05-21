"""RAGAS 0.4.x evaluation: Groq LLM + HuggingFace Inference API embeddings.

No local torch/GPU required. Uses:
  - llama-3.3-70b-versatile (Groq) as the LLM evaluator
  - all-MiniLM-L6-v2 (HF Inference API) for answer_relevancy embeddings
  - faithfulness, context_precision, context_recall are LLM-only (no embeddings)

Runs synchronously; always call from asyncio.to_thread or a worker thread.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

RAGAS_METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

_HF_EMBED_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction"
    "/sentence-transformers/all-MiniLM-L6-v2"
)


class _HFInferenceAPIEmbeddings:
    """LangChain-compatible embeddings backed by HuggingFace Inference API.

    No local model loading — uses httpx to call the free HF endpoint.
    Compatible with RAGAS LangchainEmbeddingsWrapper.
    """

    def __init__(self) -> None:
        self.hf_token = os.getenv("HF_TOKEN", "")

    def _fetch(self, texts: list[str]) -> list[list[float]]:
        import httpx

        response = httpx.post(
            _HF_EMBED_URL,
            headers={"Authorization": f"Bearer {self.hf_token}"},
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._fetch(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._fetch([text])[0]


def _build_ragas_llm() -> Any:
    """Build RAGAS InstructorLLM using Groq's OpenAI-compatible API.

    RAGAS 0.4.x collections metrics require InstructorLLM (via llm_factory),
    not LangchainLLMWrapper. Groq is OpenAI-compatible so we pass an OpenAI
    client pointed at Groq's base URL.
    """
    from openai import OpenAI
    from ragas.llms import llm_factory

    groq_client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    # Use 8b for RAGAS — higher TPD limit, sufficient quality for faithfulness/precision/recall
    return llm_factory("llama-3.1-8b-instant", provider="openai", client=groq_client)


def _build_ragas_embeddings() -> Any:
    """RAGAS 0.4 requires embedding_factory — only needed for AnswerRelevancy.
    We skip that metric and use LLM-only metrics instead."""
    return None


def evaluate_ragas_batch(
    samples: list[tuple[str, str, str, str]],
    **_kwargs,
) -> list[dict[str, Optional[float]]]:
    """Run RAGAS evaluation on (question, answer, context, ground_truth) tuples.

    Groq LLM + HuggingFace Inference API embeddings — no local models.
    """
    empty = {k: None for k in RAGAS_METRIC_KEYS}
    if not samples:
        return []

    try:
        import warnings
        from ragas import EvaluationDataset, evaluate as ragas_evaluate
        from ragas.dataset_schema import SingleTurnSample
        # Use the old-style singleton metrics (they ARE ragas.metrics.base.Metric instances).
        # The new collections classes inherit BaseMetric which evaluate() rejects.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import (
                faithfulness as _faithfulness,
                context_precision as _ctx_precision,
                context_recall as _ctx_recall,
            )

        ragas_samples = [
            SingleTurnSample(
                user_input=q,
                retrieved_contexts=[ctx] if ctx else [""],
                response=ans,
                reference=gt,
            )
            for q, ans, ctx, gt in samples
        ]
        dataset = EvaluationDataset(samples=ragas_samples)

        llm = _build_ragas_llm()
        # Set LLM on each singleton metric
        _faithfulness.llm = llm
        _ctx_precision.llm = llm
        _ctx_recall.llm = llm

        result = ragas_evaluate(
            dataset=dataset,
            metrics=[_faithfulness, _ctx_precision, _ctx_recall],
            raise_exceptions=False,
            show_progress=False,
        )

        df = result.to_pandas()
        return [
            {
                "faithfulness": _safe_float(row.get("faithfulness")),
                "answer_relevancy": _safe_float(row.get("answer_relevancy")),
                "context_precision": _safe_float(row.get("context_precision")),
                "context_recall": _safe_float(row.get("context_recall")),
            }
            for _, row in df.iterrows()
        ]

    except ImportError as exc:
        logger.warning("RAGAS not available: %s", exc)
        return [{**empty} for _ in samples]
    except Exception as exc:
        logger.warning("RAGAS evaluation failed: %s", exc)
        return [{**empty} for _ in samples]


def evaluate_ragas_single(
    question: str,
    answer: str,
    context: str,
    ground_truth: str,
) -> dict[str, Optional[float]]:
    results = evaluate_ragas_batch([(question, answer, context, ground_truth)])
    return results[0] if results else {k: None for k in RAGAS_METRIC_KEYS}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        return None if (f != f) else f
    except (TypeError, ValueError):
        return None
