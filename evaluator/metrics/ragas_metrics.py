"""RAGAS-based metrics: faithfulness, answer_relevancy, context_precision, context_recall.

Wraps the RAGAS library evaluation pipeline, configured with either
OpenAI or Anthropic as the backend LLM.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Metric keys returned per sample
RAGAS_METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _build_ragas_llm(provider: str = "openai", model: Optional[str] = None) -> Any:
    """Build a LangChain LLM wrapper for RAGAS."""
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model or "claude-haiku-4-5-20251001",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or "gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _build_ragas_embeddings(provider: str = "openai") -> Any:
    """Build embeddings for RAGAS context metrics."""
    if provider == "anthropic":
        # RAGAS doesn't support Anthropic embeddings natively; fall back to OpenAI
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))


def evaluate_ragas_batch(
    samples: list[tuple[str, str, str, str]],
    provider: str = "openai",
    llm_model: Optional[str] = None,
) -> list[dict[str, Optional[float]]]:
    """Run RAGAS evaluation on a batch of (question, answer, context, ground_truth) tuples.

    Args:
        samples: List of (question, answer, context, ground_truth).
        provider: LLM backend — 'openai' or 'anthropic'.
        llm_model: Optional model override.

    Returns:
        List of dicts with RAGAS metric scores per sample.
        Missing values are None.
    """
    empty = {k: None for k in RAGAS_METRIC_KEYS}

    if not samples:
        return []

    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        questions, answers, contexts, ground_truths = zip(*samples)
        data = {
            "question": list(questions),
            "answer": list(answers),
            "contexts": [[ctx] for ctx in contexts],
            "ground_truth": list(ground_truths),
        }
        dataset = Dataset.from_dict(data)
        llm = _build_ragas_llm(provider, llm_model)
        embeddings = _build_ragas_embeddings(provider)

        result = ragas_evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )

        df = result.to_pandas()
        output = []
        for _, row in df.iterrows():
            output.append(
                {
                    "faithfulness": _safe_float(row.get("faithfulness")),
                    "answer_relevancy": _safe_float(row.get("answer_relevancy")),
                    "context_precision": _safe_float(row.get("context_precision")),
                    "context_recall": _safe_float(row.get("context_recall")),
                }
            )
        return output

    except ImportError as exc:
        logger.warning("RAGAS not available: %s — skipping RAGAS metrics", exc)
        return [{**empty} for _ in samples]
    except Exception as exc:
        logger.warning("RAGAS evaluation failed: %s — returning None scores", exc)
        return [{**empty} for _ in samples]


def evaluate_ragas_single(
    question: str,
    answer: str,
    context: str,
    ground_truth: str,
    provider: str = "openai",
    llm_model: Optional[str] = None,
) -> dict[str, Optional[float]]:
    """Evaluate RAGAS metrics for a single (question, answer, context, ground_truth)."""
    results = evaluate_ragas_batch(
        [(question, answer, context, ground_truth)],
        provider=provider,
        llm_model=llm_model,
    )
    return results[0] if results else {k: None for k in RAGAS_METRIC_KEYS}


def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    try:
        if value is None:
            return None
        f = float(value)
        return None if (f != f) else f  # NaN check
    except (TypeError, ValueError):
        return None
