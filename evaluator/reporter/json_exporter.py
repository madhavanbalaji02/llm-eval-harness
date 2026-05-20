"""Machine-readable JSON export of evaluation results."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..core import AggregatedResults, EvalResult

logger = logging.getLogger(__name__)


class JSONExporter:
    """Exports evaluation results to a structured JSON file."""

    def export(
        self,
        results: list[EvalResult],
        aggregated: AggregatedResults,
        output_path: str | Path,
        run_metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Write results + aggregated metrics to a JSON file.

        Args:
            results: Per-item evaluation results.
            aggregated: Aggregated statistics.
            output_path: Destination file path (created if missing).
            run_metadata: Optional extra metadata to include (dataset path, run config, etc.).

        Returns:
            Resolved path of the written file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = self._build_payload(results, aggregated, run_metadata or {})

        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=_json_default)

        logger.info("Results exported to %s", path)
        return path

    def _build_payload(
        self,
        results: list[EvalResult],
        aggregated: AggregatedResults,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
            "summary": {
                "model": aggregated.model,
                "n_samples": aggregated.n_samples,
                "n_successful": aggregated.n_successful,
                "error_rate": aggregated.error_rate,
                "latency": {
                    "p50_ms": aggregated.latency_p50,
                    "p95_ms": aggregated.latency_p95,
                    "p99_ms": aggregated.latency_p99,
                    "mean_ms": aggregated.latency_mean,
                    "min_ms": aggregated.latency_min,
                    "max_ms": aggregated.latency_max,
                    "mean_ttft_ms": aggregated.mean_ttft_ms,
                    "mean_tokens_per_sec": aggregated.mean_tokens_per_sec,
                },
                "accuracy": {
                    "exact_match_rate": aggregated.exact_match_rate,
                    "mean_semantic_similarity": aggregated.mean_semantic_similarity,
                    "mean_llm_judge_score": aggregated.mean_llm_judge_score,
                },
                "hallucination": {
                    "rate": aggregated.hallucination_rate,
                    "mean_faithfulness": aggregated.mean_faithfulness,
                    "mean_nli_score": aggregated.mean_nli_score,
                },
                "ragas": {
                    "mean_answer_relevancy": aggregated.mean_answer_relevancy,
                    "mean_context_precision": aggregated.mean_context_precision,
                    "mean_context_recall": aggregated.mean_context_recall,
                },
                "cost": {
                    "total_usd": aggregated.total_cost_usd,
                    "mean_per_query_usd": aggregated.mean_cost_per_query,
                    "total_prompt_tokens": aggregated.total_prompt_tokens,
                    "total_completion_tokens": aggregated.total_completion_tokens,
                },
            },
            "results": [_result_to_dict(r) for r in results],
        }


def _result_to_dict(r: EvalResult) -> dict[str, Any]:
    return {
        "id": r.id,
        "model": r.model,
        "question": r.question,
        "answer": r.answer,
        "ground_truth": r.ground_truth,
        "context": r.context,
        "metadata": r.metadata,
        "error": r.error,
        "latency": {
            "ms": r.latency_ms,
            "ttft_ms": r.ttft_ms,
            "tokens_per_sec": r.tokens_per_sec,
        },
        "tokens": {
            "prompt": r.prompt_tokens,
            "completion": r.completion_tokens,
            "total": r.total_tokens,
        },
        "cost_usd": r.cost_usd,
        "accuracy": {
            "exact_match": r.exact_match,
            "semantic_similarity": r.semantic_similarity,
            "llm_judge_score": r.llm_judge_score,
        },
        "hallucination": {
            "is_hallucination": r.is_hallucination,
            "nli_label": r.nli_label,
            "nli_score": r.nli_score,
            "faithfulness": r.faithfulness,
        },
        "ragas": {
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        },
    }


def _json_default(obj: Any) -> Any:
    """JSON serializer for types not handled by the standard library."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_results_json(path: str | Path) -> dict[str, Any]:
    """Load a previously exported results JSON file."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
