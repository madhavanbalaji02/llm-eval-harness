"""Core evaluator: orchestrates runners, metrics, and aggregation."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

from .datasets.loader import DatasetItem
from .runners import BaseRunner, RunResult

logger = logging.getLogger(__name__)


class EvalResult(BaseModel):
    """Result for a single evaluation item."""

    id: str
    question: str
    model: str
    answer: str
    ground_truth: str
    context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Latency
    latency_ms: float = 0.0
    ttft_ms: Optional[float] = None
    tokens_per_sec: Optional[float] = None

    # Token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Cost
    cost_usd: float = 0.0

    # Accuracy
    exact_match: bool = False
    semantic_similarity: float = 0.0
    llm_judge_score: Optional[float] = None

    # Hallucination
    faithfulness: Optional[float] = None
    nli_label: Optional[str] = None
    nli_score: Optional[float] = None
    is_hallucination: bool = False

    # RAGAS
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None

    # Error tracking
    error: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


class AggregatedResults(BaseModel):
    """Aggregated metrics across all evaluation items."""

    model: str
    n_samples: int
    n_successful: int = 0

    # Latency (ms)
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_mean: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0
    mean_ttft_ms: Optional[float] = None
    mean_tokens_per_sec: Optional[float] = None

    # Accuracy
    exact_match_rate: float = 0.0
    mean_semantic_similarity: float = 0.0
    mean_llm_judge_score: Optional[float] = None

    # Hallucination
    hallucination_rate: float = 0.0
    mean_faithfulness: Optional[float] = None
    mean_nli_score: Optional[float] = None

    # RAGAS
    mean_answer_relevancy: Optional[float] = None
    mean_context_precision: Optional[float] = None
    mean_context_recall: Optional[float] = None

    # Cost
    total_cost_usd: float = 0.0
    mean_cost_per_query: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    # Error stats
    error_rate: float = 0.0

    results: list[EvalResult] = Field(default_factory=list)

    @classmethod
    def from_results(cls, model: str, results: list[EvalResult]) -> "AggregatedResults":
        if not results:
            return cls(model=model, n_samples=0)

        successful = [r for r in results if r.error is None]
        n = len(results)
        ns = len(successful)

        def _mean(vals: list[float]) -> Optional[float]:
            return float(np.mean(vals)) if vals else None

        def _pct(vals: list[float], p: int) -> float:
            return float(np.percentile(vals, p)) if vals else 0.0

        latencies = [r.latency_ms for r in successful]
        ttfts = [r.ttft_ms for r in successful if r.ttft_ms is not None]
        tps_list = [r.tokens_per_sec for r in successful if r.tokens_per_sec is not None]
        sem_sims = [r.semantic_similarity for r in successful]
        judge_scores = [r.llm_judge_score for r in successful if r.llm_judge_score is not None]
        faithfulness_scores = [r.faithfulness for r in successful if r.faithfulness is not None]
        nli_scores = [r.nli_score for r in successful if r.nli_score is not None]
        ar = [r.answer_relevancy for r in successful if r.answer_relevancy is not None]
        cp = [r.context_precision for r in successful if r.context_precision is not None]
        cr = [r.context_recall for r in successful if r.context_recall is not None]

        return cls(
            model=model,
            n_samples=n,
            n_successful=ns,
            latency_p50=_pct(latencies, 50),
            latency_p95=_pct(latencies, 95),
            latency_p99=_pct(latencies, 99),
            latency_mean=float(np.mean(latencies)) if latencies else 0.0,
            latency_min=float(np.min(latencies)) if latencies else 0.0,
            latency_max=float(np.max(latencies)) if latencies else 0.0,
            mean_ttft_ms=_mean(ttfts),
            mean_tokens_per_sec=_mean(tps_list),
            exact_match_rate=sum(r.exact_match for r in successful) / ns if ns else 0.0,
            mean_semantic_similarity=float(np.mean(sem_sims)) if sem_sims else 0.0,
            mean_llm_judge_score=_mean(judge_scores),
            hallucination_rate=sum(r.is_hallucination for r in successful) / ns if ns else 0.0,
            mean_faithfulness=_mean(faithfulness_scores),
            mean_nli_score=_mean(nli_scores),
            mean_answer_relevancy=_mean(ar),
            mean_context_precision=_mean(cp),
            mean_context_recall=_mean(cr),
            total_cost_usd=sum(r.cost_usd for r in successful),
            mean_cost_per_query=float(np.mean([r.cost_usd for r in successful])) if successful else 0.0,
            total_prompt_tokens=sum(r.prompt_tokens for r in successful),
            total_completion_tokens=sum(r.completion_tokens for r in successful),
            error_rate=(n - ns) / n if n else 0.0,
            results=results,
        )


class Evaluator:
    """Orchestrates LLM evaluation across a dataset.

    Runs each dataset item through the configured runner, computes all
    enabled metrics, and returns structured EvalResult objects.
    """

    def __init__(
        self,
        runner: BaseRunner,
        enable_ragas: bool = True,
        enable_nli: bool = True,
        enable_judge: bool = True,
        enable_semantic: bool = True,
        judge_model: str = "gpt-4o-mini",
        max_concurrency: int = 5,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.runner = runner
        self.enable_ragas = enable_ragas
        self.enable_nli = enable_nli
        self.enable_judge = enable_judge
        self.enable_semantic = enable_semantic
        self.judge_model = judge_model
        self.max_concurrency = max_concurrency
        self.system_prompt = system_prompt

        self._sentence_model: Any = None
        self._nli_checker: Any = None

    def _get_sentence_model(self) -> Any:
        if self._sentence_model is None and self.enable_semantic:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
            self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._sentence_model

    def _get_nli_checker(self) -> Any:
        if self._nli_checker is None and self.enable_nli:
            from .metrics.hallucination import NLIHallucinationChecker

            logger.info("Loading NLI model (cross-encoder/nli-deberta-v3-small)...")
            self._nli_checker = NLIHallucinationChecker()
        return self._nli_checker

    async def _evaluate_item(
        self,
        item: DatasetItem,
        semaphore: asyncio.Semaphore,
    ) -> EvalResult:
        async with semaphore:
            try:
                return await self._run_single(item)
            except Exception as exc:
                logger.error("Error evaluating item %s: %s", item.id, exc)
                return EvalResult(
                    id=item.id,
                    question=item.question,
                    model=self.runner.model,
                    answer="",
                    ground_truth=item.ground_truth,
                    context=item.context,
                    metadata=item.metadata,
                    error=str(exc),
                )

    async def _run_single(self, item: DatasetItem) -> EvalResult:
        from .metrics.accuracy import compute_exact_match, compute_semantic_similarity, compute_llm_judge
        from .metrics.cost import estimate_cost
        from .metrics.hallucination import check_nli_hallucination

        run: RunResult = await self.runner.run(
            prompt=item.question,
            system_prompt=self.system_prompt,
        )

        # Tokens-per-second
        tokens_per_sec: Optional[float] = None
        if run.completion_tokens > 0 and run.latency_ms > 0:
            tokens_per_sec = (run.completion_tokens / run.latency_ms) * 1000

        # Cost
        cost = estimate_cost(
            model=self.runner.model,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
        )

        # Exact match
        em = compute_exact_match(run.response, item.ground_truth)

        # Semantic similarity
        sem_sim = 0.0
        if self.enable_semantic:
            model = self._get_sentence_model()
            if model is not None:
                sem_sim = compute_semantic_similarity(run.response, item.ground_truth, model)

        # NLI hallucination
        nli_label: Optional[str] = None
        nli_score: Optional[float] = None
        is_hallucination = False
        if self.enable_nli and item.context:
            checker = self._get_nli_checker()
            if checker is not None:
                nli_label, nli_score, is_hallucination = check_nli_hallucination(
                    checker, item.context, run.response
                )

        # LLM judge
        judge_score: Optional[float] = None
        if self.enable_judge:
            try:
                judge_score = await compute_llm_judge(
                    question=item.question,
                    answer=run.response,
                    ground_truth=item.ground_truth,
                    model=self.judge_model,
                    api_key=self.runner.api_key,
                )
            except Exception as exc:
                logger.warning("LLM judge failed for %s: %s", item.id, exc)

        return EvalResult(
            id=item.id,
            question=item.question,
            model=self.runner.model,
            answer=run.response,
            ground_truth=item.ground_truth,
            context=item.context,
            metadata=item.metadata,
            latency_ms=run.latency_ms,
            ttft_ms=run.ttft_ms,
            tokens_per_sec=tokens_per_sec,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            total_tokens=run.prompt_tokens + run.completion_tokens,
            cost_usd=cost,
            exact_match=em,
            semantic_similarity=sem_sim,
            llm_judge_score=judge_score,
            nli_label=nli_label,
            nli_score=nli_score,
            is_hallucination=is_hallucination,
        )

    async def evaluate_dataset(
        self,
        items: list[DatasetItem],
        progress_callback: Optional[Any] = None,
    ) -> list[EvalResult]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._evaluate_item(item, semaphore) for item in items]

        results: list[EvalResult] = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            if progress_callback:
                progress_callback(result)

        # Sort back to original order
        id_order = {item.id: i for i, item in enumerate(items)}
        results.sort(key=lambda r: id_order.get(r.id, 0))

        # RAGAS batch evaluation
        if self.enable_ragas:
            results = await self._apply_ragas(results, items)

        return results

    async def _apply_ragas(
        self,
        results: list[EvalResult],
        items: list[DatasetItem],
    ) -> list[EvalResult]:
        from .metrics.ragas_metrics import evaluate_ragas_batch

        successful = [(r, i) for r, i in zip(results, items) if r.error is None and i.context]
        if not successful:
            return results

        try:
            ragas_scores = await asyncio.to_thread(
                evaluate_ragas_batch,
                [(r.question, r.answer, i.context, r.ground_truth) for r, i in successful],
            )
            for (result, _), scores in zip(successful, ragas_scores):
                result.answer_relevancy = scores.get("answer_relevancy")
                result.context_precision = scores.get("context_precision")
                result.context_recall = scores.get("context_recall")
                if scores.get("faithfulness") is not None:
                    result.faithfulness = scores["faithfulness"]
                    if result.faithfulness is not None and result.faithfulness < 0.5:
                        result.is_hallucination = True
        except Exception as exc:
            logger.warning("RAGAS evaluation failed: %s", exc)

        return results

    def aggregate(self, results: list[EvalResult]) -> AggregatedResults:
        return AggregatedResults.from_results(self.runner.model, results)
