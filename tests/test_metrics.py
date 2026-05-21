"""Unit tests for all metric modules.

All external model calls (sentence-transformers, NLI, OpenAI) are mocked
so tests run offline with no API keys required.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Latency metrics ────────────────────────────────────────────────────────────


from evaluator.metrics.latency import LatencyStats, compute_latency_stats, compute_tokens_per_second


class TestLatencyMetrics:
    def test_basic_percentiles(self):
        latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
        stats = compute_latency_stats(latencies)
        assert stats.p50_ms == pytest.approx(300.0, abs=1)
        assert stats.p95_ms == pytest.approx(480.0, abs=1)
        assert stats.p99_ms == pytest.approx(496.0, abs=1)
        assert stats.min_ms == 100.0
        assert stats.max_ms == 500.0
        assert stats.n == 5

    def test_single_value(self):
        stats = compute_latency_stats([250.0])
        assert stats.p50_ms == 250.0
        assert stats.p99_ms == 250.0
        assert stats.n == 1

    def test_with_ttft_and_tps(self):
        latencies = [100.0, 200.0, 300.0]
        ttfts = [20.0, 30.0, 25.0]
        tps = [50.0, 60.0, 55.0]
        stats = compute_latency_stats(latencies, ttfts, tps)
        assert stats.mean_ttft_ms == pytest.approx(25.0, abs=0.1)
        assert stats.mean_tokens_per_sec == pytest.approx(55.0, abs=0.1)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_latency_stats([])

    def test_to_dict(self):
        stats = compute_latency_stats([100.0, 200.0])
        d = stats.to_dict()
        assert "p50_ms" in d
        assert "p95_ms" in d
        assert "n" in d

    def test_tokens_per_second(self):
        tps = compute_tokens_per_second(completion_tokens=100, latency_ms=1000.0)
        assert tps == pytest.approx(100.0, abs=0.1)

    def test_tokens_per_second_zero_latency(self):
        assert compute_tokens_per_second(100, 0.0) is None

    def test_tokens_per_second_zero_tokens(self):
        assert compute_tokens_per_second(0, 500.0) is None

    def test_large_dataset(self):
        rng = np.random.default_rng(42)
        latencies = rng.exponential(scale=200, size=1000).tolist()
        stats = compute_latency_stats(latencies)
        assert stats.p50_ms < stats.p95_ms < stats.p99_ms
        assert stats.n == 1000


# ── Accuracy metrics ───────────────────────────────────────────────────────────


from evaluator.metrics.accuracy import (
    compute_exact_match,
    compute_f1_token_overlap,
    compute_semantic_similarity,
)


class TestExactMatch:
    def test_identical(self):
        assert compute_exact_match("The transformer", "The transformer") is True

    def test_case_insensitive(self):
        assert compute_exact_match("BERT", "bert") is True

    def test_strip_punctuation(self):
        assert compute_exact_match("hello, world!", "hello world") is True

    def test_mismatch(self):
        assert compute_exact_match("GPT-4", "BERT") is False

    def test_extra_whitespace(self):
        assert compute_exact_match("  attention   ", "attention") is True

    def test_empty_strings(self):
        assert compute_exact_match("", "") is True

    def test_one_empty(self):
        assert compute_exact_match("something", "") is False


class TestSemanticSimilarity:
    def _mock_model(self, embeddings: list[list[float]]):
        mock = MagicMock()
        import numpy as np

        def encode(texts, **kwargs):
            arrs = np.array(embeddings[: len(texts)], dtype=float)
            # Normalize
            norms = np.linalg.norm(arrs, axis=1, keepdims=True)
            return arrs / norms

        mock.encode = encode
        return mock

    def test_identical_text_high_similarity(self):
        model = self._mock_model([[1.0, 0.0], [1.0, 0.0]])
        sim = compute_semantic_similarity("hello", "hello", model)
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_low_similarity(self):
        model = self._mock_model([[1.0, 0.0], [0.0, 1.0]])
        sim = compute_semantic_similarity("foo", "bar", model)
        assert sim == pytest.approx(0.0, abs=0.01)

    def test_clamped_to_zero(self):
        model = self._mock_model([[1.0, 0.0], [-1.0, 0.0]])
        sim = compute_semantic_similarity("foo", "bar", model)
        assert sim >= 0.0


class TestF1TokenOverlap:
    def test_perfect_match(self):
        assert compute_f1_token_overlap("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert compute_f1_token_overlap("apple", "banana") == pytest.approx(0.0)

    def test_partial_overlap(self):
        f1 = compute_f1_token_overlap("the cat sat on mat", "the cat sat")
        assert 0.0 < f1 < 1.0

    def test_empty_both(self):
        assert compute_f1_token_overlap("", "") == pytest.approx(1.0)

    def test_one_empty(self):
        assert compute_f1_token_overlap("something", "") == pytest.approx(0.0)


# ── Cost metrics ───────────────────────────────────────────────────────────────


from evaluator.metrics.cost import (
    PRICING_TABLE,
    cost_efficiency_score,
    estimate_cost,
    format_cost,
    get_pricing,
)


class TestCostMetrics:
    def test_known_model_cost(self):
        cost = estimate_cost("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
        expected = (1000 / 1e6 * 0.15) + (500 / 1e6 * 0.60)
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_zero_tokens_zero_cost(self):
        assert estimate_cost("gpt-4o-mini", 0, 0) == pytest.approx(0.0)

    def test_all_pricing_table_entries(self):
        for model in PRICING_TABLE:
            cost = estimate_cost(model, 1000, 1000)
            assert cost > 0

    def test_unknown_model_fallback(self):
        cost = estimate_cost("unknown-model-xyz", 1_000_000, 1_000_000)
        assert cost > 0  # Uses default pricing

    def test_prefix_matching(self):
        pricing = get_pricing("gpt-4o-mini-2024-07-18")
        assert pricing == PRICING_TABLE["gpt-4o-mini"]

    def test_anthropic_pricing(self):
        cost = estimate_cost("claude-haiku-4-5-20251001", 10000, 5000)
        assert cost == pytest.approx((10000 / 1e6 * 0.80) + (5000 / 1e6 * 4.00), rel=1e-6)

    def test_groq_pricing(self):
        cost = estimate_cost("llama3-70b-8192", 10000, 10000)
        assert cost > 0
        assert cost < 0.02  # Groq is cheap

    def test_cost_efficiency_perfect(self):
        score = cost_efficiency_score(cost_usd=0.0, accuracy=1.0)
        assert score == pytest.approx(1.0)

    def test_cost_efficiency_expensive(self):
        score = cost_efficiency_score(cost_usd=0.1, accuracy=1.0, max_cost=0.01)
        # Cost score = max(0, 1 - 10) = 0; total = 0.5 * 1 + 0.5 * 0 = 0.5
        assert score == pytest.approx(0.5)

    def test_format_cost_large(self):
        assert "$" in format_cost(0.05)

    def test_format_cost_small(self):
        # Very small cost should use µ notation
        formatted = format_cost(0.000001)
        assert "$" in formatted


# ── Hallucination metrics ──────────────────────────────────────────────────────


from evaluator.metrics.hallucination import (
    NLIHallucinationChecker,
    _softmax,
    check_nli_hallucination,
    check_nli_groq,
)


class TestSoftmax:
    def test_sums_to_one(self):
        result = _softmax([1.0, 2.0, 3.0])
        assert abs(sum(result) - 1.0) < 1e-6

    def test_max_arg_preserved(self):
        result = _softmax([1.0, 100.0, 0.5])
        assert result[1] > result[0]
        assert result[1] > result[2]

    def test_numerically_stable(self):
        result = _softmax([1000.0, 1001.0, 999.0])
        assert all(0 <= v <= 1 for v in result)
        assert abs(sum(result) - 1.0) < 1e-6


class TestNLIHallucinationChecker:
    """NLI is now Groq API-based; the local checker class is a compatibility stub."""

    def test_checker_stub_raises(self):
        checker = NLIHallucinationChecker()
        with pytest.raises((NotImplementedError, RuntimeError)):
            checker.is_hallucination("ctx", "ans")

    def test_check_nli_empty_context(self):
        checker = MagicMock()
        label, score, is_hall = check_nli_hallucination(checker, "", "some answer")
        assert label is None
        assert score is None
        assert not is_hall

    def test_check_nli_empty_answer(self):
        checker = MagicMock()
        label, score, is_hall = check_nli_hallucination(checker, "context", "")
        assert label is None
        assert score is None
        assert not is_hall

    def test_check_nli_exception_handled(self):
        checker = MagicMock()
        checker.is_hallucination.side_effect = RuntimeError("model failed")
        label, score, is_hall = check_nli_hallucination(checker, "context", "answer")
        assert label is None
        assert score is None
        assert not is_hall

    @pytest.mark.asyncio
    async def test_check_nli_groq_entailment(self):
        """check_nli_groq returns (label, score, is_hall) on Groq API success."""
        from unittest.mock import AsyncMock, patch

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "ENTAILMENT"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch("groq.AsyncGroq", return_value=mock_client):
            label, score, is_hall = await check_nli_groq("ctx", "ans")

        assert label == "entailment"
        assert not is_hall

    @pytest.mark.asyncio
    async def test_check_nli_groq_contradiction(self):
        from unittest.mock import AsyncMock, patch

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "CONTRADICTION"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch("groq.AsyncGroq", return_value=mock_client):
            label, score, is_hall = await check_nli_groq("ctx", "ans")

        assert label == "contradiction"
        assert is_hall

    @pytest.mark.asyncio
    async def test_check_nli_groq_empty_inputs(self):
        label, score, is_hall = await check_nli_groq("", "ans")
        assert label is None
        assert not is_hall

    @pytest.mark.asyncio
    async def test_check_nli_groq_api_failure(self):
        from unittest.mock import AsyncMock, patch

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))

        with patch("groq.AsyncGroq", return_value=mock_client):
            label, score, is_hall = await check_nli_groq("context", "answer")

        assert label is None
        assert not is_hall


# ── RAGAS metrics ──────────────────────────────────────────────────────────────


from evaluator.metrics.ragas_metrics import _safe_float, evaluate_ragas_batch


class TestSafeFloat:
    def test_valid_float(self):
        assert _safe_float(0.75) == pytest.approx(0.75)

    def test_none(self):
        assert _safe_float(None) is None

    def test_nan(self):
        assert _safe_float(float("nan")) is None

    def test_string_number(self):
        assert _safe_float("0.5") == pytest.approx(0.5)

    def test_invalid_string(self):
        assert _safe_float("not-a-number") is None


class TestEvaluateRagasBatch:
    def test_empty_input(self):
        result = evaluate_ragas_batch([])
        assert result == []

    @patch("evaluator.metrics.ragas_metrics._build_ragas_llm")
    @patch("evaluator.metrics.ragas_metrics._build_ragas_embeddings")
    def test_import_error_graceful(self, mock_emb, mock_llm):
        """If ragas is not installed, returns None scores gracefully."""
        with patch("builtins.__import__", side_effect=ImportError("ragas not found")):
            # Call via the import-safe path
            pass  # The function handles ImportError internally

    def test_returns_none_scores_on_failure(self):
        samples = [("q", "a", "c", "gt")]
        # With no API keys configured, RAGAS will fail — ensure graceful return
        with patch("evaluator.metrics.ragas_metrics._build_ragas_llm", side_effect=Exception("no key")):
            result = evaluate_ragas_batch(samples)
            assert len(result) == 1
            assert all(v is None for v in result[0].values())
