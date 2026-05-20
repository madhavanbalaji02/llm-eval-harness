"""Unit tests for reporter modules: console, JSON exporter, HTML report."""

from __future__ import annotations

import json
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evaluator.core import AggregatedResults, EvalResult


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_result(**overrides) -> EvalResult:
    defaults = {
        "id": "qa_001",
        "question": "What is attention?",
        "model": "gpt-4o-mini",
        "answer": "It is a mechanism that weights input tokens.",
        "ground_truth": "Attention weights input tokens by relevance.",
        "context": "The transformer uses attention to selectively focus on tokens.",
        "latency_ms": 250.0,
        "ttft_ms": 80.0,
        "tokens_per_sec": 45.0,
        "prompt_tokens": 30,
        "completion_tokens": 12,
        "total_tokens": 42,
        "cost_usd": 0.000025,
        "exact_match": False,
        "semantic_similarity": 0.82,
        "llm_judge_score": 4.0,
        "faithfulness": 0.9,
        "nli_label": "entailment",
        "nli_score": 0.91,
        "is_hallucination": False,
        "answer_relevancy": 0.88,
        "context_precision": 0.85,
        "context_recall": 0.80,
    }
    defaults.update(overrides)
    return EvalResult(**defaults)


def _make_results(n: int = 5) -> list[EvalResult]:
    results = []
    for i in range(n):
        is_hall = i == 0
        results.append(
            _make_result(
                id=f"qa_{i:03d}",
                latency_ms=100.0 + i * 50,
                semantic_similarity=0.7 + i * 0.05,
                is_hallucination=is_hall,
                nli_label="contradiction" if is_hall else "entailment",
                exact_match=(i % 2 == 0),
            )
        )
    return results


def _make_aggregated(model: str = "gpt-4o-mini", n: int = 5) -> AggregatedResults:
    return AggregatedResults.from_results(model, _make_results(n))


# ── JSON Exporter ──────────────────────────────────────────────────────────────


class TestJSONExporter:
    @pytest.fixture
    def exporter(self):
        from evaluator.reporter.json_exporter import JSONExporter

        return JSONExporter()

    def test_export_creates_file(self, exporter, tmp_path):
        results = _make_results(3)
        agg = _make_aggregated(n=3)
        out = tmp_path / "test_output.json"
        path = exporter.export(results, agg, out, {"model": "gpt-4o-mini"})
        assert path.exists()

    def test_export_valid_json(self, exporter, tmp_path):
        results = _make_results(3)
        agg = _make_aggregated(n=3)
        out = tmp_path / "results.json"
        exporter.export(results, agg, out)
        payload = json.loads(out.read_text())
        assert "schema_version" in payload
        assert "summary" in payload
        assert "results" in payload

    def test_export_summary_structure(self, exporter, tmp_path):
        results = _make_results(5)
        agg = _make_aggregated(n=5)
        out = tmp_path / "results.json"
        exporter.export(results, agg, out)
        payload = json.loads(out.read_text())
        summary = payload["summary"]
        assert "latency" in summary
        assert "accuracy" in summary
        assert "hallucination" in summary
        assert "cost" in summary
        assert "ragas" in summary

    def test_export_results_count(self, exporter, tmp_path):
        results = _make_results(7)
        agg = _make_aggregated(n=7)
        out = tmp_path / "results.json"
        exporter.export(results, agg, out)
        payload = json.loads(out.read_text())
        assert len(payload["results"]) == 7

    def test_per_result_fields(self, exporter, tmp_path):
        results = _make_results(1)
        agg = _make_aggregated(n=1)
        out = tmp_path / "single.json"
        exporter.export(results, agg, out)
        payload = json.loads(out.read_text())
        r = payload["results"][0]
        assert "id" in r
        assert "question" in r
        assert "answer" in r
        assert "accuracy" in r
        assert "hallucination" in r
        assert "latency" in r
        assert "tokens" in r
        assert "cost_usd" in r

    def test_creates_parent_directory(self, exporter, tmp_path):
        results = _make_results(1)
        agg = _make_aggregated(n=1)
        nested = tmp_path / "nested" / "deep" / "out.json"
        exporter.export(results, agg, nested)
        assert nested.exists()

    def test_error_result_serialized(self, exporter, tmp_path):
        r = _make_result(error="API timeout", answer="")
        agg = AggregatedResults.from_results("gpt-4o-mini", [r])
        out = tmp_path / "err.json"
        exporter.export([r], agg, out)
        payload = json.loads(out.read_text())
        assert payload["results"][0]["error"] == "API timeout"

    def test_load_results_json(self, exporter, tmp_path):
        from evaluator.reporter.json_exporter import load_results_json

        results = _make_results(2)
        agg = _make_aggregated(n=2)
        out = tmp_path / "r.json"
        exporter.export(results, agg, out)
        loaded = load_results_json(out)
        assert loaded["summary"]["model"] == "gpt-4o-mini"


# ── Console Reporter ───────────────────────────────────────────────────────────


class TestConsoleReporter:
    @pytest.fixture
    def reporter(self):
        from rich.console import Console

        from evaluator.reporter.console import ConsoleReporter

        buf = StringIO()
        console = Console(file=buf, width=120)
        return ConsoleReporter(console=console), buf

    def test_print_result_ok(self, reporter):
        rep, buf = reporter
        r = _make_result()
        rep.print_result(r)
        output = buf.getvalue()
        assert "qa_001" in output

    def test_print_result_with_error(self, reporter):
        rep, buf = reporter
        r = _make_result(error="timeout", answer="")
        rep.print_result(r)
        output = buf.getvalue()
        assert "qa_001" in output

    def test_print_summary_contains_model(self, reporter):
        rep, buf = reporter
        agg = _make_aggregated("gpt-4o-mini")
        rep.print_summary(agg)
        output = buf.getvalue()
        assert "gpt-4o-mini" in output

    def test_print_summary_contains_latency(self, reporter):
        rep, buf = reporter
        agg = _make_aggregated()
        rep.print_summary(agg)
        output = buf.getvalue()
        assert "Latency" in output or "ms" in output

    def test_print_summary_contains_cost(self, reporter):
        rep, buf = reporter
        agg = _make_aggregated()
        rep.print_summary(agg)
        output = buf.getvalue()
        assert "$" in output or "Cost" in output

    def test_print_comparison_multiple_models(self, reporter):
        rep, buf = reporter
        agg1 = _make_aggregated("gpt-4o-mini")
        agg2 = _make_aggregated("claude-haiku-4-5-20251001")
        rep.print_comparison([agg1, agg2])
        output = buf.getvalue()
        assert "gpt-4o-mini" in output
        assert "claude-haiku" in output

    def test_print_comparison_empty(self, reporter):
        rep, buf = reporter
        rep.print_comparison([])  # Should not raise


# ── HTML Reporter ──────────────────────────────────────────────────────────────


class TestHTMLReporter:
    @pytest.fixture
    def reporter(self):
        from evaluator.reporter.html_report import HTMLReporter

        return HTMLReporter()

    @pytest.fixture
    def payload(self):
        from evaluator.reporter.json_exporter import JSONExporter

        results = _make_results(5)
        agg = _make_aggregated(n=5)
        exporter = JSONExporter()
        return exporter._build_payload(results, agg, {"model": "gpt-4o-mini"})

    def test_generate_creates_file(self, reporter, payload, tmp_path):
        out = tmp_path / "report.html"
        reporter.generate(payload, out)
        assert out.exists()

    def test_html_is_valid(self, reporter, payload, tmp_path):
        out = tmp_path / "report.html"
        reporter.generate(payload, out)
        content = out.read_text()
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "</html>" in content

    def test_html_contains_model_name(self, reporter, payload, tmp_path):
        out = tmp_path / "report.html"
        reporter.generate(payload, out)
        content = out.read_text()
        assert "gpt-4o-mini" in content

    def test_html_contains_plotly_script(self, reporter, payload, tmp_path):
        out = tmp_path / "report.html"
        reporter.generate(payload, out)
        content = out.read_text()
        assert "plotly" in content.lower()

    def test_html_custom_title(self, reporter, payload, tmp_path):
        out = tmp_path / "report.html"
        reporter.generate(payload, out, title="My Custom Report")
        content = out.read_text()
        assert "My Custom Report" in content

    def test_results_table_included(self, reporter, payload, tmp_path):
        out = tmp_path / "report.html"
        reporter.generate(payload, out)
        content = out.read_text()
        assert "results-table" in content or "<table" in content

    def test_empty_results_no_crash(self, reporter, tmp_path):
        from evaluator.reporter.json_exporter import JSONExporter

        results = []
        agg = AggregatedResults.from_results("test-model", [])
        exporter = JSONExporter()
        payload = exporter._build_payload(results, agg, {})
        out = tmp_path / "empty.html"
        reporter.generate(payload, out)
        assert out.exists()


# ── AggregatedResults ──────────────────────────────────────────────────────────


class TestAggregatedResults:
    def test_from_empty_results(self):
        agg = AggregatedResults.from_results("model", [])
        assert agg.n_samples == 0
        assert agg.exact_match_rate == 0.0

    def test_from_successful_results(self):
        results = _make_results(10)
        agg = AggregatedResults.from_results("gpt-4o-mini", results)
        assert agg.n_samples == 10
        assert agg.n_successful == 10
        assert 0 <= agg.exact_match_rate <= 1
        assert agg.latency_p50 <= agg.latency_p95 <= agg.latency_p99

    def test_error_rate_computed(self):
        results = _make_results(4)
        results.append(_make_result(id="err", error="timeout", answer=""))
        agg = AggregatedResults.from_results("model", results)
        assert agg.n_samples == 5
        assert agg.n_successful == 4
        assert agg.error_rate == pytest.approx(0.2)

    def test_hallucination_rate(self):
        results = _make_results(10)
        # First result is hallucination by fixture
        agg = AggregatedResults.from_results("model", results)
        assert agg.hallucination_rate == pytest.approx(0.1)

    def test_total_cost_sum(self):
        results = _make_results(5)
        expected_total = sum(r.cost_usd for r in results)
        agg = AggregatedResults.from_results("model", results)
        assert agg.total_cost_usd == pytest.approx(expected_total, rel=1e-5)
