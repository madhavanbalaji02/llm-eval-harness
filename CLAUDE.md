# LLM Evaluation Harness — Claude Code Guide

## Project overview

Production-grade Python framework for benchmarking LLMs on latency, accuracy, hallucination, and cost. Supports OpenAI, Anthropic, and Groq. Built as an AI engineer portfolio project.

## Commands

```bash
# Run all tests (offline — no API keys needed)
python3 -m pytest tests/ -v

# Run with coverage
python3 -m pytest tests/ --cov=evaluator --cov-report=term-missing

# Install in dev mode
pip install -e ".[dev]"

# CLI evaluation
python3 scripts/run_eval.py --model gpt-4o-mini --dataset evaluator/datasets/sample_qa.jsonl --runs 3 --report html

# Streamlit dashboard
streamlit run dashboard/app.py

# Lint
ruff check evaluator/ scripts/ tests/
```

## Architecture

```
evaluator/
  core.py              # EvalResult, AggregatedResults (Pydantic v2), Evaluator class
  metrics/
    latency.py         # p50/p95/p99, TTFT, tokens/sec
    accuracy.py        # exact match, semantic similarity, LLM-as-judge (async)
    hallucination.py   # NLI via cross-encoder/nli-deberta-v3-small
    cost.py            # PRICING_TABLE + estimate_cost()
    ragas_metrics.py   # faithfulness, answer_relevancy, context_precision, context_recall
  runners/
    __init__.py        # BaseRunner (ABC), RunResult (Pydantic v2)
    openai_runner.py   # OpenAI async streaming runner
    anthropic_runner.py # Anthropic async streaming runner
    groq_runner.py     # Groq async streaming runner
  datasets/
    loader.py          # load_dataset() → list[DatasetItem]
    sample_qa.jsonl    # 20 AI/ML Q&A pairs (question, ground_truth, context, metadata)
  reporter/
    console.py         # Rich terminal output
    html_report.py     # Self-contained HTML + Plotly charts
    json_exporter.py   # Machine-readable results JSON
dashboard/app.py       # Streamlit multi-run comparison dashboard
scripts/run_eval.py    # CLI entrypoint
tests/
  test_metrics.py      # 53 tests — all metric functions
  test_runners.py      # 21 tests — mocked async API calls
  test_reporter.py     # 21 tests — console, HTML, JSON reporters
```

## Key design decisions

- **Pydantic v2** throughout — use `model_config` dict, not inner `Config` class
- **Async-first** — `Evaluator.evaluate_dataset` uses `asyncio.as_completed` with a semaphore
- **Lazy model loading** — NLI cross-encoder and sentence-transformer loaded on first use only
- **RAGAS runs in a thread** — it's synchronous; wrap with `asyncio.to_thread`
- **Graceful degradation** — RAGAS, NLI, and judge all have `try/except` wrappers; missing API keys produce `None` scores, not crashes
- **Rich markup escaping** — when embedding item IDs in console output, use `\[{id}]` not `[{id}]` (Rich parses brackets as style tags)

## Adding a new model runner

1. Create `evaluator/runners/myprovider_runner.py` implementing `BaseRunner.run()` with streaming TTFT
2. Register in `evaluator/runners/__init__.py` `__all__`
3. Add prefix match in `scripts/run_eval.py` `resolve_runner()`
4. Add pricing entry in `evaluator/metrics/cost.py` `PRICING_TABLE`

## Adding a new metric

1. Add function to the appropriate `evaluator/metrics/*.py`
2. Add field to `EvalResult` in `evaluator/core.py` (Optional, default None)
3. Call it inside `Evaluator._run_single()`
4. Aggregate in `AggregatedResults.from_results()`
5. Display in `ConsoleReporter.print_summary()` and HTML report
6. Test in `tests/test_metrics.py`

## Dataset format (JSONL)

```json
{"id": "qa_001", "question": "...", "ground_truth": "...", "context": "...", "metadata": {"topic": "transformer", "difficulty": "medium"}}
```

All fields except `metadata` are required. `context` is used for RAGAS and NLI hallucination checks.

## Test patterns

- All LLM API calls mocked — tests run fully offline
- Use `AsyncContextManagerMock` from `test_runners.py` for async streaming mocks
- NLI model mocked via `MagicMock` injected into `checker._model`
- Sentence-transformer mocked by returning pre-computed normalized embeddings

## Environment variables

See `.env.example`. Copy to `.env` before running live evaluations.
Required: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GROQ_API_KEY` (at least one).
