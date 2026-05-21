# LLM Evaluation Harness — Claude Code Guide

## Project overview

Production-grade Python framework for benchmarking LLMs on latency, accuracy, hallucination, and cost. Supports OpenAI, Anthropic, and Groq. Deployed on HuggingFace Spaces. Built as an AI engineer portfolio project.

## Commands

```bash
# Run all tests (offline — no API keys needed)
python3 -m pytest tests/ -v                          # 97 tests

# Full eval — all metrics, no flags needed
python3 scripts/run_eval.py --model llama-3.3-70b-versatile \
  --max-items 10 --output results/run.json --report console --concurrency 1

# Fast eval — latency + cost only
python3 scripts/run_eval.py --model llama-3.1-8b-instant \
  --max-items 10 --no-ragas --no-nli --no-judge --no-semantic

# Streamlit dashboard (local)
streamlit run dashboard/app.py

# HuggingFace Spaces deployment
streamlit run spaces_app.py

# Lint
ruff check evaluator/ scripts/ tests/
```

## Architecture

```
evaluator/
  core.py                # EvalResult, AggregatedResults (Pydantic v2), Evaluator
  metrics/
    accuracy.py          # exact match; LLM-scored semantic sim (Groq); LLM judge (Groq/OpenAI)
    hallucination.py     # NLI via Groq zero-shot classification (check_nli_groq)
    cost.py              # PRICING_TABLE + estimate_cost()
    latency.py           # p50/p95/p99, TTFT, tokens/sec
    ragas_metrics.py     # faithfulness, context_precision, context_recall via RAGAS 0.4 + Groq
    _hf_worker.py        # legacy ProcessPoolExecutor worker (unused in prod; kept for reference)
  runners/
    __init__.py          # BaseRunner (ABC), RunResult (Pydantic v2)
    openai_runner.py     # OpenAI async streaming runner
    anthropic_runner.py  # Anthropic async streaming runner
    groq_runner.py       # Groq async streaming runner
  datasets/
    loader.py            # load_dataset() → list[DatasetItem]
    sample_qa.jsonl      # 20 AI/ML Q&A pairs (question, ground_truth, context, metadata)
  reporter/
    console.py           # Rich terminal output
    html_report.py       # Self-contained HTML + Plotly charts
    json_exporter.py     # Machine-readable results JSON
dashboard/app.py         # Streamlit multi-run comparison dashboard (upload JSONs)
spaces_app.py            # HuggingFace Spaces version (pre-loads results/ JSONs)
scripts/run_eval.py      # CLI entrypoint
tests/
  test_metrics.py        # 57 tests — all metric functions including Groq NLI mocks
  test_runners.py        # 21 tests — mocked async API calls
  test_reporter.py       # 21 tests — console, HTML, JSON reporters
```

## Critical: macOS MPS deadlock — what works and what doesn't

`import torch` deadlocks on macOS inside Claude Code's subprocess environment due to
Metal Performance Shaders (MPS) mutex contention. This affects ALL local HuggingFace
model loading (sentence-transformers, cross-encoders, RAGAS HF embeddings).

**What was tried and failed:**
- `TOKENIZERS_PARALLELISM=false` + `OMP_NUM_THREADS=1` — MPS mutex still blocks
- `SentenceTransformer(device="cpu")` — deadlocks before model loading
- `ThreadPoolExecutor(max_workers=1)` — threads share inherited MPS state
- `ProcessPoolExecutor(spawn)` — spawn still hits Metal framework mutex on macOS

**What works — API-based inference:**
- Semantic similarity: Groq `llama-3.1-8b-instant` scores 0–1 via prompt
- NLI hallucination: Groq `llama-3.1-8b-instant` zero-shot (ENTAILMENT/CONTRADICTION/NEUTRAL)
- RAGAS: Groq via OpenAI-compat client (`llm_factory` from ragas.llms)
- No local model downloads, no GPU/MPS, works everywhere

**Note:** When running from a real terminal (not Claude Code harness), local models work fine.

## RAGAS 0.4.x quirks

- Import from `ragas.metrics` (deprecated singletons) NOT `ragas.metrics.collections` (new classes)
  - Singletons ARE `Metric` instances → `evaluate()` accepts them
  - New collection classes inherit `BaseMetric` NOT `Metric` → `evaluate()` rejects them
- Set LLM on singletons: `faithfulness.llm = llm_factory("model", provider="openai", client=groq_client)`
- `answer_relevancy` needs embedding API (skip if no OpenAI key)
- RAGAS evaluate() runs sync internally → call via `asyncio.to_thread`

## Groq rate limits (free tier)

- llama-3.3-70b-versatile: 100k TPD (tokens per day) — exhausted quickly with RAGAS + judge
- llama-3.1-8b-instant: higher TPD — use for RAGAS LLM, NLI, and judge to preserve 70b quota
- Run evals at `--concurrency 1` to avoid TPM rate limits

## Key design decisions

- **All quality metrics are API-based** — Groq for semantic/NLI/judge, RAGAS via Groq
- **Pydantic v2** throughout — `model_config` dict, not inner `Config` class
- **Async-first** — `Evaluator.evaluate_dataset` uses `asyncio.as_completed` with semaphore
- **RAGAS in thread** — synchronous internally; wrap with `asyncio.to_thread`
- **Graceful degradation** — all metrics wrapped in try/except; None scores on failure
- **Rich markup escaping** — use `\[{id}]` not `[{id}]` in console output (Rich parses brackets)

## NLI test mocking

NLI now uses Groq API (`check_nli_groq`). To mock in tests:
```python
with patch("groq.AsyncGroq", return_value=mock_client):
    label, score, is_hall = await check_nli_groq("context", "answer")
```
NOT `patch("evaluator.metrics.hallucination.AsyncGroq")` — import is inside function.

## Environment variables

```
GROQ_API_KEY=gsk_...         # Required for all quality metrics (semantic, NLI, judge, RAGAS)
ANTHROPIC_API_KEY=sk-ant-... # Required for Claude models
OPENAI_API_KEY=sk-...        # Optional (judge falls back to Groq; RAGAS works without it)
HF_TOKEN=hf_...              # HuggingFace token (for HF Spaces deployment)
```

## Deployed result files (committed to repo)

```
results/groq_70b_full.json   # llama-3.3-70b: sem=0.880, judge=4.00, faith=0.808
results/groq_8b_full.json    # llama-3.1-8b:  sem=0.852, judge=4.00, faith=0.825
results/claude_haiku_full.json  # claude-haiku: sem=0.857, judge=4.10, faith=0.531 (!)
results/claude_opus_full.json   # claude-opus:  TBD
```

HuggingFace Space: https://madhavan02-llm-eval-harness.hf.space
GitHub: https://github.com/madhavanbalaji02/llm-eval-harness
