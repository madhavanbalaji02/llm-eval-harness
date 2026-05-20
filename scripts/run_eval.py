#!/usr/bin/env python3
"""CLI entrypoint for the LLM Evaluation Harness.

Usage:
    python scripts/run_eval.py --model gpt-4o-mini --dataset evaluator/datasets/sample_qa.jsonl
    python scripts/run_eval.py --model claude-haiku-4-5-20251001 --runs 3 --report html
    python scripts/run_eval.py --model llama3-70b-8192 --no-ragas --no-nli --output results/fast.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

load_dotenv()

console = Console()
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM Evaluation Harness — benchmark language models on Q&A datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        required=True,
        help="Model identifier: gpt-4o, gpt-4o-mini, claude-haiku-4-5-20251001, llama3-70b-8192, …",
    )
    p.add_argument(
        "--dataset",
        default="evaluator/datasets/sample_qa.jsonl",
        help="Path to JSONL evaluation dataset (default: sample_qa.jsonl)",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of evaluation passes per item (for latency averaging). Default: 1",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Default: results/<model>_<timestamp>.json",
    )
    p.add_argument(
        "--report",
        choices=["none", "console", "html", "both"],
        default="console",
        help="Report format(s) to generate. Default: console",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit evaluation to first N items (useful for quick tests)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent API requests. Default: 5",
    )
    p.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to prepend to all queries",
    )
    p.add_argument("--no-ragas", action="store_true", help="Skip RAGAS evaluation (faster)")
    p.add_argument("--no-nli", action="store_true", help="Skip NLI hallucination check (faster)")
    p.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge scoring (saves cost)")
    p.add_argument("--no-semantic", action="store_true", help="Skip sentence-transformer similarity")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return p


def resolve_runner(model: str, api_key: str | None = None):
    from evaluator.runners import AnthropicRunner, GroqRunner, OpenAIRunner

    model_lower = model.lower()
    if any(model_lower.startswith(p) for p in ("gpt-", "o1", "o3", "text-")):
        return OpenAIRunner(model=model, api_key=api_key or os.getenv("OPENAI_API_KEY"))
    if any(model_lower.startswith(p) for p in ("claude-",)):
        return AnthropicRunner(model=model, api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    if any(
        model_lower.startswith(p)
        for p in ("llama", "mixtral", "gemma", "deepseek", "qwen")
    ):
        return GroqRunner(model=model, api_key=api_key or os.getenv("GROQ_API_KEY"))

    # Default to OpenAI-compatible
    console.print(f"[yellow]⚠ Unknown model prefix for '{model}', defaulting to OpenAI runner[/yellow]")
    return OpenAIRunner(model=model, api_key=api_key or os.getenv("OPENAI_API_KEY"))


def build_output_path(model: str, output: str | None) -> Path:
    if output:
        return Path(output)
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("/", "_").replace(".", "-")
    return Path("results") / f"{model_slug}_{ts}.json"


async def run_evaluation(args: argparse.Namespace) -> None:
    from evaluator.core import Evaluator
    from evaluator.datasets.loader import load_dataset
    from evaluator.reporter.console import ConsoleReporter
    from evaluator.reporter.html_report import HTMLReporter
    from evaluator.reporter.json_exporter import JSONExporter

    # Load dataset
    console.print(f"[cyan]Loading dataset:[/cyan] {args.dataset}")
    try:
        items = load_dataset(args.dataset, max_items=args.max_items)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Dataset error: {exc}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Loaded[/cyan] {len(items)} items")

    # Build runner
    runner = resolve_runner(args.model)
    console.print(f"[cyan]Runner:[/cyan] {type(runner).__name__} → {args.model}")

    # Build evaluator
    evaluator = Evaluator(
        runner=runner,
        enable_ragas=not args.no_ragas,
        enable_nli=not args.no_nli,
        enable_judge=not args.no_judge,
        enable_semantic=not args.no_semantic,
        max_concurrency=args.concurrency,
        system_prompt=args.system_prompt,
    )

    results = []

    # If runs > 1, repeat each item and aggregate latency
    eval_items = items * args.runs if args.runs > 1 else items

    console.print(f"\n[bold]Running evaluation[/bold] ({len(eval_items)} queries, concurrency={args.concurrency})…\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating", total=len(eval_items))

        def on_result(result):
            results.append(result)
            progress.advance(task)

        results = await evaluator.evaluate_dataset(eval_items, progress_callback=on_result)

    # Aggregate
    aggregated = evaluator.aggregate(results)

    # Console report
    reporter = ConsoleReporter(console=console)
    if args.report in ("console", "both"):
        reporter.print_summary(aggregated)

    # Export JSON
    output_path = build_output_path(args.model, args.output)
    exporter = JSONExporter()
    run_meta = {
        "model": args.model,
        "dataset": str(args.dataset),
        "runs": args.runs,
        "max_items": args.max_items,
        "flags": {
            "ragas": not args.no_ragas,
            "nli": not args.no_nli,
            "judge": not args.no_judge,
            "semantic": not args.no_semantic,
        },
    }
    payload = exporter._build_payload(results, aggregated, run_meta)
    saved = exporter.export(results, aggregated, output_path, run_meta)
    console.print(f"\n[green]✓ Results saved:[/green] {saved}")

    # HTML report
    if args.report in ("html", "both"):
        html_path = saved.with_suffix(".html")
        html_reporter = HTMLReporter()
        html_reporter.generate(payload, html_path)
        console.print(f"[green]✓ HTML report:[/green] {html_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    console.print(
        "\n[bold blue]╔══════════════════════════════════╗[/bold blue]\n"
        "[bold blue]║   LLM Evaluation Harness v1.0    ║[/bold blue]\n"
        "[bold blue]╚══════════════════════════════════╝[/bold blue]\n"
    )

    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
