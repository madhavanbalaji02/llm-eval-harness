"""Rich-formatted terminal report for evaluation results."""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from ..core import AggregatedResults, EvalResult
from ..metrics.cost import format_cost


class ConsoleReporter:
    """Renders evaluation results to the terminal using Rich."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    def print_result(self, result: EvalResult) -> None:
        """Print a single EvalResult as it arrives (live progress mode)."""
        status = "[red]ERR[/red]" if result.error else "[green]OK[/green]"
        em = "[green]✓[/green]" if result.exact_match else "[red]✗[/red]"
        hall = "[red]HALL[/red]" if result.is_hallucination else "[dim]ok[/dim]"
        sem = f"{result.semantic_similarity:.2f}"
        lat = f"{result.latency_ms:.0f}ms"
        # Use \[ to escape brackets so Rich doesn't parse the id as a markup tag
        self.console.print(
            f"  {status} \\[{result.id}] {lat} | EM:{em} | Sem:{sem} | {hall} | {format_cost(result.cost_usd)}"
        )

    def print_summary(self, agg: AggregatedResults) -> None:
        """Print a Rich-formatted summary panel for aggregated results."""
        self.console.print()
        self.console.rule(f"[bold cyan]Evaluation Summary — {agg.model}[/bold cyan]")

        # Latency table
        lat_table = Table(box=box.SIMPLE, show_header=True, header_style="bold yellow")
        lat_table.add_column("Metric", style="dim")
        lat_table.add_column("Value", justify="right")

        lat_table.add_row("P50 Latency", f"{agg.latency_p50:.1f} ms")
        lat_table.add_row("P95 Latency", f"{agg.latency_p95:.1f} ms")
        lat_table.add_row("P99 Latency", f"{agg.latency_p99:.1f} ms")
        lat_table.add_row("Mean Latency", f"{agg.latency_mean:.1f} ms")
        if agg.mean_ttft_ms is not None:
            lat_table.add_row("Mean TTFT", f"{agg.mean_ttft_ms:.1f} ms")
        if agg.mean_tokens_per_sec is not None:
            lat_table.add_row("Tokens/sec", f"{agg.mean_tokens_per_sec:.1f}")

        # Accuracy table
        acc_table = Table(box=box.SIMPLE, show_header=True, header_style="bold green")
        acc_table.add_column("Metric", style="dim")
        acc_table.add_column("Value", justify="right")

        acc_table.add_row("Exact Match Rate", f"{agg.exact_match_rate:.1%}")
        acc_table.add_row("Mean Semantic Sim.", f"{agg.mean_semantic_similarity:.3f}")
        if agg.mean_llm_judge_score is not None:
            acc_table.add_row("LLM Judge Score", f"{agg.mean_llm_judge_score:.2f} / 5.0")
        if agg.mean_answer_relevancy is not None:
            acc_table.add_row("RAGAS Relevancy", f"{agg.mean_answer_relevancy:.3f}")
        if agg.mean_context_precision is not None:
            acc_table.add_row("RAGAS Precision", f"{agg.mean_context_precision:.3f}")
        if agg.mean_context_recall is not None:
            acc_table.add_row("RAGAS Recall", f"{agg.mean_context_recall:.3f}")

        # Hallucination table
        hall_table = Table(box=box.SIMPLE, show_header=True, header_style="bold red")
        hall_table.add_column("Metric", style="dim")
        hall_table.add_column("Value", justify="right")

        hall_color = "red" if agg.hallucination_rate > 0.2 else "yellow" if agg.hallucination_rate > 0.05 else "green"
        hall_table.add_row("Hallucination Rate", f"[{hall_color}]{agg.hallucination_rate:.1%}[/{hall_color}]")
        if agg.mean_faithfulness is not None:
            hall_table.add_row("RAGAS Faithfulness", f"{agg.mean_faithfulness:.3f}")
        if agg.mean_nli_score is not None:
            hall_table.add_row("Mean NLI Score", f"{agg.mean_nli_score:.3f}")

        # Cost table
        cost_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        cost_table.add_column("Metric", style="dim")
        cost_table.add_column("Value", justify="right")

        cost_table.add_row("Total Cost", format_cost(agg.total_cost_usd))
        cost_table.add_row("Cost/Query", format_cost(agg.mean_cost_per_query))
        cost_table.add_row("Prompt Tokens", f"{agg.total_prompt_tokens:,}")
        cost_table.add_row("Completion Tokens", f"{agg.total_completion_tokens:,}")

        self.console.print(Panel(lat_table, title="[yellow]Latency[/yellow]", border_style="yellow"))
        self.console.print(Panel(acc_table, title="[green]Accuracy[/green]", border_style="green"))
        self.console.print(Panel(hall_table, title="[red]Hallucination[/red]", border_style="red"))
        self.console.print(Panel(cost_table, title="[magenta]Cost[/magenta]", border_style="magenta"))

        # Error rate warning
        if agg.error_rate > 0:
            self.console.print(
                f"\n[bold red]⚠  Error rate: {agg.error_rate:.1%} ({agg.n_samples - agg.n_successful}/{agg.n_samples} failed)[/bold red]"
            )
        self.console.print(f"\n[dim]Total samples: {agg.n_samples} | Successful: {agg.n_successful}[/dim]")

    def print_comparison(self, agg_results: list[AggregatedResults]) -> None:
        """Print a side-by-side comparison table for multiple models."""
        if not agg_results:
            return

        self.console.print()
        self.console.rule("[bold cyan]Model Comparison[/bold cyan]")

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold white")
        table.add_column("Metric", style="bold", no_wrap=True)
        for agg in agg_results:
            table.add_column(agg.model.split("/")[-1], justify="right")

        def row(label: str, *values: str) -> None:
            table.add_row(label, *values)

        def fmt_opt(v, fmt=".3f") -> str:
            return f"{v:{fmt}}" if v is not None else "—"

        row("P50 Latency (ms)", *[f"{a.latency_p50:.0f}" for a in agg_results])
        row("P95 Latency (ms)", *[f"{a.latency_p95:.0f}" for a in agg_results])
        row("TTFT (ms)", *[fmt_opt(a.mean_ttft_ms, ".0f") for a in agg_results])
        row("Tokens/sec", *[fmt_opt(a.mean_tokens_per_sec, ".1f") for a in agg_results])
        row("─" * 20, *["─" * 10 for _ in agg_results])
        row("Exact Match", *[f"{a.exact_match_rate:.1%}" for a in agg_results])
        row("Semantic Sim.", *[f"{a.mean_semantic_similarity:.3f}" for a in agg_results])
        row("LLM Judge (/5)", *[fmt_opt(a.mean_llm_judge_score) for a in agg_results])
        row("─" * 20, *["─" * 10 for _ in agg_results])
        row("Hallucination %", *[f"{a.hallucination_rate:.1%}" for a in agg_results])
        row("Faithfulness", *[fmt_opt(a.mean_faithfulness) for a in agg_results])
        row("RAGAS Relevancy", *[fmt_opt(a.mean_answer_relevancy) for a in agg_results])
        row("─" * 20, *["─" * 10 for _ in agg_results])
        row("Cost/Query", *[format_cost(a.mean_cost_per_query) for a in agg_results])
        row("Total Cost", *[format_cost(a.total_cost_usd) for a in agg_results])

        self.console.print(table)
