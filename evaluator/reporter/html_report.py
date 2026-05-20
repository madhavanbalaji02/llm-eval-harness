"""Self-contained HTML report with Plotly charts."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class HTMLReporter:
    """Generates a self-contained HTML report from evaluation results.

    All Plotly charts are embedded as JSON and rendered via CDN, so the
    output is a single .html file that works offline (after first load).
    """

    def generate(
        self,
        results_payload: dict[str, Any],
        output_path: str | Path,
        title: Optional[str] = None,
    ) -> Path:
        """Write a self-contained HTML report.

        Args:
            results_payload: The dict produced by JSONExporter._build_payload.
            output_path: Where to write the .html file.
            title: Optional page title.

        Returns:
            Resolved path of the written file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        summary = results_payload.get("summary", {})
        raw_results = results_payload.get("results", [])
        model = summary.get("model", "unknown")
        report_title = title or f"LLM Eval: {model}"

        charts = self._build_charts(summary, raw_results)
        table_html = self._build_results_table(raw_results)
        html = self._render_html(report_title, model, summary, charts, table_html)

        path.write_text(html, encoding="utf-8")
        logger.info("HTML report written to %s", path)
        return path

    # ------------------------------------------------------------------
    # Chart builders
    # ------------------------------------------------------------------

    def _build_charts(self, summary: dict, results: list[dict]) -> dict[str, Any]:
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
        except ImportError:
            logger.warning("plotly not installed — skipping charts")
            return {}

        charts = {}

        # 1. Latency bar chart
        lat = summary.get("latency", {})
        lat_fig = go.Figure(
            data=[
                go.Bar(
                    x=["P50", "P95", "P99", "Mean"],
                    y=[
                        lat.get("p50_ms", 0),
                        lat.get("p95_ms", 0),
                        lat.get("p99_ms", 0),
                        lat.get("mean_ms", 0),
                    ],
                    marker_color=["#3b82f6", "#f59e0b", "#ef4444", "#10b981"],
                    text=[f"{v:.0f}ms" for v in [
                        lat.get("p50_ms", 0), lat.get("p95_ms", 0),
                        lat.get("p99_ms", 0), lat.get("mean_ms", 0),
                    ]],
                    textposition="outside",
                )
            ]
        )
        lat_fig.update_layout(
            title="Latency Percentiles",
            yaxis_title="Latency (ms)",
            plot_bgcolor="white",
            height=350,
        )
        charts["latency"] = json.loads(pio.to_json(lat_fig))

        # 2. Accuracy radar chart
        acc = summary.get("accuracy", {})
        hall = summary.get("hallucination", {})
        ragas = summary.get("ragas", {})

        radar_values = [
            acc.get("exact_match_rate", 0),
            acc.get("mean_semantic_similarity", 0),
            (acc.get("mean_llm_judge_score") or 0) / 5.0,
            1.0 - (hall.get("rate", 0) or 0),
            ragas.get("mean_faithfulness") or ragas.get("mean_answer_relevancy") or 0,
        ]
        radar_labels = [
            "Exact Match",
            "Semantic Sim.",
            "LLM Judge (norm)",
            "Non-Hallucination",
            "RAGAS Score",
        ]
        radar_values_closed = radar_values + [radar_values[0]]
        radar_labels_closed = radar_labels + [radar_labels[0]]

        radar_fig = go.Figure(
            data=[
                go.Scatterpolar(
                    r=radar_values_closed,
                    theta=radar_labels_closed,
                    fill="toself",
                    fillcolor="rgba(59, 130, 246, 0.2)",
                    line_color="#3b82f6",
                    name=summary.get("model", "model"),
                )
            ]
        )
        radar_fig.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            showlegend=True,
            title="Quality Radar",
            height=400,
        )
        charts["radar"] = json.loads(pio.to_json(radar_fig))

        # 3. Per-item semantic similarity scatter
        if results:
            ids = [r["id"] for r in results]
            sem_sims = [r.get("accuracy", {}).get("semantic_similarity", 0) for r in results]
            latencies = [r.get("latency", {}).get("ms", 0) for r in results]
            colors = ["red" if r.get("hallucination", {}).get("is_hallucination") else "blue" for r in results]

            scatter_fig = go.Figure(
                data=[
                    go.Scatter(
                        x=latencies,
                        y=sem_sims,
                        mode="markers+text",
                        text=ids,
                        textposition="top center",
                        marker={"color": colors, "size": 10, "opacity": 0.7},
                        hovertemplate="<b>%{text}</b><br>Latency: %{x:.0f}ms<br>Sem. Sim: %{y:.3f}<extra></extra>",
                    )
                ]
            )
            scatter_fig.update_layout(
                title="Latency vs Semantic Similarity (red = hallucination)",
                xaxis_title="Latency (ms)",
                yaxis_title="Semantic Similarity",
                plot_bgcolor="white",
                height=400,
            )
            charts["scatter"] = json.loads(pio.to_json(scatter_fig))

        # 4. Cost distribution
        if results:
            costs = [r.get("cost_usd", 0) * 1000 for r in results]  # to milli-dollars
            cost_fig = go.Figure(
                data=[go.Histogram(x=costs, nbinsx=10, marker_color="#8b5cf6")]
            )
            cost_fig.update_layout(
                title="Cost Distribution per Query (m$)",
                xaxis_title="Cost (milli-USD)",
                yaxis_title="Count",
                plot_bgcolor="white",
                height=300,
            )
            charts["cost"] = json.loads(pio.to_json(cost_fig))

        return charts

    def _build_results_table(self, results: list[dict]) -> str:
        if not results:
            return "<p>No results.</p>"

        rows = []
        for r in results:
            acc = r.get("accuracy", {})
            hall = r.get("hallucination", {})
            lat = r.get("latency", {})
            em = "✓" if acc.get("exact_match") else "✗"
            is_hall = "⚠" if hall.get("is_hallucination") else "—"
            sem = f"{acc.get('semantic_similarity', 0):.3f}"
            judge = f"{acc.get('llm_judge_score', '—')}"
            latency = f"{lat.get('ms', 0):.0f}ms"
            cost = f"${r.get('cost_usd', 0) * 1000:.3f}m"
            error = r.get("error") or ""

            q_short = (r.get("question", "")[:80] + "…") if len(r.get("question", "")) > 80 else r.get("question", "")
            a_short = (r.get("answer", "")[:80] + "…") if len(r.get("answer", "")) > 80 else r.get("answer", "")

            hall_class = ' class="hallucination"' if hall.get("is_hallucination") else ""
            rows.append(
                f'<tr{hall_class}>'
                f'<td>{r.get("id","")}</td>'
                f'<td title="{r.get("question","")}">{q_short}</td>'
                f'<td title="{r.get("answer","")}">{a_short}</td>'
                f'<td class="center">{em}</td>'
                f'<td class="center">{sem}</td>'
                f'<td class="center">{judge}</td>'
                f'<td class="center">{is_hall}</td>'
                f'<td class="center">{latency}</td>'
                f'<td class="center">{cost}</td>'
                f'<td class="error">{error}</td>'
                f'</tr>'
            )

        return (
            '<table id="results-table">'
            '<thead><tr>'
            '<th>ID</th><th>Question</th><th>Answer</th><th>EM</th>'
            '<th>Sem.Sim</th><th>Judge</th><th>Hall.</th>'
            '<th>Latency</th><th>Cost</th><th>Error</th>'
            '</tr></thead>'
            '<tbody>' + "\n".join(rows) + "</tbody>"
            "</table>"
        )

    def _render_html(
        self,
        title: str,
        model: str,
        summary: dict,
        charts: dict[str, Any],
        table_html: str,
    ) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %Human:%M UTC").replace("Human", "%H")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        def chart_div(key: str, chart_id: str) -> str:
            if key not in charts:
                return ""
            return (
                f'<div class="chart" id="{chart_id}"></div>'
                f'<script>Plotly.react("{chart_id}", '
                f'{json.dumps(charts[key]["data"])}, '
                f'{json.dumps(charts[key]["layout"])});</script>'
            )

        acc = summary.get("accuracy", {})
        hall = summary.get("hallucination", {})
        cost = summary.get("cost", {})
        lat = summary.get("latency", {})

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 3px solid #3b82f6; padding-bottom: 12px; }}
  h2 {{ color: #1e40af; margin-top: 32px; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 24px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin: 24px 0; }}
  .kpi {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.1); text-align: center; }}
  .kpi .value {{ font-size: 2em; font-weight: 700; color: #1d4ed8; }}
  .kpi .label {{ font-size: 0.8em; color: #64748b; margin-top: 4px; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 24px 0; }}
  .chart {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); font-size: 0.85em; }}
  th {{ background: #1e40af; color: white; padding: 10px 8px; text-align: left; }}
  td {{ padding: 8px; border-bottom: 1px solid #e2e8f0; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .center {{ text-align: center; }}
  tr.hallucination {{ background: #fef2f2; }}
  .error {{ color: #ef4444; font-size: 0.8em; }}
  tr:hover {{ background: #f1f5f9; }}
  @media (max-width: 768px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>📊 {title}</h1>
<div class="meta">Model: <strong>{model}</strong> | Generated: {ts} | Samples: {summary.get('n_successful', 0)}/{summary.get('n_samples', 0)}</div>

<h2>Key Metrics</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="value">{lat.get('p50_ms', 0):.0f}ms</div><div class="label">P50 Latency</div></div>
  <div class="kpi"><div class="value">{lat.get('p95_ms', 0):.0f}ms</div><div class="label">P95 Latency</div></div>
  <div class="kpi"><div class="value">{acc.get('exact_match_rate', 0):.1%}</div><div class="label">Exact Match</div></div>
  <div class="kpi"><div class="value">{acc.get('mean_semantic_similarity', 0):.3f}</div><div class="label">Semantic Sim.</div></div>
  <div class="kpi"><div class="value">{hall.get('rate', 0):.1%}</div><div class="label">Hallucination Rate</div></div>
  <div class="kpi"><div class="value">${cost.get('total_usd', 0) * 1000:.3f}m</div><div class="label">Total Cost</div></div>
</div>

<h2>Charts</h2>
<div class="chart-grid">
  {chart_div('latency', 'chart-latency')}
  {chart_div('radar', 'chart-radar')}
  {chart_div('scatter', 'chart-scatter')}
  {chart_div('cost', 'chart-cost')}
</div>

<h2>Per-Question Results</h2>
{table_html}

</body>
</html>"""
