"""HuggingFace Spaces deployment of the LLM Eval Dashboard.

Differences from dashboard/app.py:
- Results from results/groq_70b.json and results/groq_8b.json are pre-loaded
  on startup, so the Space works immediately without requiring file uploads.
- File uploader is still available for users who want to add their own runs.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Eval Harness — Model Comparison Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Pre-loaded result files (committed to the repo) ───────────────────────────

PRELOADED_FILES = [
    Path("results/groq_70b.json"),    # llama-3.3-70b-versatile
    Path("results/groq_8b.json"),     # llama-3.1-8b-instant
    Path("results/claude_haiku.json"), # claude-haiku-4-5-20251001
]

# ── Helpers (identical to dashboard/app.py) ───────────────────────────────────


def load_result_file(uploaded) -> dict[str, Any] | None:
    try:
        content = uploaded.read().decode("utf-8")
        return json.loads(content)
    except Exception as exc:
        st.error(f"Failed to parse {uploaded.name}: {exc}")
        return None


def load_result_path(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Failed to load {path.name}: {exc}")
        return None


def extract_summary(payload: dict) -> dict[str, Any]:
    s = payload.get("summary", {})
    lat = s.get("latency", {})
    acc = s.get("accuracy", {})
    hall = s.get("hallucination", {})
    ragas = s.get("ragas", {})
    cost = s.get("cost", {})
    meta = payload.get("metadata", {})
    return {
        "model": s.get("model", meta.get("model", "unknown")),
        "n_samples": s.get("n_samples", 0),
        "n_successful": s.get("n_successful", 0),
        "error_rate": s.get("error_rate", 0),
        "p50_ms": lat.get("p50_ms", 0),
        "p95_ms": lat.get("p95_ms", 0),
        "p99_ms": lat.get("p99_ms", 0),
        "mean_ms": lat.get("mean_ms", 0),
        "ttft_ms": lat.get("mean_ttft_ms"),
        "tokens_per_sec": lat.get("mean_tokens_per_sec"),
        "exact_match": acc.get("exact_match_rate", 0),
        "semantic_sim": acc.get("mean_semantic_similarity", 0),
        "judge_score": acc.get("mean_llm_judge_score"),
        "hallucination_rate": hall.get("rate", 0),
        "faithfulness": hall.get("mean_faithfulness"),
        "nli_score": hall.get("mean_nli_score"),
        "answer_relevancy": ragas.get("mean_answer_relevancy"),
        "context_precision": ragas.get("mean_context_precision"),
        "context_recall": ragas.get("mean_context_recall"),
        "total_cost_usd": cost.get("total_usd", 0),
        "cost_per_query_usd": cost.get("mean_per_query_usd", 0),
        "prompt_tokens": cost.get("total_prompt_tokens", 0),
        "completion_tokens": cost.get("total_completion_tokens", 0),
    }


def results_to_df(payload: dict) -> pd.DataFrame:
    rows = []
    model = payload.get("summary", {}).get("model", payload.get("metadata", {}).get("model", "?"))
    for r in payload.get("results", []):
        acc = r.get("accuracy", {})
        hall = r.get("hallucination", {})
        lat = r.get("latency", {})
        ragas = r.get("ragas", {})
        rows.append({
            "model": model,
            "id": r.get("id", ""),
            "question": r.get("question", ""),
            "answer": r.get("answer", ""),
            "ground_truth": r.get("ground_truth", ""),
            "exact_match": acc.get("exact_match", False),
            "semantic_sim": acc.get("semantic_similarity", 0),
            "judge_score": acc.get("llm_judge_score"),
            "is_hallucination": hall.get("is_hallucination", False),
            "nli_label": hall.get("nli_label"),
            "faithfulness": hall.get("faithfulness"),
            "latency_ms": lat.get("ms", 0),
            "ttft_ms": lat.get("ttft_ms"),
            "cost_usd": r.get("cost_usd", 0),
            "answer_relevancy": ragas.get("answer_relevancy"),
            "context_precision": ragas.get("context_precision"),
            "context_recall": ragas.get("context_recall"),
            "error": r.get("error"),
        })
    return pd.DataFrame(rows)


def _opt_fmt(v, fmt=".3f") -> str:
    return f"{v:{fmt}}" if v is not None else "—"


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("📁 Add Your Own Results")
uploaded_files = st.sidebar.file_uploader(
    "Upload additional result JSON files",
    type=["json"],
    accept_multiple_files=True,
    help="Upload result files from run_eval.py to compare against the pre-loaded benchmarks",
)

# ── Load payloads ─────────────────────────────────────────────────────────────

payloads: list[dict] = []

# Pre-load committed result files
preload_labels = []
for p in PRELOADED_FILES:
    if p.exists():
        payload = load_result_path(p)
        if payload:
            payloads.append(payload)
            preload_labels.append(p.name)

# Additional uploaded files
for f in uploaded_files:
    p = load_result_file(f)
    if p:
        payloads.append(p)

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("📊 LLM Eval Harness — Model Comparison Dashboard")
st.caption(
    "Pre-loaded: Groq `llama-3.3-70b-versatile` vs `llama-3.1-8b-instant` on 10 AI/ML Q&A pairs. "
    "Upload your own results via the sidebar to add more models."
)

if not payloads:
    st.error("No result files found. Make sure `results/groq_70b.json` and `results/groq_8b.json` are committed.")
    st.stop()

summaries = [extract_summary(p) for p in payloads]
all_dfs = [results_to_df(p) for p in payloads]
combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
models = [s["model"] for s in summaries]

if preload_labels:
    st.info(f"📂 Pre-loaded: {', '.join(preload_labels)}" + (f" + {len(uploaded_files)} uploaded" if uploaded_files else ""))

# ── Benchmark summary table ───────────────────────────────────────────────────

st.subheader("Benchmark Summary")

summary_rows = []
for s in summaries:
    summary_rows.append({
        "Model": s["model"],
        "P50 (ms)": f"{s['p50_ms']:.0f}",
        "P95 (ms)": f"{s['p95_ms']:.0f}",
        "TTFT (ms)": _opt_fmt(s["ttft_ms"], ".0f"),
        "Tok/sec": _opt_fmt(s["tokens_per_sec"], ".0f"),
        "Exact Match": f"{s['exact_match']:.1%}",
        "Semantic Sim.": f"{s['semantic_sim']:.3f}" if s["semantic_sim"] else "—",
        "Judge (/5)": _opt_fmt(s["judge_score"]),
        "Hall. Rate": f"{s['hallucination_rate']:.1%}",
        "RAGAS Rel.": _opt_fmt(s["answer_relevancy"]),
        "Cost/Q (m$)": f"{s['cost_per_query_usd'] * 1000:.4f}",
        "Total ($)": f"{s['total_cost_usd']:.5f}",
        "Errors": f"{s['error_rate']:.0%}",
    })

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────

st.subheader("Key Performance Indicators")
cols = st.columns(len(summaries))
for col, s in zip(cols, summaries):
    short = s["model"].split("/")[-1]
    col.markdown(f"**{short}**")
    col.metric("⚡ P50 Latency", f"{s['p50_ms']:.0f} ms")
    col.metric("🚀 TTFT", f"{_opt_fmt(s['ttft_ms'], '.0f')} ms")
    col.metric("📝 Tokens/sec", _opt_fmt(s["tokens_per_sec"], ".0f"))
    col.metric("💰 Cost/Query", f"${s['cost_per_query_usd'] * 1000:.4f}m")

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────

tab_charts, tab_table, tab_qa = st.tabs(["📈 Charts", "📋 Results Table", "🔍 Per-Question"])

with tab_charts:
    col1, col2 = st.columns(2)

    with col1:
        fig_lat = go.Figure()
        for s in summaries:
            fig_lat.add_trace(go.Bar(
                name=s["model"].split("/")[-1],
                x=["P50", "P95", "P99", "Mean"],
                y=[s["p50_ms"], s["p95_ms"], s["p99_ms"], s["mean_ms"]],
            ))
        fig_lat.update_layout(barmode="group", title="Latency Comparison (ms)",
                               yaxis_title="ms", plot_bgcolor="white")
        st.plotly_chart(fig_lat, use_container_width=True)

    with col2:
        fig_cost = go.Figure(data=[go.Bar(
            x=[s["model"].split("/")[-1] for s in summaries],
            y=[s["cost_per_query_usd"] * 1000 for s in summaries],
            marker_color=["#3b82f6", "#10b981", "#f59e0b", "#ef4444"][:len(summaries)],
            text=[f"${s['cost_per_query_usd'] * 1000:.4f}m" for s in summaries],
            textposition="outside",
        )])
        fig_cost.update_layout(title="Cost per Query (milli-USD)",
                                yaxis_title="m$", plot_bgcolor="white")
        st.plotly_chart(fig_cost, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig_ttft = go.Figure(data=[go.Bar(
            x=[s["model"].split("/")[-1] for s in summaries],
            y=[s["ttft_ms"] or 0 for s in summaries],
            marker_color=["#8b5cf6", "#06b6d4"][:len(summaries)],
            text=[f"{s['ttft_ms']:.0f}ms" if s["ttft_ms"] else "—" for s in summaries],
            textposition="outside",
        )])
        fig_ttft.update_layout(title="Time to First Token (ms)",
                                yaxis_title="ms", plot_bgcolor="white")
        st.plotly_chart(fig_ttft, use_container_width=True)

    with col4:
        fig_tps = go.Figure(data=[go.Bar(
            x=[s["model"].split("/")[-1] for s in summaries],
            y=[s["tokens_per_sec"] or 0 for s in summaries],
            marker_color=["#f59e0b", "#ef4444"][:len(summaries)],
            text=[f"{s['tokens_per_sec']:.0f}" if s["tokens_per_sec"] else "—" for s in summaries],
            textposition="outside",
        )])
        fig_tps.update_layout(title="Throughput (tokens/sec)",
                               yaxis_title="tokens/sec", plot_bgcolor="white")
        st.plotly_chart(fig_tps, use_container_width=True)

    # Radar — only if there are quality metrics
    has_quality = any(
        s.get("semantic_sim") or s.get("answer_relevancy") or s.get("faithfulness")
        for s in summaries
    )
    if has_quality:
        fig_radar = go.Figure()
        for s in summaries:
            vals = [
                s["exact_match"],
                s["semantic_sim"] or 0,
                1.0 - (s["hallucination_rate"] or 0),
                s["faithfulness"] or 0,
                s["answer_relevancy"] or 0,
            ]
            labels = ["Exact Match", "Semantic Sim.", "Non-Hall.", "Faithfulness", "Relevancy"]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=labels + [labels[0]],
                fill="toself", name=s["model"].split("/")[-1], opacity=0.6,
            ))
        fig_radar.update_layout(
            polar={"radialaxis": {"range": [0, 1]}},
            title="Quality Radar (higher = better)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("💡 Quality metrics (semantic similarity, RAGAS, hallucination) not available in these results. Re-run without `--no-semantic --no-ragas --no-nli` flags to see them here.")

with tab_table:
    csv_buf = StringIO()
    pd.DataFrame(summary_rows).to_csv(csv_buf, index=False)
    st.download_button("⬇️ Export Summary CSV", csv_buf.getvalue(), "benchmark_summary.csv", "text/csv")

with tab_qa:
    if not combined_df.empty:
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            sel_models = st.multiselect("Filter by model", options=combined_df["model"].unique().tolist(),
                                        default=combined_df["model"].unique().tolist())
        with filter_col2:
            sort_by = st.selectbox("Sort by", ["id", "latency_ms", "cost_usd"])

        filtered = combined_df[combined_df["model"].isin(sel_models)].sort_values(sort_by)
        st.dataframe(
            filtered[["model", "id", "question", "answer", "exact_match",
                       "semantic_sim", "is_hallucination", "latency_ms", "cost_usd", "error"]].reset_index(drop=True),
            use_container_width=True, height=500,
        )
        detail_csv = StringIO()
        filtered.to_csv(detail_csv, index=False)
        st.download_button("⬇️ Export Filtered CSV", detail_csv.getvalue(), "per_question.csv", "text/csv")

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "**LLM Evaluation Harness** — "
    "[GitHub](https://github.com/your-handle/llm-eval-harness) · "
    "Built with Streamlit + Plotly + Groq"
)
