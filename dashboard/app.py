"""Streamlit dashboard: upload results JSON files and compare models side-by-side."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Eval Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def load_result_file(uploaded) -> dict[str, Any] | None:
    try:
        content = uploaded.read().decode("utf-8")
        return json.loads(content)
    except Exception as exc:
        st.error(f"Failed to parse {uploaded.name}: {exc}")
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
        # Latency
        "p50_ms": lat.get("p50_ms", 0),
        "p95_ms": lat.get("p95_ms", 0),
        "p99_ms": lat.get("p99_ms", 0),
        "mean_ms": lat.get("mean_ms", 0),
        "ttft_ms": lat.get("mean_ttft_ms"),
        "tokens_per_sec": lat.get("mean_tokens_per_sec"),
        # Accuracy
        "exact_match": acc.get("exact_match_rate", 0),
        "semantic_sim": acc.get("mean_semantic_similarity", 0),
        "judge_score": acc.get("mean_llm_judge_score"),
        # Hallucination
        "hallucination_rate": hall.get("rate", 0),
        "faithfulness": hall.get("mean_faithfulness"),
        "nli_score": hall.get("mean_nli_score"),
        # RAGAS
        "answer_relevancy": ragas.get("mean_answer_relevancy"),
        "context_precision": ragas.get("mean_context_precision"),
        "context_recall": ragas.get("mean_context_recall"),
        # Cost
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
        rows.append(
            {
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
            }
        )
    return pd.DataFrame(rows)


def _opt_fmt(v, fmt=".3f") -> str:
    return f"{v:{fmt}}" if v is not None else "—"


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("📁 Upload Results")
uploaded_files = st.sidebar.file_uploader(
    "Upload evaluation JSON files",
    type=["json"],
    accept_multiple_files=True,
    help="Upload one or more result files exported by run_eval.py",
)

# Load from results/ directory as a convenience
results_dir = Path("results")
local_files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
if local_files:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Local results/")
    selected_local = st.sidebar.multiselect(
        "Or select local files",
        [f.name for f in local_files],
    )

# ── Load payloads ─────────────────────────────────────────────────────────────

payloads: list[dict] = []

for f in uploaded_files:
    p = load_result_file(f)
    if p:
        payloads.append(p)

if "selected_local" in dir() and selected_local:
    for fname in selected_local:
        fpath = results_dir / fname
        try:
            payloads.append(json.loads(fpath.read_text(encoding="utf-8")))
        except Exception as exc:
            st.error(f"Failed to load {fname}: {exc}")

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("📊 LLM Evaluation Dashboard")
st.caption("Upload result JSON files produced by `run_eval.py` to compare models.")

if not payloads:
    st.info(
        "👆 Upload one or more evaluation result JSON files using the sidebar to get started.\n\n"
        "Run `python scripts/run_eval.py --model gpt-4o-mini --report none --output results/run.json` to generate a result file."
    )
    st.stop()

summaries = [extract_summary(p) for p in payloads]
all_dfs = [results_to_df(p) for p in payloads]
combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

models = [s["model"] for s in summaries]
st.success(f"Loaded **{len(payloads)}** run(s): {', '.join(models)}")

# ── KPI summary cards ─────────────────────────────────────────────────────────

st.subheader("Quick Comparison")
cols = st.columns(len(summaries))
for col, s in zip(cols, summaries):
    model_short = s["model"].split("/")[-1]
    col.metric(label=f"⚡ P50 Latency — {model_short}", value=f"{s['p50_ms']:.0f} ms")
    col.metric("✅ Exact Match", f"{s['exact_match']:.1%}")
    col.metric("🧠 Sem. Sim.", f"{s['semantic_sim']:.3f}")
    col.metric("⚠️ Hallucination", f"{s['hallucination_rate']:.1%}")
    col.metric("💰 Cost/Query", f"${s['cost_per_query_usd'] * 1000:.3f}m")

st.divider()

# ── Comparison charts ─────────────────────────────────────────────────────────

tab_charts, tab_table, tab_details = st.tabs(["📈 Charts", "📋 Results Table", "🔍 Per-Question"])

with tab_charts:
    col1, col2 = st.columns(2)

    with col1:
        # Latency comparison
        fig_lat = go.Figure()
        for s in summaries:
            model_short = s["model"].split("/")[-1]
            fig_lat.add_trace(
                go.Bar(
                    name=model_short,
                    x=["P50", "P95", "P99", "Mean"],
                    y=[s["p50_ms"], s["p95_ms"], s["p99_ms"], s["mean_ms"]],
                )
            )
        fig_lat.update_layout(
            barmode="group",
            title="Latency Comparison (ms)",
            yaxis_title="ms",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    with col2:
        # Accuracy comparison
        metric_names = ["Exact Match", "Semantic Sim.", "Hall.-Free (inv)"]
        fig_acc = go.Figure()
        for s in summaries:
            model_short = s["model"].split("/")[-1]
            fig_acc.add_trace(
                go.Bar(
                    name=model_short,
                    x=metric_names,
                    y=[
                        s["exact_match"],
                        s["semantic_sim"],
                        1.0 - s["hallucination_rate"],
                    ],
                )
            )
        fig_acc.update_layout(
            barmode="group",
            title="Accuracy & Quality Metrics",
            yaxis={"range": [0, 1], "tickformat": ".0%"},
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Cost comparison
        fig_cost = go.Figure(
            data=[
                go.Bar(
                    x=[s["model"].split("/")[-1] for s in summaries],
                    y=[s["cost_per_query_usd"] * 1000 for s in summaries],
                    marker_color="#8b5cf6",
                    text=[f"${s['cost_per_query_usd'] * 1000:.3f}m" for s in summaries],
                    textposition="outside",
                )
            ]
        )
        fig_cost.update_layout(
            title="Cost per Query (milli-USD)",
            yaxis_title="m$",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with col4:
        # Radar chart for multi-model comparison
        radar_metrics = ["Exact Match", "Semantic Sim.", "Non-Hall.", "Faithfulness", "Relevancy"]
        fig_radar = go.Figure()
        for s in summaries:
            model_short = s["model"].split("/")[-1]
            vals = [
                s["exact_match"],
                s["semantic_sim"],
                1.0 - s["hallucination_rate"],
                s["faithfulness"] or 0,
                s["answer_relevancy"] or 0,
            ]
            vals_closed = vals + [vals[0]]
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=vals_closed,
                    theta=radar_metrics + [radar_metrics[0]],
                    fill="toself",
                    name=model_short,
                    opacity=0.6,
                )
            )
        fig_radar.update_layout(
            polar={"radialaxis": {"range": [0, 1]}},
            title="Multi-Model Quality Radar",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # RAGAS metrics if available
    ragas_keys = ["answer_relevancy", "context_precision", "context_recall", "faithfulness"]
    has_ragas = any(s.get(k) is not None for s in summaries for k in ragas_keys)
    if has_ragas:
        st.subheader("RAGAS Metrics")
        fig_ragas = go.Figure()
        for s in summaries:
            model_short = s["model"].split("/")[-1]
            fig_ragas.add_trace(
                go.Bar(
                    name=model_short,
                    x=["Answer Relevancy", "Context Precision", "Context Recall", "Faithfulness"],
                    y=[
                        s.get("answer_relevancy") or 0,
                        s.get("context_precision") or 0,
                        s.get("context_recall") or 0,
                        s.get("faithfulness") or 0,
                    ],
                )
            )
        fig_ragas.update_layout(
            barmode="group",
            title="RAGAS Metrics",
            yaxis={"range": [0, 1]},
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_ragas, use_container_width=True)

with tab_table:
    st.subheader("Aggregated Summary Table")
    summary_rows = []
    for s in summaries:
        summary_rows.append(
            {
                "Model": s["model"],
                "Samples": s["n_successful"],
                "P50 (ms)": f"{s['p50_ms']:.0f}",
                "P95 (ms)": f"{s['p95_ms']:.0f}",
                "TTFT (ms)": _opt_fmt(s["ttft_ms"], ".0f"),
                "Tok/sec": _opt_fmt(s["tokens_per_sec"], ".1f"),
                "Exact Match": f"{s['exact_match']:.1%}",
                "Semantic Sim.": f"{s['semantic_sim']:.3f}",
                "Judge Score": _opt_fmt(s["judge_score"]),
                "Hall. Rate": f"{s['hallucination_rate']:.1%}",
                "Faithfulness": _opt_fmt(s["faithfulness"]),
                "RAGAS Rel.": _opt_fmt(s["answer_relevancy"]),
                "Cost/Q (m$)": f"{s['cost_per_query_usd'] * 1000:.3f}",
                "Total Cost ($)": f"{s['total_cost_usd']:.4f}",
                "Errors": f"{s['error_rate']:.1%}",
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)

    csv_buf = StringIO()
    summary_df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Export Summary CSV",
        csv_buf.getvalue(),
        "eval_summary.csv",
        "text/csv",
    )

with tab_details:
    st.subheader("Per-Question Results")

    if not combined_df.empty:
        # Filters
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            selected_models = st.multiselect(
                "Filter by model",
                options=combined_df["model"].unique().tolist(),
                default=combined_df["model"].unique().tolist(),
            )
        with filter_col2:
            show_hall_only = st.checkbox("Show hallucinations only", value=False)
        with filter_col3:
            min_sem = st.slider("Min semantic similarity", 0.0, 1.0, 0.0, 0.01)

        filtered = combined_df[combined_df["model"].isin(selected_models)]
        if show_hall_only:
            filtered = filtered[filtered["is_hallucination"] == True]
        filtered = filtered[filtered["semantic_sim"] >= min_sem]

        # Sort
        sort_col = st.selectbox(
            "Sort by",
            ["id", "latency_ms", "semantic_sim", "cost_usd", "is_hallucination"],
            index=0,
        )
        sort_asc = st.checkbox("Ascending", value=True)
        filtered = filtered.sort_values(sort_col, ascending=sort_asc)

        st.dataframe(
            filtered[
                [
                    "model", "id", "question", "answer", "ground_truth",
                    "exact_match", "semantic_sim", "judge_score",
                    "is_hallucination", "nli_label", "latency_ms", "cost_usd", "error",
                ]
            ].reset_index(drop=True),
            use_container_width=True,
            height=500,
        )

        detail_csv = StringIO()
        filtered.to_csv(detail_csv, index=False)
        st.download_button(
            "⬇️ Export Filtered Results CSV",
            detail_csv.getvalue(),
            "eval_per_question.csv",
            "text/csv",
        )
    else:
        st.info("No per-question data available.")

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption("LLM Evaluation Harness v1.0 | Built with Streamlit + Plotly")
