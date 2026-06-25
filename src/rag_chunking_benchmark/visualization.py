"""Plotly chart builders for Streamlit dashboards."""

from .common import *
from .schemas import MethodMetrics

def _bar(methods, values, title, color_idx=0):
    colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(methods))]
    fig = go.Figure(go.Bar(x=methods, y=values, marker_color=colors))
    fig.update_layout(title=title, template="plotly_dark",
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#e0e0e0"), height=380)
    return fig

def chart_final_scores(method_metrics: List[MethodMetrics]):
    methods = [m.method for m in method_metrics]
    scores  = [m.final_score for m in method_metrics]
    return _bar(methods, scores, "Final Benchmark Scores")

def chart_answer_quality(method_metrics: List[MethodMetrics]):
    methods = [m.method for m in method_metrics]
    fig = go.Figure()
    for attr, label, color in [
        ("avg_relevance", "Relevance", "#4F8EF7"),
        ("avg_correctness", "Correctness", "#4FC78A"),
        ("avg_completeness", "Completeness", "#F7C94F"),
    ]:
        fig.add_trace(go.Bar(name=label,
                             x=methods,
                             y=[getattr(m, attr) for m in method_metrics],
                             marker_color=color))
    fig.update_layout(barmode="group", title="Answer Quality by Method",
                      template="plotly_dark",
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#e0e0e0"), height=380)
    return fig

def chart_latency(method_metrics: List[MethodMetrics]):
    methods = [m.method for m in method_metrics]
    latencies = [m.total_pipeline_time_ms for m in method_metrics]
    return _bar(methods, latencies, "Total Pipeline Latency (ms)", color_idx=2)

def chart_tokens(method_metrics: List[MethodMetrics]):
    methods = [m.method for m in method_metrics]
    ctx_tokens = [m.avg_context_tokens for m in method_metrics]
    ans_tokens = [m.avg_answer_tokens  for m in method_metrics]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Avg Context Tokens", x=methods, y=ctx_tokens, marker_color="#4F8EF7"))
    fig.add_trace(go.Bar(name="Avg Answer Tokens",  x=methods, y=ans_tokens, marker_color="#F76F4F"))
    fig.update_layout(barmode="group", title="Token Usage by Method",
                      template="plotly_dark",
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#e0e0e0"), height=380)
    return fig

def chart_scatter(method_metrics: List[MethodMetrics]):
    methods = [m.method for m in method_metrics]
    quality = [m.avg_correctness for m in method_metrics]
    latency = [m.total_pipeline_time_ms for m in method_metrics]
    fig = px.scatter(
        x=latency, y=quality, text=methods,
        labels={"x": "Total Latency (ms)", "y": "Avg Correctness"},
        title="Quality vs Latency Trade-off",
        template="plotly_dark",
        color_discrete_sequence=CHART_COLORS,
    )
    fig.update_traces(textposition="top center", marker_size=14)
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#e0e0e0"), height=420)
    return fig

def chart_radar(method_metrics: List[MethodMetrics]):
    categories = ["Relevance", "Correctness", "Faithfulness",
                  "Speed (inv)", "Context Sufficiency", "Completeness"]
    # normalise speed: 1 - minmax
    lats = [m.total_pipeline_time_ms for m in method_metrics]
    lat_max = max(lats) if max(lats) > 0 else 1
    fig = go.Figure()
    for i, m in enumerate(method_metrics):
        speed_inv = 1.0 - m.total_pipeline_time_ms / lat_max
        vals = [m.avg_relevance, m.avg_correctness, m.avg_faithfulness,
                speed_inv, m.avg_context_sufficiency, m.avg_completeness]
        vals += [vals[0]]  # close polygon
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=categories + [categories[0]],
            fill="toself", name=m.method,
            line_color=CHART_COLORS[i % len(CHART_COLORS)],
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      title="Multi-Metric Radar Chart",
                      template="plotly_dark",
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#e0e0e0"), height=480)
    return fig
