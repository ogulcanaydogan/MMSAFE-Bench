"""Interactive Plotly chart generation for safety reports."""

from __future__ import annotations

from typing import Any


def create_asr_by_category_chart(by_category: dict[str, dict[str, Any]]) -> str:
    """Create a vertical bar chart of attack success rate by hazard category.

    Args:
        by_category: Mapping of category codes to dicts with ``asr`` key (0.0–1.0).

    Returns:
        Plotly HTML fragment string (without full page wrapper).
    """
    import plotly.graph_objects as go

    categories = sorted(by_category.keys())
    asr_values = [by_category[c]["asr"] * 100 for c in categories]

    colors = [
        "#ef4444" if v > 50 else "#f97316" if v > 25 else "#22c55e"
        for v in asr_values
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=asr_values,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in asr_values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Attack Success Rate by Hazard Category",
        xaxis_title="Hazard Category",
        yaxis_title="ASR (%)",
        yaxis_range=[0, max(110, max(asr_values) * 1.2) if asr_values else 100],
        template="plotly_white",
        height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_asr_by_model_chart(by_model: dict[str, dict[str, Any]]) -> str:
    """Create a horizontal bar chart of attack success rate by model.

    Args:
        by_model: Mapping of model names to dicts with ``asr`` key (0.0–1.0).

    Returns:
        Plotly HTML fragment string.
    """
    import plotly.graph_objects as go

    models = sorted(by_model.keys())
    asr_values = [by_model[m]["asr"] * 100 for m in models]

    colors = [
        "#ef4444" if v > 50 else "#f97316" if v > 25 else "#22c55e"
        for v in asr_values
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                y=models,
                x=asr_values,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}%" for v in asr_values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Attack Success Rate by Model",
        xaxis_title="ASR (%)",
        xaxis_range=[0, max(110, max(asr_values) * 1.2) if asr_values else 100],
        template="plotly_white",
        height=max(300, len(models) * 50 + 100),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_asr_by_attack_chart(by_attack: dict[str, dict[str, Any]]) -> str:
    """Create a bar chart of attack success rate by attack strategy.

    Args:
        by_attack: Mapping of attack names to dicts with ``asr`` key (0.0–1.0).

    Returns:
        Plotly HTML fragment string.
    """
    import plotly.graph_objects as go

    attacks = sorted(by_attack.keys(), key=lambda k: -by_attack[k]["asr"])
    asr_values = [by_attack[a]["asr"] * 100 for a in attacks]

    fig = go.Figure(
        data=[
            go.Bar(
                x=attacks,
                y=asr_values,
                marker_color="#6366f1",
                text=[f"{v:.1f}%" for v in asr_values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Attack Success Rate by Strategy",
        xaxis_title="Attack Strategy",
        yaxis_title="ASR (%)",
        yaxis_range=[0, max(110, max(asr_values) * 1.2) if asr_values else 100],
        template="plotly_white",
        height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_radar_chart(by_category: dict[str, dict[str, Any]]) -> str:
    """Create a radar (spider) chart of ASR across hazard categories.

    Args:
        by_category: Mapping of category codes to dicts with ``asr`` key (0.0–1.0).

    Returns:
        Plotly HTML fragment string.
    """
    import plotly.graph_objects as go

    categories = sorted(by_category.keys())
    asr_values = [by_category[c]["asr"] * 100 for c in categories]
    # Close the polygon
    categories_closed = categories + [categories[0]] if categories else []
    asr_closed = asr_values + [asr_values[0]] if asr_values else []

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=asr_closed,
                theta=categories_closed,
                fill="toself",
                fillcolor="rgba(239, 68, 68, 0.2)",
                line=dict(color="#ef4444"),
                name="ASR",
            )
        ]
    )
    fig.update_layout(
        title="Safety Profile — ASR by Category",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template="plotly_white",
        height=450,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_refusal_heatmap(
    by_category: dict[str, dict[str, Any]],
) -> str:
    """Create a single-row heatmap of refusal rates by hazard category.

    Args:
        by_category: Mapping of category codes to dicts with optional ``rr`` key.

    Returns:
        Plotly HTML fragment string.
    """
    import plotly.graph_objects as go

    categories = sorted(by_category.keys())
    rr_values = [by_category[c].get("rr", 0.0) * 100 for c in categories]

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=[rr_values],
                x=categories,
                y=["Refusal Rate"],
                colorscale="Blues",
                text=[[f"{v:.1f}%" for v in rr_values]],
                texttemplate="%{text}",
                showscale=True,
                colorbar_title="RR (%)",
            )
        ]
    )
    fig.update_layout(
        title="Refusal Rate by Hazard Category",
        template="plotly_white",
        height=200,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)
