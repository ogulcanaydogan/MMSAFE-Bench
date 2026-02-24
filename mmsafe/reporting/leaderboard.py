"""Model safety leaderboard generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from mmsafe.metrics.aggregator import aggregate_by_model
from mmsafe.pipeline.result_types import EvalRun

_TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass
class LeaderboardEntry:
    """A single entry in the safety leaderboard."""

    model: str
    safety_score: float
    asr: float
    rr: float
    samples: int


def build_leaderboard(run: EvalRun) -> list[LeaderboardEntry]:
    """Build a sorted leaderboard from evaluation results.

    Models are ranked by safety score (100 - ASR*100), highest first.
    """
    by_model = aggregate_by_model(run)
    entries = []

    for model, metrics in by_model.items():
        safety_score = (1.0 - metrics.attack_success_rate) * 100
        entries.append(
            LeaderboardEntry(
                model=model,
                safety_score=round(safety_score, 2),
                asr=metrics.attack_success_rate,
                rr=metrics.refusal_rate,
                samples=metrics.total_samples,
            )
        )

    entries.sort(key=lambda e: e.safety_score, reverse=True)
    return entries


def save_leaderboard_json(entries: list[LeaderboardEntry], output_path: Path) -> Path:
    """Save the leaderboard as JSON."""
    data = [
        {
            "rank": i + 1,
            "model": e.model,
            "safety_score": e.safety_score,
            "asr": round(e.asr, 4),
            "refusal_rate": round(e.rr, 4),
            "samples": e.samples,
        }
        for i, e in enumerate(entries)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    return output_path


def generate_leaderboard_html(entries: list[LeaderboardEntry], output_path: Path) -> Path:
    """Generate an interactive HTML leaderboard."""
    chart_comparison = _create_comparison_chart(entries)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("leaderboard.html.j2")

    html = template.render(
        entries=entries,
        chart_comparison=chart_comparison,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path


def merge_leaderboard(
    existing_path: Path,
    new_entries: list[LeaderboardEntry],
) -> list[LeaderboardEntry]:
    """Merge new results into an existing leaderboard, keeping best score per model."""
    existing: list[dict[str, Any]] = []
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())

    # Build lookup of best scores
    best: dict[str, LeaderboardEntry] = {}
    for entry_data in existing:
        e = LeaderboardEntry(
            model=entry_data["model"],
            safety_score=entry_data["safety_score"],
            asr=entry_data["asr"],
            rr=entry_data["refusal_rate"],
            samples=entry_data["samples"],
        )
        best[e.model] = e

    for new in new_entries:
        if new.model not in best or new.safety_score > best[new.model].safety_score:
            best[new.model] = new

    merged = list(best.values())
    merged.sort(key=lambda e: e.safety_score, reverse=True)
    return merged


def _create_comparison_chart(entries: list[LeaderboardEntry]) -> str:
    """Create a comparison bar chart for the leaderboard."""
    if not entries:
        return ""

    import plotly.graph_objects as go

    models = [e.model for e in entries]
    scores = [e.safety_score for e in entries]
    colors = [
        "#22c55e" if s >= 75 else "#f97316" if s >= 50 else "#ef4444"
        for s in scores
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                y=models,
                x=scores,
                orientation="h",
                marker_color=colors,
                text=[f"{s:.1f}" for s in scores],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Model Safety Scores (higher = safer)",
        xaxis_title="Safety Score",
        xaxis_range=[0, 110],
        template="plotly_white",
        height=max(300, len(models) * 50 + 100),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)
