"""HTML report generation with Plotly charts."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

import mmsafe
from mmsafe.metrics.aggregator import generate_summary
from mmsafe.pipeline.result_types import EvalRun
from mmsafe.reporting.charts import (
    create_asr_by_attack_chart,
    create_asr_by_category_chart,
    create_asr_by_model_chart,
    create_radar_chart,
    create_refusal_heatmap,
)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def generate_html_report(run: EvalRun, output_path: Path) -> Path:
    """Generate an interactive HTML report with Plotly charts.

    Returns the path to the written report.
    """
    summary = generate_summary(run)

    # Generate charts (only if data exists)
    chart_category = ""
    chart_radar = ""
    chart_model = ""
    chart_attack = ""
    chart_refusal = ""

    if summary["by_category"]:
        chart_category = create_asr_by_category_chart(summary["by_category"])
        chart_radar = create_radar_chart(summary["by_category"])
        chart_refusal = create_refusal_heatmap(summary["by_category"])
    if summary["by_model"]:
        chart_model = create_asr_by_model_chart(summary["by_model"])
    if summary["by_attack"]:
        chart_attack = create_asr_by_attack_chart(summary["by_attack"])

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html.j2")

    html = template.render(
        summary=summary,
        version=mmsafe.__version__,
        chart_category=chart_category,
        chart_radar=chart_radar,
        chart_model=chart_model,
        chart_attack=chart_attack,
        chart_refusal=chart_refusal,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path
