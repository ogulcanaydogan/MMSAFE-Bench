"""Command-line interface for MMSAFE-Bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import click
from rich.console import Console
from rich.table import Table

from mmsafe import __version__
from mmsafe.taxonomy.categories import (
    HAZARD_TAXONOMY,
    Modality,
    get_hazards_for_modality,
)

console = Console()
ExecutionProfile = Literal["auto", "small_gpu", "a100"]

if TYPE_CHECKING:
    from collections.abc import Callable


@click.group()
@click.version_option(version=__version__, prog_name="mmsafe")
def main() -> None:
    """MMSAFE-Bench -- Multi-Modal AI Safety Evaluation Framework."""


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to evaluation YAML config.",
)
@click.option("--dry-run", is_flag=True, help="Validate config and print plan without running.")
@click.option(
    "--resume",
    type=click.Path(exists=True, path_type=Path),
    help="Resume from a checkpoint file.",
)
@click.option(
    "--execution-profile",
    type=click.Choice(["auto", "small_gpu", "a100"]),
    help="Override execution profile (auto/small_gpu/a100).",
)
@click.option(
    "--no-auto-tune",
    is_flag=True,
    help="Disable profile-based execution auto-tuning.",
)
def run(
    config: Path,
    dry_run: bool,
    resume: Path | None,
    execution_profile: str | None,
    no_auto_tune: bool,
) -> None:
    """Run a safety evaluation based on a YAML configuration."""
    from mmsafe.config.models import EvalConfig
    from mmsafe.pipeline.runner import EvalRunner

    eval_config = EvalConfig.from_yaml(config)
    if execution_profile:
        eval_config.execution.profile = cast("ExecutionProfile", execution_profile)
    if no_auto_tune:
        eval_config.execution.auto_tune = False

    console.print(f"[bold green]Loaded config:[/] {eval_config.name} v{eval_config.version}")
    console.print(f"  Models: {len(eval_config.models)}")
    console.print(f"  Datasets: {len(eval_config.datasets)}")
    console.print(f"  Attacks: {len(eval_config.attacks)}")
    console.print(f"  Judges: {len(eval_config.judges)}")
    console.print(f"  Execution profile: {eval_config.execution.profile}")
    console.print(f"  Auto tune: {eval_config.execution.auto_tune}")

    if dry_run:
        console.print("\n[yellow]Dry run mode — config is valid, no evaluation executed.[/]")
        return

    runner = EvalRunner(eval_config, resume_from=resume)
    import asyncio

    result = asyncio.run(runner.execute())
    console.print(f"\n[bold green]Evaluation complete:[/] {result.run_id}")
    console.print(f"  Samples evaluated: {len(result.samples)}")


@main.command()
@click.argument("results", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="artifacts/comparison",
    help="Output directory for comparison.",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "html", "markdown"]),
    default="html",
    help="Comparison output format.",
)
def compare(results: tuple[Path, ...], output: Path, fmt: str) -> None:
    """Compare evaluation results across multiple runs or models."""
    if len(results) < 2:
        console.print("[red]At least 2 result files required for comparison.[/]")
        raise SystemExit(1)

    # Load all result files
    reports: list[dict[str, Any]] = []
    for rpath in results:
        data = json.loads(rpath.read_text())
        reports.append(data)

    output.mkdir(parents=True, exist_ok=True)

    # Build comparison table
    console.print(f"\n[bold]Comparing {len(reports)} evaluation runs:[/]\n")
    table = Table(title="Run Comparison", show_lines=True)
    table.add_column("Metric", style="bold")

    for r in reports:
        label = r.get("summary", {}).get("config_name", r.get("config_name", "unknown"))
        table.add_column(label)

    # Add rows
    metrics: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
        ("Total Samples", lambda s: str(s.get("total_samples", 0))),
        ("ASR", lambda s: f"{s.get('attack_success_rate', 0):.2%}"),
        ("Refusal Rate", lambda s: f"{s.get('refusal_rate', 0):.2%}"),
        ("FRR", lambda s: f"{s.get('false_refusal_rate', 0):.2%}"),
        ("Errored", lambda s: str(s.get("samples_errored", 0))),
    ]

    for label, fn in metrics:
        values = []
        for r in reports:
            summary = r.get("summary", {}).get("overall", r.get("overall", {}))
            values.append(fn(summary))
        table.add_row(label, *values)

    console.print(table)

    # Save comparison
    comparison_data = {
        "runs": [
            {
                "run_id": r.get("run_id", "unknown"),
                "config_name": r.get("summary", {}).get("config_name", r.get("config_name", "")),
                "summary": r.get("summary", {}).get("overall", {}),
            }
            for r in reports
        ],
    }
    out_file = output / "comparison.json"
    out_file.write_text(json.dumps(comparison_data, indent=2))
    console.print(f"\n[green]Comparison saved to {out_file}[/]")


@main.command()
@click.argument("result", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "html", "markdown"]),
    default="html",
    help="Report format.",
)
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output path.")
def report(result: Path, fmt: str, output: Path | None) -> None:
    """Generate a formatted report from evaluation results."""
    from mmsafe.reporting.json_reporter import load_json_report

    data = load_json_report(result)

    # Reconstruct EvalRun from JSON (lightweight: just pass through summary)
    if output is None:
        suffix = {"json": ".json", "html": ".html", "markdown": ".md"}[fmt]
        output = result.with_suffix(suffix).with_stem(result.stem + "_report")

    if fmt == "json":
        # For JSON report from raw results, re-save with summary
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2, default=str))
        console.print(f"[green]JSON report written to {output}[/]")

    elif fmt == "html":
        _generate_report_from_data(data, output, "html")
        console.print(f"[green]HTML report written to {output}[/]")

    elif fmt == "markdown":
        _generate_report_from_data(data, output, "markdown")
        console.print(f"[green]Markdown report written to {output}[/]")


def _generate_report_from_data(data: dict[str, Any], output_path: Path, fmt: str) -> None:
    """Generate a report from pre-loaded JSON data."""
    summary = data.get("summary", data)

    if fmt == "html":
        from jinja2 import Environment, FileSystemLoader

        from mmsafe.reporting.charts import (
            create_asr_by_attack_chart,
            create_asr_by_category_chart,
            create_asr_by_model_chart,
            create_radar_chart,
            create_refusal_heatmap,
        )

        templates_dir = Path(__file__).parent / "reporting" / "templates"

        chart_category = ""
        chart_radar = ""
        chart_model = ""
        chart_attack = ""
        chart_refusal = ""

        if summary.get("by_category"):
            chart_category = create_asr_by_category_chart(summary["by_category"])
            chart_radar = create_radar_chart(summary["by_category"])
            chart_refusal = create_refusal_heatmap(summary["by_category"])
        if summary.get("by_model"):
            chart_model = create_asr_by_model_chart(summary["by_model"])
        if summary.get("by_attack"):
            chart_attack = create_asr_by_attack_chart(summary["by_attack"])

        env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=True)
        template = env.get_template("report.html.j2")

        html = template.render(
            summary=summary,
            version=__version__,
            chart_category=chart_category,
            chart_radar=chart_radar,
            chart_model=chart_model,
            chart_attack=chart_attack,
            chart_refusal=chart_refusal,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

    elif fmt == "markdown":
        overall = summary.get("overall", {})
        lines: list[str] = []
        _h = lines.append

        _h(f"# MMSAFE-Bench Safety Report: {summary.get('config_name', 'N/A')}")
        _h("")
        _h(f"**Run ID**: `{summary.get('run_id', 'N/A')}`  ")
        _h(f"**Total Samples**: {overall.get('total_samples', 0)}  ")
        _h("")
        _h("## Overall Metrics")
        _h("")
        _h("| Metric | Value |")
        _h("|--------|-------|")
        _h(f"| ASR | {overall.get('attack_success_rate', 0):.2%} |")
        _h(f"| Refusal Rate | {overall.get('refusal_rate', 0):.2%} |")
        _h(f"| FRR | {overall.get('false_refusal_rate', 0):.2%} |")
        _h("")
        _h("---")
        _h("*Generated by MMSAFE-Bench*")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n")


@main.command()
@click.argument("result", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="artifacts/leaderboard.json",
    help="Leaderboard output path.",
)
@click.option("--html", "html_out", type=click.Path(path_type=Path), help="HTML leaderboard path.")
def leaderboard(result: Path, output: Path, html_out: Path | None) -> None:
    """Update the model safety leaderboard with new results."""
    data = json.loads(result.read_text())
    summary = data.get("summary", data)

    # Build entries from summary data
    from mmsafe.reporting.leaderboard import LeaderboardEntry, save_leaderboard_json

    by_model = summary.get("by_model", {})
    entries = []
    for model, metrics in by_model.items():
        asr = metrics.get("asr", 0.0)
        entries.append(
            LeaderboardEntry(
                model=model,
                safety_score=round((1.0 - asr) * 100, 2),
                asr=asr,
                rr=metrics.get("rr", 0.0),
                samples=metrics.get("samples", 0),
            )
        )
    entries.sort(key=lambda e: e.safety_score, reverse=True)

    # Merge with existing if present
    from mmsafe.reporting.leaderboard import merge_leaderboard

    if output.exists():
        entries = merge_leaderboard(output, entries)

    save_leaderboard_json(entries, output)
    console.print(f"[green]Leaderboard saved to {output}[/] ({len(entries)} models)")

    if html_out:
        from mmsafe.reporting.leaderboard import generate_leaderboard_html

        generate_leaderboard_html(entries, html_out)
        console.print(f"[green]HTML leaderboard saved to {html_out}[/]")

    # Print to console
    table = Table(title="Safety Leaderboard", show_lines=True)
    table.add_column("Rank", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Safety Score")
    table.add_column("ASR")
    table.add_column("Samples")

    for i, e in enumerate(entries):
        if e.safety_score >= 75:
            score_style = "green"
        elif e.safety_score >= 50:
            score_style = "yellow"
        else:
            score_style = "red"
        table.add_row(
            str(i + 1),
            e.model,
            f"[{score_style}]{e.safety_score:.1f}[/]",
            f"{e.asr:.2%}",
            str(e.samples),
        )
    console.print(table)


@main.command()
def providers() -> None:
    """List available model providers and their capabilities."""
    from mmsafe.providers.registry import PROVIDER_REGISTRY

    table = Table(title="Available Model Providers", show_lines=True)
    table.add_column("Provider", style="bold cyan")
    table.add_column("Modalities")
    table.add_column("Models")
    table.add_column("Status")

    for name, info in PROVIDER_REGISTRY.items():
        modalities = ", ".join(m.value for m in info["modalities"])
        models = ", ".join(info["models"][:3])
        if len(info["models"]) > 3:
            models += f" (+{len(info['models']) - 3} more)"
        table.add_row(name, modalities, models, "[green]available[/]")

    console.print(table)


@main.command()
def attacks() -> None:
    """List available attack strategies."""
    from mmsafe.attacks.registry import ATTACK_REGISTRY

    table = Table(title="Available Attack Strategies", show_lines=True)
    table.add_column("Strategy", style="bold cyan")
    table.add_column("Description")
    table.add_column("Modalities")
    table.add_column("Multi-turn")

    for name, info in ATTACK_REGISTRY.items():
        modalities = ", ".join(m.value for m in info["modalities"])
        multi_turn = "yes" if info.get("multi_turn") else "no"
        table.add_row(name, info["description"], modalities, multi_turn)

    console.print(table)


@main.command()
@click.option("--modality", "-m", type=click.Choice(["text", "image", "video", "audio"]))
def taxonomy(modality: str | None) -> None:
    """Display the safety hazard taxonomy."""
    if modality:
        specs = get_hazards_for_modality(Modality(modality))
        title = f"Safety Hazards — {modality.upper()}"
    else:
        specs = list(HAZARD_TAXONOMY.values())
        title = "Safety Hazard Taxonomy (MLCommons + Extensions)"

    table = Table(title=title, show_lines=True)
    table.add_column("Code", style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Severity")
    table.add_column("Modalities")
    table.add_column("Aligned")

    severity_colors = {1: "green", 2: "yellow", 3: "orange1", 4: "red", 5: "bold red"}

    for spec in specs:
        sev_color = severity_colors.get(spec.severity.value, "white")
        modalities = ", ".join(m.value for m in spec.applicable_modalities)
        aligned = "MLCommons" if spec.mlcommons_aligned else "Custom"
        table.add_row(
            spec.category.value,
            spec.name,
            f"[{sev_color}]{spec.severity.name}[/]",
            modalities,
            aligned,
        )

    console.print(table)


@main.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to JSONL dataset file.",
)
def validate(dataset: Path) -> None:
    """Validate a prompt dataset JSONL file."""
    from mmsafe.datasets.loader import validate_dataset

    errors = validate_dataset(dataset)
    if errors:
        console.print(f"[red]Validation failed with {len(errors)} error(s):[/]")
        for err in errors:
            console.print(f"  - {err}")
        raise SystemExit(1)
    console.print(f"[green]Dataset {dataset.name} is valid.[/]")
