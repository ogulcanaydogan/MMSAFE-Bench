# MMSAFE-Bench

[![CI](https://github.com/ogulcanaydogan/MMSAFE-Bench/actions/workflows/ci.yml/badge.svg)](https://github.com/ogulcanaydogan/MMSAFE-Bench/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Multi-Modal AI Safety Evaluation Framework** for red-teaming and benchmarking generative AI models across text, image, video, and audio from a single CLI.

```
Prompt Datasets (JSONL) → Attack Strategies → Model Providers → Safety Judges → Metrics → Reports
```

## Why MMSAFE-Bench?

Existing safety benchmarks are fragmented: MM-SafetyBench covers image-text, USB covers text-only, Video-SafetyBench handles video in isolation. None work as a production CLI tool with both proprietary and open-source model support.

MMSAFE-Bench unifies safety evaluation across **all four generative modalities** with:

- **20 hazard categories** — MLCommons AILuminate S1-S12 + 8 multi-modal extensions (deepfakes, voice impersonation, cross-modal bypass, etc.)
- **9 attack strategies** — jailbreaks, encoding tricks, role-play, multi-turn escalation, adversarial suffixes, cross-modal injection, low-resource translation
- **8 model providers** — OpenAI, Anthropic, Google, Replicate, ElevenLabs, local vLLM, local Ollama, deterministic stub
- **6 safety judges** — keyword, LLM-as-judge, toxicity, NSFW classifier, composite ensemble, human evaluation export
- **Edge simulation** — test safety degradation on constrained hardware (DGX Spark, Jetson, Raspberry Pi, V100)
- **Interactive reports** — HTML dashboards with Plotly charts, Markdown tables, JSON exports, model leaderboards

## Quick Start

```bash
# Install with all dependencies
uv sync --extra dev --extra viz --extra providers

# Browse the safety taxonomy
mmsafe taxonomy

# Validate a dataset
mmsafe validate --dataset datasets/text/mlcommons_hazards.jsonl

# Dry-run an evaluation
mmsafe run --config mmsafe/config/defaults/text_eval.yaml --dry-run

# List available providers and attack strategies
mmsafe providers
mmsafe attacks
```

## Architecture

```
mmsafe/
├── config/          # Pydantic config models + YAML defaults
├── taxonomy/        # MLCommons S1-S12 + custom X1-X8 hazard categories
├── datasets/        # JSONL prompt dataset loader + validation
├── attacks/         # 9 red-teaming strategies (passthrough → composite)
├── providers/       # 8 model provider adapters (cloud + local)
├── judges/          # 6 safety evaluation judges
├── pipeline/        # Async evaluation runner + checkpointing
├── metrics/         # ASR, RR, FRR, NSFW rate + bootstrap CI
├── reporting/       # HTML/JSON/Markdown reports + Plotly charts + leaderboard
├── edge/            # Edge deployment simulation (5 device profiles)
└── cli.py           # Click CLI with 8 commands
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `mmsafe run` | Run a safety evaluation from YAML config |
| `mmsafe report` | Generate HTML/JSON/Markdown report from results |
| `mmsafe compare` | Compare multiple evaluation runs side-by-side |
| `mmsafe leaderboard` | Generate model safety leaderboard |
| `mmsafe providers` | List available model providers |
| `mmsafe attacks` | List attack strategies |
| `mmsafe taxonomy` | Display the safety hazard taxonomy |
| `mmsafe validate` | Validate a JSONL prompt dataset |

## GPU-Aware Execution Profiles

The runner supports `execution.profile` with safe fallback behavior:

- `auto`: detect available GPUs and fall back to `small_gpu` when A100 is unavailable
- `small_gpu`: conservative concurrency for smaller GPUs
- `a100`: A100-optimized concurrency

```bash
mmsafe run --config mmsafe/config/defaults/full_eval.yaml --execution-profile auto
mmsafe run --config mmsafe/config/defaults/full_eval.yaml --no-auto-tune
```

## Configuration

Evaluations are configured via YAML files. See `mmsafe/config/defaults/` for examples:

- `text_eval.yaml` — Text-only evaluation with GPT-4o
- `image_eval.yaml` — Image generation safety (DALL-E, FLUX)
- `video_eval.yaml` — Video generation (Kling, Veo)
- `audio_eval.yaml` — Audio/speech safety (ElevenLabs, Bark)
- `full_eval.yaml` — All modalities combined
- `edge_simulation.yaml` — Edge deployment constraints

## Metrics

| Metric | Description |
|--------|-------------|
| **ASR** | Attack Success Rate — fraction of attacks producing unsafe content |
| **RR** | Refusal Rate — fraction of prompts refused by the model |
| **FRR** | False Refusal Rate — fraction of benign prompts incorrectly refused |
| **NSFW Rate** | Fraction of image/video outputs flagged as NSFW |

All metrics include 95% bootstrap confidence intervals, broken down by category, attack, modality, and model.

## Local Provider Setup

```bash
# vLLM backend
export VLLM_BASE_URL="http://localhost:8000"

# Ollama backend
export OLLAMA_BASE_URL="http://localhost:11434"
```

Cloud providers are optional; unavailable providers are skipped by default unless `execution.strict_provider_init: true`.

## A100 Automation

For production-style orchestration (LowResource priority, MMSAFE auto handoff, Telegram alerts, systemd services), use:

- `docs/A100_AUTOMATION_RUNBOOK.md`

## Development

```bash
make install       # Install with dev deps
make test          # Run tests (80% coverage gate)
make lint          # Ruff + mypy
make fmt           # Auto-format
make eval-smoke    # Smoke test with stub provider
make clean         # Remove build artifacts
```

## Docker

```bash
docker build -t mmsafe .
docker run --rm mmsafe --help
docker run --rm mmsafe taxonomy
```

## License

Apache-2.0
