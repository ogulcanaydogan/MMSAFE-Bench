# GPU Allocation Policy

## Priority Order

1. `LowResource-LLM-Forge` -> **PRIORITY** (training)
2. `MMSAFE-Bench` -> **SECOND** (evaluations only when training is idle)
3. `LLM-Edge-Benchmark` -> **LOW** (baseline runs only)
4. `VAOL` -> **NONE** (CPU-bound; no GPU allocation)

## MMSAFE-Bench Enforcement

- `scripts/wait_and_run_full_eval.sh` waits for:
  - no active Forge training process (`FORGE_TRAINING_PATTERN` match),
  - at least one idle A100 GPU.
- Only then it starts full multimodal evaluation.
- If `REPLICATE_API_TOKEN` exists in `/home/weezboo/.mmsafe.env`, Replicate-backed
  multimodal providers are enabled.
- Telegram notifications are sent when `.notify.env` provides
  `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

## Operational Notes

- Default Forge process pattern:
  - `scripts/run_training.py`
- Default Forge cwd hint:
  - `LowResource-LLM-Forge`
- You can override detection pattern if your training launcher path changes:
  - `export FORGE_TRAINING_PATTERN="your/new/pattern"`
- Systemd user units:
  - `deploy/systemd/mmsafe-waiter.service`
  - `deploy/systemd/lowresource-monitor.service`
