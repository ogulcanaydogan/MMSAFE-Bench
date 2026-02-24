# Runtime Status - 2026-02-24

## Scope
- LowResource-LLM-Forge: stabilize resume on `a100`
- MMSAFE-Bench: clean full eval on `v100`
- Notifications: Telegram on `a100`, `v100`, `spark`

## Baseline (Before Clean V100 Run)
- Host: `a100`
- Canonical result: `eval-06515569babd_results.json`
- `total_samples`: `602`
- `attack_success_rate`: `0.0748`
- `refusal_rate`: `0.1013`

## Current Execution State
- LowResource training host: `a100`
- Training service: `forge-training.service` (`active`)
- Resume mode: enabled via `ENABLE_RESUME=1`
- Resume source: `/home/weezboo/projects/LowResource-LLM-Forge/artifacts/training/turkcell-7b-sft-v3-a100-bf16-stable/checkpoint-500`
- Current step snapshot: `803/8601` (2026-02-24 08:59 UTC)
- Effective command includes:
  - `--config configs/models/turkcell_7b_a100_v4_recovery.yaml`
  - `--resume-from .../checkpoint-500`

- MMSAFE clean eval host: `v100`
- Output directory: `artifacts/full_eval_v100_clean_20260224`
- Active run log: `/home/weezboo/projects/MMSAFE-Bench/artifacts/logs/full_eval_v100_20260224_072959.log`
- Latest checkpoint file: `checkpoint_eval-9ffcbab36406.json`
- Latest checkpoint completed count (snapshot): `850` (2026-02-24 07:54 UTC)

## Notification and Secrets Posture
- Updated on all hosts (`a100`, `v100`, `spark`):
  - `/home/weezboo/.mmsafe.env` with rotated `REPLICATE_API_TOKEN`
  - `/home/weezboo/.notify.env` with `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
- File permission enforced:
  - `chmod 600 /home/weezboo/.mmsafe.env`
  - `chmod 600 /home/weezboo/.notify.env`
- API send checks returned Telegram `ok=true` payloads from all three hosts.

## Pending Finalization
- Wait for `v100` clean eval completion and generated files:
  - `*_results.json`
  - `*_report.html`
  - `*_report.md`
- Wait for `a100` resumed training progression checkpoint:
  - pass milestone `step > 800`
  - verify no `nan_guard_stopping_training` recurrence in early resumed window
