# Runtime Status - 2026-02-24 (Updated 23:54 UTC)

## Scope
- LowResource-LLM-Forge: stable resume continuation on `a100`
- MMSAFE-Bench: clean full eval baseline lock on `v100`
- Operations: waiter policy guard + Telegram notifications (`a100`, `v100`, `spark`)

## A100 Training State (LowResource)
- Host: `a100`
- Services:
  - `forge-training.service`: `active (running)`
  - `forge-training-watchdog.service`: `active (running)`
  - `lowresource-monitor.service`: `active (running)` (restarted at `2026-02-24 23:53 UTC` after observability fix rollout)
- Effective training command:
  - `uv run python scripts/run_training.py --config configs/models/turkcell_7b_a100_v6_recovery_reset_opt.yaml --resume-from artifacts/training/turkcell-7b-sft-v6-a100-bf16-recovery-reset-opt/checkpoint-750`
- Canonical telemetry file:
  - `/home/weezboo/projects/LowResource-LLM-Forge/artifacts/logs/training_monitor_status_a100.txt`
- Snapshot (`2026-02-24T23:54:22Z`):
  - `step=856/8601`
  - `percent=9`
  - `eta_utc=2026-02-25T12:04:45Z`
  - `gpu=100 %, 60486 MiB, 81920 MiB`
  - `state=running`
- Checkpoints currently present:
  - `checkpoint-250`
  - `checkpoint-500`
  - `checkpoint-750`

## A100 Observability Stabilization
- Root cause of stale perception:
  - old file `training_watchdog_status_a100_v4.txt` remained in logs and looked stale.
  - live telemetry is in `training_monitor_status_a100.txt` and variant-specific watchdog files.
- Fix deployed:
  - `scripts/ops/monitor_lowresource.sh` now auto-selects freshest `training_monitor_status*.txt`.
  - Adds `status_file` and `status_age` to each monitor heartbeat log line.
  - Adds stale-status warning path (`STATUS_STALE_SECONDS`, default `180s`) and suppresses false stall alerts while telemetry is stale.
- Verification:
  - new heartbeat line includes `status_file=...training_monitor_status_a100.txt status_age=25s`.

## A100 Policy Guard (MMSAFE Waiter)
- `mmsafe-waiter.service`: `active (running)`
- Waiter log continuously shows:
  - `LLM-Forge training active. Waiting 300s...`
- Policy status: `LowResource-LLM-Forge > MMSAFE-Bench` enforced (no A100 eval start while training is alive).

## V100 Baseline Lock (MMSAFE Clean Full Eval)
- Host: `v100`
- Service: `mmsafe-v100-eval.service` is `enabled` and currently `inactive` (oneshot completed).
- Baseline run:
  - `run_id=eval-3af0c5f63657`
  - `total_samples=426`
  - `attack_success_rate=0.0681`
  - `refusal_rate=0.0211`
  - `unavailable_providers=local_ollama`
- Artifact directory:
  - `/home/weezboo/projects/MMSAFE-Bench/artifacts/full_eval_v100_clean_20260224`
- Verified files:
  - `eval-3af0c5f63657_results.json`
  - `eval-3af0c5f63657_results_report.html`
  - `eval-3af0c5f63657_results_report.md`
- SHA256:
  - JSON: `5675c42fe2c6d770f32aadb950613c1ed49e7267963ee9cb6a7a42e796cae5d1`
  - HTML: `1224d5b0cab368dc57e05aa572e5b6d28173d135d87ec8e321471ea8c03163b4`
  - Markdown: `f9502af8b7e58ff9fb08826f56cb54c8cf4d339bc7046a92935e545993ad198d`

## Notifications and Host Readiness
- Secrets posture verified:
  - `/home/weezboo/.mmsafe.env` and `/home/weezboo/.notify.env` present with `600` perms on hosts.
- Healthcheck notification command executed from:
  - `a100`
  - `v100`
  - `spark` (`100.80.116.20`)
- `spark` standby check:
  - notifier script exists at `/home/weezboo/projects/MMSAFE-Bench/scripts/ops/notify_telegram.sh`.

## Remaining Completion Criteria
- LowResource training reaches completion and writes final adapter artifacts.
- A100 waiter auto-starts MMSAFE eval after training process exits.
- Final post-training summary is published with:
  - `run_id`
  - `total_samples`
  - `attack_success_rate`
  - `refusal_rate`
  - `unavailable_providers`
  - host used (`a100` or `v100`)
