# Runtime Status - 2026-02-24 (Updated 2026-02-25 08:36 UTC)

## Scope
- LowResource-LLM-Forge: A100 restart loop stabilization + completion lock
- MMSAFE-Bench: policy-safe handoff after LowResource completion
- Operations: monitor/waiter reliability + Telegram readiness

## A100 LowResource Completion State
- Root cause confirmed:
  - periodic loop was watchdog-triggered (`action=restart_nan_limit_hit`), not stall-triggered.
  - event evidence from `training_watchdog_status_turkcell_7b_a100_v6_recovery_reset_opt.txt`.
- Training completion observed:
  - `2026-02-25T08:20:33Z`: `nan_guard_stopping_training` (limit reached at `step=990`).
  - `2026-02-25T08:20:34Z`: `training_complete` with final adapter output.
  - final artifacts verified at:
    - `/home/weezboo/projects/LowResource-LLM-Forge/artifacts/training/turkcell-7b-sft-v6-a100-bf16-recovery-reset-opt/final`
    - contains `adapter_model.safetensors`, `adapter_config.json`, `tokenizer_config.json`.

## Watchdog Stabilization Applied
- File patched on `a100`:
  - `/home/weezboo/projects/LowResource-LLM-Forge/scripts/training_watchdog.py`
- Added controls:
  - `WATCHDOG_ENABLE_STALL_RESTART=0`
  - `WATCHDOG_ENABLE_NAN_RESTART=1`
  - `WATCHDOG_NAN_CONSECUTIVE_LIMIT=5`
  - `WATCHDOG_EVENTS_LOG=artifacts/logs/training_watchdog_events_turkcell_7b_a100_v6_recovery_reset_opt.jsonl`
- Added append-only event log:
  - records `restart_decision` and `restart_result` with reason, step, nan count, stalled seconds.
- Added completion lock behavior:
  - when final artifacts are present, watchdog stops training service and does not auto-start again.
  - status evidence:
    - `action=final_artifacts_present_no_start`
    - `forge-training.service` remains `failed/inactive` (not cycling).

## A100 Policy Handoff (LowResource > MMSAFE)
- `mmsafe-waiter.service` remained active and respected priority while training ran.
- After training stopped, waiter auto-triggered A100 eval:
  - `2026-02-25 08:24:55` and `2026-02-25 08:27:42` launches recorded in waiter log.
- A100 eval resume behavior:
  - resumed from existing checkpoints (`completed=1852`) and produced new run files with `Samples evaluated: 0`.
  - latest A100 runs:
    - `eval-e4ae1842347f`
    - `eval-e9481b529787`
  - both have HTML + Markdown reports.
- Waiter resume fix applied (`2026-02-25 08:33 UTC`):
  - scripts updated: `scripts/wait_and_run_full_eval.sh`, `scripts/ops/run_full_eval_v100.sh`.
  - new rule: only resume from checkpoints **without** matching `*_results.json` (unfinished runs).
  - completed-run checkpoints are now skipped to prevent zero-sample replays.
  - post-fix launch evidence: waiter resumed from `checkpoint_eval-b81a7d5fb9a9.json` (`completed=300`) and started active A100 eval.

## V100 Baseline (Reference)
- canonical clean baseline remains:
  - `run_id=eval-3af0c5f63657`
  - `total_samples=426`
  - `attack_success_rate=0.0681`
  - `refusal_rate=0.0211`
- artifacts under:
  - `/home/weezboo/projects/MMSAFE-Bench/artifacts/full_eval_v100_clean_20260224`

## Notifications / Secrets
- host env files verified (`a100`, `v100`, `spark`):
  - `/home/weezboo/.mmsafe.env` and `/home/weezboo/.notify.env` with `600` perms.
- Telegram notify script operational on all hosts:
  - `/home/weezboo/projects/MMSAFE-Bench/scripts/ops/notify_telegram.sh`

## Acceptance Snapshot
- Restart-loop root cause identified and instrumented with event logs: **Done**
- LowResource final adapter produced: **Done**
- Post-completion re-run loop blocked by watchdog completion lock: **Done**
- Waiter policy preserved and auto-handoff triggered: **Done**
- V100 baseline integrity retained: **Done**

## Remaining Risk / Next Refinement
- A100 waiter resume-to-completed issue is mitigated by checkpoint/result pairing filter.
- Optional next refinement: move A100 post-training runs to a dedicated output directory per cycle for cleaner lineage separation.
