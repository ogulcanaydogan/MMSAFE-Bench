#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NOTIFY_SCRIPT="${NOTIFY_SCRIPT:-$ROOT_DIR/scripts/ops/notify_telegram.sh}"
LOWRESOURCE_ROOT="${LOWRESOURCE_ROOT:-/home/weezboo/projects/LowResource-LLM-Forge}"
TRAINING_PATTERN="${TRAINING_PATTERN:-LowResource-LLM-Forge/.venv/bin/python3 scripts/run_training.py}"
STATUS_FILE="${STATUS_FILE:-$LOWRESOURCE_ROOT/artifacts/logs/training_monitor_status_a100.txt}"
MONITOR_LOG="${MONITOR_LOG:-$LOWRESOURCE_ROOT/artifacts/logs/monitor_a100_training.log}"
POLL_SECONDS="${MONITOR_POLL_SECONDS:-120}"
STALL_SECONDS="${MONITOR_STALL_SECONDS:-1800}"

notify() {
  local level="$1"
  shift

  if [[ -x "$NOTIFY_SCRIPT" ]]; then
    "$NOTIFY_SCRIPT" "$level" "$*" || true
  fi
}

training_active() {
  if ! command -v pgrep >/dev/null 2>&1; then
    return 1
  fi

  local matches
  matches="$(pgrep -af "$TRAINING_PATTERN" || true)"
  if [[ -z "$matches" ]]; then
    return 1
  fi

  echo "$matches" | grep -Eq 'python([0-9.]*)? .*run_training\.py'
}

read_status_value() {
  local key="$1"
  local fallback="$2"

  if [[ ! -f "$STATUS_FILE" ]]; then
    echo "$fallback"
    return 0
  fi

  local value
  value="$(awk -F= -v k="$key" '$1 == k {print $2}' "$STATUS_FILE" | tail -n1)"
  if [[ -z "$value" ]]; then
    echo "$fallback"
  else
    echo "$value"
  fi
}

final_artifact_exists() {
  if [[ ! -d "$LOWRESOURCE_ROOT/artifacts/training" ]]; then
    return 1
  fi

  find "$LOWRESOURCE_ROOT/artifacts/training" -maxdepth 5 -type f \
    \( -name "adapter_config.json" -o -name "adapter_model.safetensors" -o -name "tokenizer_config.json" \) \
    | grep -q "/final/"
}

mkdir -p "$(dirname "$MONITOR_LOG")"
touch "$MONITOR_LOG"
exec >>"$MONITOR_LOG" 2>&1

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] monitor started: poll=${POLL_SECONDS}s stall=${STALL_SECONDS}s"
notify info "LowResource monitor started on $(hostname)."

last_step=-1
last_progress_ts="$(date +%s)"
sent_stall=0
sent_crash=0
sent_complete=0

while true; do
  now_ts="$(date +%s)"
  now_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  is_active=0
  if training_active; then
    is_active=1
  fi

  step="$(read_status_value step -1)"
  target_steps="$(read_status_value target_steps -1)"
  gpu_line="$(read_status_value gpu unknown)"
  state_line="$(read_status_value state unknown)"

  if [[ "$step" =~ ^[0-9]+$ ]] && (( step > last_step )); then
    last_step="$step"
    last_progress_ts="$now_ts"
    sent_stall=0
  fi

  if (( is_active == 1 )); then
    sent_crash=0

    if [[ "$target_steps" =~ ^[0-9]+$ ]] && [[ "$step" =~ ^[0-9]+$ ]] && (( target_steps > 0 )) && (( step >= target_steps )); then
      if (( sent_complete == 0 )); then
        notify info "LowResource training complete: step=${step}/${target_steps}."
        sent_complete=1
      fi
    elif final_artifact_exists; then
      if (( sent_complete == 0 )); then
        notify info "LowResource training complete: final adapter artifacts detected."
        sent_complete=1
      fi
    else
      sent_complete=0
      stalled_for=$(( now_ts - last_progress_ts ))
      if (( stalled_for >= STALL_SECONDS )) && (( sent_stall == 0 )); then
        notify warn "LowResource training appears stalled for ${stalled_for}s (last_step=${last_step})."
        sent_stall=1
      fi
    fi
  else
    if final_artifact_exists; then
      if (( sent_complete == 0 )); then
        notify info "LowResource training process ended and final artifacts exist."
        sent_complete=1
      fi
      sent_crash=0
    else
      sent_complete=0
      if (( sent_crash == 0 )); then
        notify critical "LowResource training process is down and no final adapter artifacts were found."
        sent_crash=1
      fi
    fi
  fi

  echo "[$now_utc] running=${is_active} step=${step} target=${target_steps} stalled_for=$(( now_ts - last_progress_ts ))s state=${state_line} gpu=\"${gpu_line}\""
  sleep "$POLL_SECONDS"
done
