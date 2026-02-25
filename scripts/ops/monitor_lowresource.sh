#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NOTIFY_SCRIPT="${NOTIFY_SCRIPT:-$ROOT_DIR/scripts/ops/notify_telegram.sh}"
LOWRESOURCE_ROOT="${LOWRESOURCE_ROOT:-/home/weezboo/projects/LowResource-LLM-Forge}"
TRAINING_PATTERN="${TRAINING_PATTERN:-LowResource-LLM-Forge/.venv/bin/python3 scripts/run_training.py}"
STATUS_FILE="${STATUS_FILE:-$LOWRESOURCE_ROOT/artifacts/logs/training_monitor_status_a100.txt}"
STATUS_FILE_GLOB="${STATUS_FILE_GLOB:-$LOWRESOURCE_ROOT/artifacts/logs/training_monitor_status*.txt}"
STATUS_STALE_SECONDS="${STATUS_STALE_SECONDS:-180}"
RESET_DROP_THRESHOLD="${RESET_DROP_THRESHOLD:-100}"
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
  local status_path="$3"

  if [[ ! -f "$status_path" ]]; then
    echo "$fallback"
    return 0
  fi

  local value
  value="$(awk -F= -v k="$key" '$1 == k {print $2}' "$status_path" | tail -n1)"
  if [[ -z "$value" ]]; then
    echo "$fallback"
  else
    echo "$value"
  fi
}

status_mtime() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo 0
    return 0
  fi
  stat -c %Y "$path" 2>/dev/null || echo 0
}

resolve_status_file() {
  local preferred="$STATUS_FILE"
  local freshest_file=""
  local freshest_mtime=0
  local path mtime preferred_mtime

  shopt -s nullglob
  for path in $STATUS_FILE_GLOB; do
    [[ -f "$path" ]] || continue
    mtime="$(status_mtime "$path")"
    if [[ "$mtime" =~ ^[0-9]+$ ]] && (( mtime > freshest_mtime )); then
      freshest_file="$path"
      freshest_mtime="$mtime"
    fi
  done
  shopt -u nullglob

  if [[ -n "$preferred" ]] && [[ -f "$preferred" ]]; then
    preferred_mtime="$(status_mtime "$preferred")"
    if [[ ! "$preferred_mtime" =~ ^[0-9]+$ ]]; then
      preferred_mtime=0
    fi
    if [[ -z "$freshest_file" ]] || (( preferred_mtime >= freshest_mtime )); then
      echo "$preferred"
      return 0
    fi
  fi

  if [[ -n "$freshest_file" ]]; then
    echo "$freshest_file"
    return 0
  fi

  echo "$preferred"
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
sent_status_stale=0

while true; do
  now_ts="$(date +%s)"
  now_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  status_file="$(resolve_status_file)"
  status_mtime_ts="$(status_mtime "$status_file")"
  status_age=-1
  if [[ "$status_mtime_ts" =~ ^[0-9]+$ ]] && (( status_mtime_ts > 0 )); then
    status_age=$(( now_ts - status_mtime_ts ))
  fi

  is_active=0
  if training_active; then
    is_active=1
  fi

  step="$(read_status_value step -1 "$status_file")"
  target_steps="$(read_status_value target_steps -1 "$status_file")"
  gpu_line="$(read_status_value gpu unknown "$status_file")"
  state_line="$(read_status_value state unknown "$status_file")"
  status_stale=0
  if (( status_age >= 0 )) && (( status_age >= STATUS_STALE_SECONDS )); then
    status_stale=1
  fi

  if [[ "$step" =~ ^[0-9]+$ ]]; then
    if (( step > last_step )); then
      last_step="$step"
      last_progress_ts="$now_ts"
      sent_stall=0
    elif (( is_active == 1 )) && (( last_step >= 0 )) && (( step + RESET_DROP_THRESHOLD < last_step )); then
      previous_step="$last_step"
      last_step="$step"
      last_progress_ts="$now_ts"
      sent_stall=0
      notify info "LowResource step reset detected (${previous_step} -> ${step}); monitoring continues from resumed step."
    fi
  fi

  if (( is_active == 1 )); then
    sent_crash=0

    if (( status_stale == 1 )); then
      if (( sent_status_stale == 0 )); then
        notify warn "LowResource status file is stale (${status_age}s): ${status_file}"
        sent_status_stale=1
      fi
    else
      sent_status_stale=0
    fi

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
      if (( status_stale == 0 )) && (( stalled_for >= STALL_SECONDS )) && (( sent_stall == 0 )); then
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
    sent_status_stale=0
  fi

  echo "[$now_utc] running=${is_active} step=${step} target=${target_steps} stalled_for=$(( now_ts - last_progress_ts ))s status_file=${status_file} status_age=${status_age}s state=${state_line} gpu=\"${gpu_line}\""
  sleep "$POLL_SECONDS"
done
