#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/mmsafe/config/defaults/full_eval_a100_ready.yaml}"

POLL_SECONDS="${A100_POLL_SECONDS:-300}"
MEM_THRESHOLD_MB="${A100_IDLE_MEM_MB:-2048}"
UTIL_THRESHOLD_PCT="${A100_IDLE_UTIL_PCT:-15}"
FORGE_TRAINING_PATTERN="${FORGE_TRAINING_PATTERN:-scripts/run_training.py}"
FORGE_TRAINING_CWD_HINT="${FORGE_TRAINING_CWD_HINT:-LowResource-LLM-Forge}"
UV_BIN="${UV_BIN:-uv}"
MMSAFE_ENV_FILE="${MMSAFE_ENV_FILE:-$HOME/.mmsafe.env}"
NOTIFY_SCRIPT="${NOTIFY_SCRIPT:-$ROOT_DIR/scripts/ops/notify_telegram.sh}"
EVAL_RETRY_SECONDS="${EVAL_RETRY_SECONDS:-120}"
MMSAFE_OUTPUT_DIR="${MMSAFE_OUTPUT_DIR:-$ROOT_DIR/artifacts/full_eval_a100_ready}"

LOG_DIR="$ROOT_DIR/artifacts/logs"
mkdir -p "$LOG_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Cannot detect A100 availability." >&2
  exit 1
fi

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
  else
    echo "uv not found. Set UV_BIN or install uv in PATH." >&2
    exit 1
  fi
fi

# Optional runtime secrets/config file (e.g. REPLICATE_API_TOKEN).
if [[ -f "$MMSAFE_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$MMSAFE_ENV_FILE"
  set +a
fi

notify() {
  local level="$1"
  shift

  if [[ -x "$NOTIFY_SCRIPT" ]]; then
    "$NOTIFY_SCRIPT" "$level" "$*" || true
  fi
}

has_idle_a100() {
  local rows
  rows="$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "$rows" ]]; then
    return 1
  fi

  while IFS=',' read -r raw_name raw_mem raw_util; do
    local name mem util
    name="$(echo "$raw_name" | xargs)"
    mem="$(echo "$raw_mem" | xargs)"
    util="$(echo "$raw_util" | xargs)"

    [[ "$name" == *A100* ]] || continue
    [[ "$mem" =~ ^[0-9]+$ ]] || continue
    [[ "$util" =~ ^[0-9]+$ ]] || continue

    if (( mem < MEM_THRESHOLD_MB && util <= UTIL_THRESHOLD_PCT )); then
      return 0
    fi
  done <<< "$rows"

  return 1
}

forge_training_active() {
  if ! command -v pgrep >/dev/null 2>&1; then
    return 1
  fi

  local matches
  matches="$(pgrep -af "$FORGE_TRAINING_PATTERN" || true)"
  if [[ -z "$matches" ]]; then
    return 1
  fi

  if [[ -n "$FORGE_TRAINING_CWD_HINT" ]] && echo "$matches" | grep -Fq "$FORGE_TRAINING_CWD_HINT"; then
    return 0
  fi

  if echo "$matches" | grep -Eq 'python([0-9.]*)? .*run_training\.py'; then
    return 0
  fi

  return 1
}

latest_checkpoint_file() {
  ls -1t "$MMSAFE_OUTPUT_DIR"/checkpoints/checkpoint_eval-*.json 2>/dev/null | head -n1 || true
}

checkpoint_completed_count() {
  local checkpoint_path="$1"
  python3 - "$checkpoint_path" <<'PY' 2>/dev/null || echo "0"
import json
import sys

path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    print(0)
    raise SystemExit

count = data.get("completed_count")
if isinstance(count, int):
    print(count)
else:
    completed_ids = data.get("completed_sample_ids") or []
    print(len(completed_ids))
PY
}

best_checkpoint_file() {
  local best_file="" best_count=-1 best_mtime=-1
  local file count mtime

  shopt -s nullglob
  for file in "$MMSAFE_OUTPUT_DIR"/checkpoints/checkpoint_eval-*.json; do
    count="$(checkpoint_completed_count "$file")"
    if [[ ! "$count" =~ ^[0-9]+$ ]]; then
      count=0
    fi
    mtime="$(stat -c %Y "$file" 2>/dev/null || echo 0)"

    if (( count > best_count )) || (( count == best_count && mtime > best_mtime )); then
      best_file="$file"
      best_count="$count"
      best_mtime="$mtime"
    fi
  done
  shopt -u nullglob

  echo "$best_file"
}

latest_results_file() {
  ls -1t "$MMSAFE_OUTPUT_DIR"/*_results.json 2>/dev/null | head -n1 || true
}

format_results_summary() {
  local results_file="$1"

  if ! command -v python3 >/dev/null 2>&1; then
    return 0
  fi

  python3 - "$results_file" <<'PY' || true
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)

summary = data.get("summary", {})
overall = summary.get("overall", {})
metadata = data.get("metadata", {})
unavailable = metadata.get("unavailable_providers", [])

fields = [
    f"run_id={data.get('run_id', 'n/a')}",
    f"total_samples={overall.get('total_samples', data.get('metadata', {}).get('total_samples', 'n/a'))}",
    f"asr={overall.get('attack_success_rate', 'n/a')}",
    f"rr={overall.get('refusal_rate', 'n/a')}",
    f"unavailable_providers={','.join(unavailable) if unavailable else 'none'}",
]
print(" | ".join(fields))
PY
}

generate_reports() {
  local results_file="$1"
  local html_report markdown_report rc=0

  html_report="${results_file%.json}_report.html"
  markdown_report="${results_file%.json}_report.md"

  if ! "$UV_BIN" run mmsafe report "$results_file" --format html >/dev/null 2>&1; then
    rc=1
  fi
  if ! "$UV_BIN" run mmsafe report "$results_file" --format markdown >/dev/null 2>&1; then
    rc=1
  fi

  if [[ "$rc" -eq 0 ]]; then
    notify info "MMSAFE reports generated: html=${html_report} markdown=${markdown_report}"
  else
    notify warn "MMSAFE eval succeeded but report generation had errors for ${results_file}"
  fi
}

run_eval_once() {
  local run_log checkpoint_file results_file summary
  local -a cmd resume_args

  run_log="$LOG_DIR/full_eval_a100_$(date +%Y%m%d_%H%M%S).log"
  checkpoint_file="$(best_checkpoint_file)"
  resume_args=()
  if [[ -n "$checkpoint_file" ]]; then
    resume_args=(--resume "$checkpoint_file")
  fi

  cmd=("$UV_BIN" run mmsafe run --config "$CONFIG_PATH" --execution-profile a100)
  if [[ "${#resume_args[@]}" -gt 0 ]]; then
    cmd+=("${resume_args[@]}")
  fi

  if [[ -n "$checkpoint_file" ]]; then
    notify info "MMSAFE full eval started on $(hostname) (resume=${checkpoint_file})."
  else
    notify info "MMSAFE full eval started on $(hostname) (fresh run)."
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting evaluation. Log: $run_log"
  set +e
  set -o pipefail
  "${cmd[@]}" 2>&1 | tee "$run_log"
  local rc="${PIPESTATUS[0]}"
  set -e

  if [[ "$rc" -eq 0 ]]; then
    results_file="$(latest_results_file)"
    if [[ -n "$results_file" ]]; then
      summary="$(format_results_summary "$results_file")"
      notify info "MMSAFE eval completed. ${summary} | log=${run_log} | results=${results_file}"
      generate_reports "$results_file"
    else
      notify info "MMSAFE eval completed without results file. log=${run_log}"
    fi
  else
    notify critical "MMSAFE eval failed with exit_code=${rc}. log=${run_log}"
  fi

  return "$rc"
}

echo "Waiting for idle A100 GPU..."
echo "Config: $CONFIG_PATH"
echo "Poll interval: ${POLL_SECONDS}s"
echo "Idle threshold: mem<${MEM_THRESHOLD_MB}MB and util<=${UTIL_THRESHOLD_PCT}%"
echo "Priority gate: wait while LLM-Forge training is active (${FORGE_TRAINING_PATTERN})"
echo "Output directory: ${MMSAFE_OUTPUT_DIR}"
notify info "MMSAFE waiter active on $(hostname). Waiting for LLM-Forge to become idle."

eval_completed_for_cycle=0
while true; do
  if forge_training_active; then
    eval_completed_for_cycle=0
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] LLM-Forge training active. Waiting ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
    continue
  fi

  if [[ "$eval_completed_for_cycle" -eq 1 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Eval already completed for this idle cycle. Waiting ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
    continue
  fi

  if ! has_idle_a100; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] A100 busy. Rechecking in ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
    continue
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Idle A100 detected. Launching full evaluation."
  if run_eval_once; then
    eval_completed_for_cycle=1
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation failed. Retrying in ${EVAL_RETRY_SECONDS}s."
    sleep "$EVAL_RETRY_SECONDS"
  fi
done
