#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/mmsafe/config/defaults/full_eval_a100_ready.yaml}"
EXECUTION_PROFILE="${EXECUTION_PROFILE:-small_gpu}"
MMSAFE_ENV_FILE="${MMSAFE_ENV_FILE:-$HOME/.mmsafe.env}"
UV_BIN="${UV_BIN:-uv}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/artifacts/full_eval_a100_ready}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/artifacts/logs}"
NOTIFY_SCRIPT="${NOTIFY_SCRIPT:-$ROOT_DIR/scripts/ops/notify_telegram.sh}"

mkdir -p "$LOG_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

if [[ -f "$MMSAFE_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$MMSAFE_ENV_FILE"
  set +a
fi

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
  else
    echo "uv not found. Set UV_BIN or install uv in PATH." >&2
    exit 1
  fi
fi

notify() {
  local level="$1"
  shift
  if [[ -x "$NOTIFY_SCRIPT" ]]; then
    "$NOTIFY_SCRIPT" "$level" "$*" || true
  fi
}

latest_checkpoint_file() {
  ls -1t "$OUTPUT_DIR"/checkpoints/checkpoint_eval-*.json 2>/dev/null | head -n1 || true
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
  for file in "$OUTPUT_DIR"/checkpoints/checkpoint_eval-*.json; do
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
  ls -1t "$OUTPUT_DIR"/*_results.json 2>/dev/null | head -n1 || true
}

format_results_summary() {
  local results_file="$1"
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
    f"total_samples={overall.get('total_samples', metadata.get('total_samples', 'n/a'))}",
    f"asr={overall.get('attack_success_rate', 'n/a')}",
    f"rr={overall.get('refusal_rate', 'n/a')}",
    f"unavailable_providers={','.join(unavailable) if unavailable else 'none'}",
]
print(" | ".join(fields))
PY
}

generate_reports() {
  local results_file="$1"
  if "$UV_BIN" run mmsafe report "$results_file" --format html >/dev/null 2>&1 \
    && "$UV_BIN" run mmsafe report "$results_file" --format markdown >/dev/null 2>&1; then
    notify info "V100 reports generated for ${results_file}"
  else
    notify warn "V100 eval succeeded but report generation had errors for ${results_file}"
  fi
}

run_log="$LOG_DIR/full_eval_v100_$(date +%Y%m%d_%H%M%S).log"
checkpoint_file="$(best_checkpoint_file)"
cmd=("$UV_BIN" run mmsafe run --config "$CONFIG_PATH" --execution-profile "$EXECUTION_PROFILE")
if [[ -n "$checkpoint_file" ]]; then
  cmd+=(--resume "$checkpoint_file")
  notify info "V100 full eval started (resume=${checkpoint_file})"
else
  notify info "V100 full eval started (fresh run)"
fi

echo "Starting eval on V100. Log: $run_log"
echo "Command: ${cmd[*]}"
set +e
set -o pipefail
"${cmd[@]}" 2>&1 | tee "$run_log"
rc="${PIPESTATUS[0]}"
set -e

if [[ "$rc" -eq 0 ]]; then
  results_file="$(latest_results_file)"
  if [[ -n "$results_file" ]]; then
    summary="$(format_results_summary "$results_file")"
    notify info "V100 eval completed. ${summary} | log=${run_log} | results=${results_file}"
    generate_reports "$results_file"
  else
    notify info "V100 eval completed without results file. log=${run_log}"
  fi
else
  notify critical "V100 eval failed with exit_code=${rc}. log=${run_log}"
fi

exit "$rc"
