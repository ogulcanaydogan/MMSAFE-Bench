#!/usr/bin/env bash
set -euo pipefail

level="${1:-info}"
shift || true

message="${*:-}"
if [[ -z "${message// }" ]]; then
  message="(empty message)"
fi

bot_token="${TELEGRAM_BOT_TOKEN:-}"
chat_id="${TELEGRAM_CHAT_ID:-}"

# No-op when bot credentials are not configured; never break upstream jobs.
if [[ -z "$bot_token" ]]; then
  exit 0
fi

case "$level" in
  info|warn|critical) ;;
  *) level="info" ;;
esac

timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
host="$(hostname)"
text="[MMSAFE][${level^^}][${host}][${timestamp}] ${message}"

# If chat_id is not configured yet, try to infer it from bot updates.
if [[ -z "$chat_id" ]]; then
  updates_json="$(
    curl -sS "https://api.telegram.org/bot${bot_token}/getUpdates" 2>/dev/null || true
  )"
  chat_id="$(echo "$updates_json" | grep -o '"chat":{"id":[-0-9]*' | tail -n1 | cut -d: -f3)"
fi

# If still unknown, skip without failing.
if [[ -z "$chat_id" ]]; then
  exit 0
fi

curl -sS --fail-with-body \
  -X POST "https://api.telegram.org/bot${bot_token}/sendMessage" \
  --data-urlencode "chat_id=${chat_id}" \
  --data-urlencode "text=${text}" \
  --data-urlencode "disable_web_page_preview=true" \
  >/dev/null || true
