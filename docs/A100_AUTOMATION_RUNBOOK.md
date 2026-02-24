# A100 Automation Runbook

This runbook enforces the GPU policy:

1. `LowResource-LLM-Forge` training has priority.
2. `MMSAFE-Bench` full eval runs only after training is idle.

## Files

- Waiter: `/home/weezboo/projects/MMSAFE-Bench/scripts/wait_and_run_full_eval.sh`
- Telegram notifier: `/home/weezboo/projects/MMSAFE-Bench/scripts/ops/notify_telegram.sh`
- LowResource monitor: `/home/weezboo/projects/MMSAFE-Bench/scripts/ops/monitor_lowresource.sh`
- Systemd units:
  - `/home/weezboo/projects/MMSAFE-Bench/deploy/systemd/mmsafe-waiter.service`
  - `/home/weezboo/projects/MMSAFE-Bench/deploy/systemd/lowresource-monitor.service`
  - `/home/weezboo/projects/MMSAFE-Bench/deploy/systemd/mmsafe-v100-eval.service`

## Required Environment Files (A100 host)

### `/home/weezboo/.mmsafe.env`

```bash
REPLICATE_API_TOKEN="<replicate_token>"
```

### `/home/weezboo/.notify.env`

```bash
TELEGRAM_BOT_TOKEN="<telegram_bot_token>"
TELEGRAM_CHAT_ID="<telegram_chat_id>"
```

Apply strict permissions:

```bash
chmod 600 /home/weezboo/.mmsafe.env /home/weezboo/.notify.env
```

## Enable Services

```bash
mkdir -p ~/.config/systemd/user
cp /home/weezboo/projects/MMSAFE-Bench/deploy/systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now mmsafe-waiter.service
systemctl --user enable --now lowresource-monitor.service
loginctl enable-linger weezboo
```

## Optional Parallel Run on V100

When A100 is busy with LowResource training, run full MMSAFE on V100:

```bash
systemctl --user enable --now mmsafe-v100-eval.service
systemctl --user status mmsafe-v100-eval.service --no-pager
```

This unit resumes automatically from:

`/home/weezboo/projects/MMSAFE-Bench/artifacts/full_eval_a100_ready/checkpoints/checkpoint_eval-*.json`

## Health Checks

```bash
systemctl --user status mmsafe-waiter.service --no-pager
systemctl --user status lowresource-monitor.service --no-pager
journalctl --user -u mmsafe-waiter.service -n 80 --no-pager
journalctl --user -u lowresource-monitor.service -n 80 --no-pager
```

## Manual Result Reporting

After full eval completes:

```bash
cd /home/weezboo/projects/MMSAFE-Bench
LATEST_RESULT="$(ls -1t artifacts/full_eval_a100_ready/*_results.json | head -n1)"
uv run mmsafe report "$LATEST_RESULT" --format html
uv run mmsafe report "$LATEST_RESULT" --format markdown
```

## Token Rotation

The Replicate token was exposed in chat and must be rotated.

1. Create a new token on Replicate.
2. Replace `REPLICATE_API_TOKEN` in `/home/weezboo/.mmsafe.env`.
3. Restart waiter:

```bash
systemctl --user restart mmsafe-waiter.service
```
