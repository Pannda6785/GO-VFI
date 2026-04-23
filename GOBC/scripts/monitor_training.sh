#!/bin/sh
set -eu

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <pid> <output_dir> [interval_seconds]" >&2
  exit 2
fi

pid="$1"
output_dir="$2"
interval="${3:-300}"
monitor_log="$output_dir/monitor.log"

: > "$monitor_log"
echo "monitor_started $(date '+%F %T') pid=$pid" >> "$monitor_log"

while kill -0 "$pid" 2>/dev/null; do
  echo "--- $(date '+%F %T') ---" >> "$monitor_log"
  ps -p "$pid" -o pid,etime,%cpu,%mem,command >> "$monitor_log"
  if [ -f "$output_dir/epoch_metrics.jsonl" ]; then
    echo "latest_epoch_metrics" >> "$monitor_log"
    tail -n 1 "$output_dir/epoch_metrics.jsonl" >> "$monitor_log"
  fi
  if [ -f "$output_dir/latest_step.pt" ]; then
    ls -lhT "$output_dir/latest_step.pt" >> "$monitor_log"
  fi
  sleep "$interval"
done

echo "monitor_finished $(date '+%F %T')" >> "$monitor_log"
if [ -f "$output_dir/epoch_metrics.jsonl" ]; then
  echo "final_epoch_metrics" >> "$monitor_log"
  tail -n 5 "$output_dir/epoch_metrics.jsonl" >> "$monitor_log"
fi
