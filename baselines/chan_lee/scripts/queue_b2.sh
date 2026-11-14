#!/bin/bash
# B2 fine-tuning conditions (args: config names; default all three). One GPU job at a time.
set -u
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
for c in ${@:-b2_siglip2_inbatch b2_siglip2_bundle b2_siglip2_bundle_hn}; do
  echo "START $(date +%H:%M) $c"
  if .venv/bin/python scripts/finetune.py --config configs/exp/$c.yaml > cache/logs/$c.log 2>&1; then
    echo "DONE $(date +%H:%M) $c $(grep -a 'score=' cache/logs/$c.log | tr '\n' ' ' | cut -c1-400)"
  else echo "FAILED $(date +%H:%M) $c $(grep -a -E 'Error|error' cache/logs/$c.log | tail -1 | cut -c1-200)"; fi
done
echo "QUEUE_FINISHED $(date +%H:%M)"
