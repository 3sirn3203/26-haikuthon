#!/bin/bash
# Main experiment queue after B1 core models (one GPU job at a time on MPS).
# Order = value for the 11/11 questions: L2 -> B3 (time limit) -> B2 (HN question) -> remaining B1.
set -u
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
log() { echo "$1 $(date +%H:%M) ${2:-}"; }
run() {  # name, command...
  local name=$1; shift
  log START "$name"
  if "$@" > cache/logs/$name.log 2>&1; then log DONE "$name $(grep -a -E 'score=|K=' cache/logs/$name.log | tail -3 | tr '\n' ' ' | cut -c1-300)"
  else log FAILED "$name $(grep -a -E 'Error|error' cache/logs/$name.log | tail -1 | cut -c1-200)"; fi
}
# wait for the running B1 job (Qwen3-VL-Embedding-2B) to release the GPU
while pgrep -f "run_exp.py --config configs/exp/b1_" > /dev/null; do sleep 30; done
log GPU_FREE
run l2_siglip2 .venv/bin/python scripts/l2_hn_detector.py --config configs/exp/b1_siglip2_so400m384.yaml
for c in b3_siglip2__qwen3vlrr_2b b3_siglip2__qwen35_2b b3_siglip2__qwen35_4b; do
  run $c .venv/bin/python scripts/run_rerank.py --config configs/exp/$c.yaml
done
for c in b2_siglip2_inbatch b2_siglip2_bundle b2_siglip2_bundle_hn; do
  [ -f configs/exp/$c.yaml ] && run $c .venv/bin/python scripts/finetune.py --config configs/exp/$c.yaml
done
for c in b1_openclip_h14 b1_eva02_l14_336 b1_metaclip2_h14_378; do
  run $c .venv/bin/python scripts/run_exp.py --config configs/exp/$c.yaml
done
log QUEUE_FINISHED
