#!/bin/bash
# Run B1 configs as their model downloads finish (watches cache/model_download.log).
# Usage: bash scripts/queue_b1.sh  (from baselines/chan_lee)
set -u
export PYTHONDONTWRITEBYTECODE=1
declare -a Q=(
  "ViT-SO400M-16-SigLIP2-384|b1_siglip2_so400m384"
  "ViT-H-14-378-quickgelu|b1_dfn5b_h14_378"
  "PE-Core-L-14-336|b1_pecore_l14_336"
  "ViT-L-14-336-quickgelu|b1_clip_l14_336"
  "Qwen/Qwen3-VL-Embedding-2B|b1_qwen3vlemb_2b"
  "ViT-H-14 laion|b1_openclip_h14"
  "EVA02-L-14-336|b1_eva02_l14_336"
  "ViT-H-14-worldwide-378|b1_metaclip2_h14_378"
)
for item in "${Q[@]}"; do
  key=${item%%|*}; exp=${item#*|}
  until grep -q "^OK [0-9] ${key}" cache/model_download.log || grep -q "^FAIL [0-9] ${key}" cache/model_download.log; do sleep 20; done
  if grep -q "^FAIL [0-9] ${key}" cache/model_download.log; then echo "SKIP ${exp} (download failed)"; continue; fi
  echo "START ${exp} $(date +%H:%M)"
  if .venv/bin/python scripts/run_exp.py --config configs/exp/${exp}.yaml > cache/logs/${exp}.log 2>&1; then
    echo "DONE $(tail -1 cache/logs/${exp}.log | cut -c1-200)"
  else
    echo "FAILED ${exp}: $(grep -E 'Error|error' cache/logs/${exp}.log | tail -1 | cut -c1-200)"
  fi
done
echo "QUEUE FINISHED $(date +%H:%M)"
