#!/bin/bash
# After B2: rerank the best fine-tuned stage-1 (chosen on VAL, not test) with Qwen3.5-4B.
set -u
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
while pgrep -f "scripts/queue_b2.sh" > /dev/null; do sleep 30; done
until grep -q '^OK Qwen/Qwen3.5-4B' cache/logs/dl_qwen35_4b.log; do
  if ! pgrep -f "Qwen3.5-4B" > /dev/null && ! grep -q '^OK' cache/logs/dl_qwen35_4b.log; then echo "FAILED download"; exit 1; fi
  sleep 30
done
best=$(.venv/bin/python - <<'PY'
import json, glob
c = {p.split('/')[1][:-5]: json.load(open(p))['scores']['score'] for p in glob.glob('results/b2_siglip2_*__val/metrics.json')}
print(max(c, key=c.get))
PY
)
echo "BEST_ON_VAL $(date +%H:%M) $best"
cat > configs/exp/${best}_stage1.yaml <<YAML
exp_id: chan_lee/${best}
category: honest
method: "stage-1 config of ${best} (fine-tuned SigLIP2 checkpoint), used as the base for B3 reranking"
uses_train_finetune: true
eval_split: test
retriever: {type: openclip, model: ViT-SO400M-16-SigLIP2-384, pretrained: webli, precision: fp16, batch_size: 32, num_workers: 8, checkpoint: cache/ckpt/${best}.pt}
YAML
grep -q 'uses_train_hard_negatives: true' configs/exp/${best}.yaml && echo "uses_train_hard_negatives: true" >> configs/exp/${best}_stage1.yaml
name=b3_${best#b2_}__qwen35_4b
cat > configs/exp/${name}.yaml <<YAML
exp_id: chan_lee/${name}
category: honest
method: "B3 two-stage: fine-tuned SigLIP2 (${best}, selected on val) top-K reranked by Qwen3.5-4B yes/no logit"
stage1_config: configs/exp/${best}_stage1.yaml
reranker: {type: vlm_yesno, model: Qwen/Qwen3.5-4B, dtype: bf16, batch_size: 10, max_pixels: 200704}
k_max: 10
k_list: [5, 10]
subset: {private: 300, public: 100, seed: 0}
time_budget_min: 60
YAML
echo "START $(date +%H:%M) $name"
if .venv/bin/python scripts/run_rerank.py --config configs/exp/${name}.yaml > cache/logs/${name}.log 2>&1; then
  echo "DONE $(date +%H:%M) $name $(grep -a 'K=' cache/logs/${name}.log | tr '\n' ' ' | cut -c1-400)"
else echo "FAILED $(date +%H:%M) $name $(grep -a -E 'Error|error' cache/logs/${name}.log | tail -1 | cut -c1-200)"; fi
echo "QUEUE_FINISHED $(date +%H:%M)"
