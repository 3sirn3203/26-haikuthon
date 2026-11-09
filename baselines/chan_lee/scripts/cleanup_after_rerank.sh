#!/bin/bash
# Free disk after the last experiment: delete data/ images and Qwen3.5-4B weights,
# only if the fine-tuned rerank finished and wrote its metrics.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "scripts/queue_ft_rerank.sh" > /dev/null; do sleep 20; done
if ls results/b3_siglip2_bundle__qwen35_4b_k10/metrics.json > /dev/null 2>&1; then
  rm -rf ../../data/test_v1 ../../data/train_v1 && echo "CLEANUP data/test_v1 data/train_v1 removed"
  .venv/bin/python - <<'PY'
from huggingface_hub import scan_cache_dir
info = scan_cache_dir()
revs = [rev.commit_hash for r in info.repos if r.repo_id == 'Qwen/Qwen3.5-4B' for rev in r.revisions]
s = info.delete_revisions(*revs); s.execute(); print(f'CLEANUP Qwen3.5-4B removed ({s.expected_freed_size/1e9:.1f} GB)')
PY
  echo "CLEANUP_DONE free $(df -h /System/Volumes/Data | tail -1 | awk '{print $4}')"
else
  echo "CLEANUP_SKIPPED rerank metrics missing; data and weights kept"
fi
