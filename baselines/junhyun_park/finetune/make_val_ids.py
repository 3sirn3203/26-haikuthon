"""chan_lee 의 번들 단위 val 분할(val_split)로 val 프롬프트 id 목록을 만든다 (학습/검증 번들 누수 방지).

레포 루트에서 (data/train_v1 이 있어야 함), 최초 1회:
  python baselines/chan_lee/scripts/pair_hn.py --split train
  python baselines/junhyun_park/finetune/make_val_ids.py
출력: baselines/junhyun_park/finetune/val_ids.txt  (줄당 train 프롬프트 id; train 프롬프트 id라 커밋해도 기밀 아님)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "baselines/chan_lee/src"))
from haiku_bl import data as D  # noqa: E402

cfg = D.load_paths()
fit, val = D.val_split(cfg)
out = Path(__file__).with_name("val_ids.txt")
out.write_text("\n".join(sorted(val)) + "\n")
print(f"fit {len(fit)} / val {len(val)} -> {out}")
