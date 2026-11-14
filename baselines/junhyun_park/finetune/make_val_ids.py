"""chan_lee 와 같은 번들 단위 val 분할(val_split)과 번들 표를 만든다 (학습/검증 번들 누수 방지, 묶음 배치용).

레포 루트에서 (이미지 없이 실행 가능):
  python baselines/junhyun_park/finetune/make_val_ids.py
출력 (train 프롬프트 id 라 커밋해도 기밀 아님):
  baselines/junhyun_park/finetune/val_ids.txt        줄당 val 프롬프트 id (500개, chan_lee val 과 동일)
  baselines/junhyun_park/finetune/train_bundles.json  {프롬프트 id: 번들 id | null(기준 문항)}

chan_lee val_split 은 train HN 짝 정보(scripts/pair_hn.py 출력)가 필요하다. pair_hn.py 는 이미지를 읽지만,
저장하는 짝(source)은 순서 규칙(HN 파일 id = n_orig + k <-> 프롬프트 k)이고 이미지는 그 규칙의 검증 근거로만 쓴다.
그래서 짝 정보가 없으면 같은 규칙으로 캐시를 만든다 (결과 분할은 pair_hn.py 를 돌린 것과 같다).
캐시는 HAIKU_CACHE_DIR(기본: 이 폴더의 ../cache, git 제외)에 둔다.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ.setdefault("HAIKU_CACHE_DIR", str(HERE.parent / "cache"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "baselines/chan_lee/src"))
from haiku_bl import data as D  # noqa: E402

N_ORIG_TRAIN, N_HN_TRAIN = 4000, 500   # train_v1: 0000-3999 원본, 4000-4499 HN

cfg = D.load_paths()
pair = cfg["cache_dir"] / "meta" / "train_hn_sources.json"
if not pair.exists():
    pair.parent.mkdir(parents=True, exist_ok=True)
    pair.write_text(json.dumps({f"{N_ORIG_TRAIN + k}": {"source": f"{k:04d}", "rule": "order"}
                                for k in range(N_HN_TRAIN)}, indent=0))
    print(f"HN 짝 정보 없음 -> 순서 규칙으로 생성: {pair}")

bundles = D.bundles("train", cfg)
fit, val = D.val_split(cfg)
hn_src = set(D.hn_sources("train", cfg).values())

(HERE / "val_ids.txt").write_text("\n".join(sorted(val)) + "\n")
(HERE / "train_bundles.json").write_text(json.dumps(bundles, sort_keys=True))
n_b = len({b for b in bundles.values() if b})
print(f"fit {len(fit)} / val {len(val)} | val 쪽 HN {len(val & hn_src)} / 학습 쪽 HN {len(fit & hn_src)}"
      f" | 번들 {n_b}개, 기준 문항 {sum(b is None for b in bundles.values())}개")
print(f"-> {HERE / 'val_ids.txt'}\n-> {HERE / 'train_bundles.json'}")
