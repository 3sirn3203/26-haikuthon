#!/usr/bin/env python
"""Build an anonymized test package like the scoring server would see it (organizer-side).

Images are symlinked under random ids, prompts are written without answers, and the answer
key goes to a separate file. Everything lands in cache/ (git-ignored).

    python scripts/make_blind_test.py [--split test|val] [--seed 0]
    python scripts/predict.py ... --images cache/blind/test/images --prompts cache/blind/test/prompts.jsonl
    python scripts/make_blind_test.py --score submission.csv
"""
import argparse
import json
import secrets
import sys
from pathlib import Path

CHAN = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CHAN / 'src'))

from haiku_bl import data as D  # noqa: E402
from haiku_bl.submission import score_submission  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--score', default=None, help='score a submission against the answer key')
    a = ap.parse_args()
    cfg = D.load_paths()
    root = cfg['cache_dir'] / 'blind' / a.split
    if a.score:
        key = json.loads((root / 'answer_key.json').read_text())
        print(json.dumps(score_submission(a.score, key['gt'], key['masks']), indent=2))
        return
    es = D.build_eval_set(a.split, cfg)
    missing = es.missing_images()
    if missing:
        sys.exit(f'{len(missing)} images missing')
    img_dir = root / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    anon = {}
    for im in es.pool:
        rid = secrets.token_hex(6)
        anon[im.iid] = rid
        (img_dir / f'{rid}{Path(im.path).suffix}').symlink_to(Path(im.path).resolve())
    with open(root / 'prompts.jsonl', 'w') as f:
        for q in es.queries:
            f.write(json.dumps({'id': q.pid, 'prompt': q.prompt}) + '\n')
    masks = {t: [q.pid for q in es.queries if t in q.tags] for t in ('public', 'private', 'val')}
    (root / 'answer_key.json').write_text(json.dumps(
        {'gt': {q.pid: anon[q.gt] for q in es.queries}, 'masks': {k: v for k, v in masks.items() if v}}))
    print(f'{len(es.pool)} images, {len(es.queries)} prompts -> {root}')


if __name__ == '__main__':
    main()
