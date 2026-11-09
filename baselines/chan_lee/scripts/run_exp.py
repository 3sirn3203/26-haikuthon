#!/usr/bin/env python
"""Single entry point for one experiment (organizer-side: uses ground truth).

    python scripts/run_exp.py --config configs/exp/b1_siglip2_so400m384.yaml [--no-cache] [--limit 200]

Writes results/<name>/{metrics.json, config.yaml, notes.md} (committed) and
cache/runs/<name>/{ranks.npy, top10.npy} (git-ignored, organizer-only analysis).
--no-cache recomputes image embeddings so the timing fields are real.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

CHAN = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CHAN / 'src'))

from haiku_bl import data as D, metrics as M, pipeline, retrievers  # noqa: E402
from haiku_bl.profiling import Profiler, pick_device  # noqa: E402

FLAGS = ('uses_train_finetune', 'uses_train_hard_negatives', 'violates_cross_query_rule')


def git_commit() -> str:
    try:
        sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=CHAN, capture_output=True,
                             text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=CHAN,
                               capture_output=True, text=True).stdout.strip()
        return sha + ('-dirty' if dirty else '')
    except OSError:
        return ''


def emb_cache_path(cfg, es, paths_cfg):
    r = cfg['retriever']
    if r.get('type') == 'random':
        return None
    name = r['model'] if r.get('pretrained') is None else f"{r['model']}:{r['pretrained']}"
    ids = '\n'.join(im.iid for im in es.pool)
    h = hashlib.sha1(ids.encode()).hexdigest()[:10]
    return paths_cfg['cache_dir'] / 'emb' / retrievers.cache_key(name, r.get('checkpoint')) / f'{es.name}_{h}.npy'


def top1_breakdown(top1, es) -> dict:
    """Where the top-1 image came from: GT, the query's own HN, a bundle sibling (or its HN), other."""
    bundle_of = {q.pid: q.bundle for q in es.queries}
    cats = dict.fromkeys(('gt', 'own_hn', 'bundle_sibling', 'bundle_sibling_hn', 'other'), 0)
    for q, j in zip(es.queries, top1):
        im = es.pool[j]
        if im.iid == q.gt:
            cats['gt'] += 1
        elif im.kind == 'hn' and im.source_pid == q.pid:
            cats['own_hn'] += 1
        elif q.bundle is not None and bundle_of.get(im.source_pid) == q.bundle:
            cats['bundle_sibling_hn' if im.kind == 'hn' else 'bundle_sibling'] += 1
        else:
            cats['other'] += 1
    return {k: v / len(es.queries) for k, v in cats.items()}


def hn_above_gt(scores, es):
    """Fraction of queries (that have an HN in the pool) whose own HN scores >= the GT."""
    idx = {im.source_pid: i for i, im in enumerate(es.pool) if im.kind == 'hn'}
    pairs = [(qi, idx[q.pid]) for qi, q in enumerate(es.queries) if q.pid in idx]
    if not pairs:
        return None
    qi, hj = map(np.array, zip(*pairs))
    return float((scores[qi, hj] >= scores[qi, es.gt_index[qi]]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--paths', default=None)
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--limit', type=int, default=None, help='first N queries only (pool unchanged)')
    ap.add_argument('--device', default=None)
    ap.add_argument('--vram-budget-gb', type=float, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths_cfg = D.load_paths(args.paths)
    exp_id = cfg['exp_id']
    author, name = exp_id.split('/', 1)
    limit = args.limit or cfg.get('limit')
    if limit:
        name += f'__limit{limit}'

    es = D.build_eval_set(cfg['eval_split'], paths_cfg, include_hn=cfg.get('include_hn', True))
    if limit:
        es = D.EvalSet(es.name, es.queries[:limit], es.pool, es.public_provisional, es.public_source)
    if cfg['retriever'].get('type') != 'random':
        missing = es.missing_images()
        if missing:
            sys.exit(f'{len(missing)} pool images missing (e.g. {missing[:3]}); check configs/paths.yaml '
                     f'image patterns and HAIKU_DATA_ROOT ({paths_cfg["data_root"]})')

    device = pick_device(args.device or cfg.get('device', 'auto'))
    prof = Profiler(device, args.vram_budget_gb or cfg.get('vram_budget_gb', 24))
    use_cache = cfg.get('use_cache', True) and not args.no_cache
    scores, s1, info = pipeline.score(cfg, es.prompts, [im.path for im in es.pool], prof, device,
                                      emb_cache=emb_cache_path(cfg, es, paths_cfg) if use_cache else None,
                                      eval_set=es)

    ranks = M.ranks_from_scores(scores, es.gt_index)
    top10 = M.topk_from_scores(scores, 10)
    by = {k: M.summarize(ranks, m) for k, m in es.masks().items()}
    t = prof.report()
    st = t['stages_sec']
    cached = info['image_embeddings_cached']
    rr = cfg.get('reranker') or {}
    result = {
        'exp_id': exp_id, 'author': author, 'category': cfg.get('category', 'honest'),
        'method': cfg.get('method', ''),
        'stage1_model': info['stage1_model'], 'stage2_model': rr.get('model'), 'top_k': rr.get('top_k'),
        **{f: bool(cfg.get(f, False)) for f in FLAGS},
        'eval_split': 'test_full' if es.name == 'test' else es.name,
        'n_queries': len(es.queries), 'pool_size': len(es.pool),
        'n_hn_in_pool': sum(im.kind == 'hn' for im in es.pool),
        'scores': M.summarize(ranks),
        'scores_by_split': {k: by[k] for k in ('public', 'private', 'val') if k in by},
        'scores_by_type': {k: by[k] for k in ('easy', 'bundle') if k in by},
        'public_split_provisional': es.public_provisional,
        'public_split_source': es.public_source,
        'expected_random': M.expected_random(len(es.pool)),
        'diagnostics': {'top1_breakdown': top1_breakdown(top10[:, 0], es),
                        'stage1_hn_above_gt_rate': hn_above_gt(s1, es)},
        'timing': {
            'device': t['device'], 'image_embeddings_cached': cached,
            'model_load_sec': st.get('model_load', 0.0),
            'image_encode_sec': None if cached else st.get('image_encode', 0.0),
            'image_io_wait_sec': None if cached else st.get('image_io_wait', 0.0),
            'text_encode_sec': st.get('text_encode', 0.0),
            'stage1_search_sec': st.get('stage1_search', 0.0),
            'rerank_sec': st.get('rerank', 0.0),
            'postprocess_sec': st.get('postprocess', 0.0),
            'total_sec': t['total_sec'],
        },
        'peak_vram_gb': t['peak_vram_gb'], 'within_vram_budget': t['within_budget'],
        'params_b': info['params_b'], 'artifact_size_gb': info['params_b'] * 2,   # fp16 estimate
        'git_commit': git_commit(), 'date': dt.datetime.now().isoformat(timespec='seconds'),
    }
    out = CHAN / 'results' / name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    if not (out / 'notes.md').exists():
        (out / 'notes.md').write_text(f'# {exp_id}\n\n{cfg.get("method", "")}\n\n## 관찰\n\n- \n')
    run = paths_cfg['cache_dir'] / 'runs' / name
    run.mkdir(parents=True, exist_ok=True)
    np.save(run / 'ranks.npy', ranks)
    np.save(run / 'top10.npy', top10)
    s = result['scores']
    print(f'{exp_id}: score={s["score"]:.4f} R@1={s["recall_at_1"]:.4f} NDCG@5={s["ndcg_at_5"]:.4f} '
          + ' '.join(f'{k}={v["score"]:.4f}' for k, v in result['scores_by_split'].items())
          + f' | total {t["total_sec"]:.1f}s peak {t["peak_vram_gb"]:.2f}GB cached={cached} -> {out}')


if __name__ == '__main__':
    main()
