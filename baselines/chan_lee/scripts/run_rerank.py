#!/usr/bin/env python
"""B3: two-stage reranking measured on a stratified query subset, all K from one run.

    python scripts/run_rerank.py --config configs/exp/b3_siglip2__qwen35_2b.yaml

Config:
  exp_id, stage1_config (a B1 yaml whose test embeddings are cached), reranker {type, model, ...},
  k_max (e.g. 50), k_list (e.g. [5, 10, 20, 50]), subset {private: 300, public: 100, seed: 0}

Optional time_budget_min stops scoring early (queries are processed in a seeded random order so a
partial run stays stratified); metrics then use the queries finished.
Each query's top-k_max (by stage-1 score) is rescored independently and saved after every query
to cache/rerank/<name>.npz, so the run resumes after interruption. Reranker scores of a pair do
not depend on K, so the top-K result for any K <= k_max is the rescored top-K of that run.
For each K, writes results/<name>_k<K>/metrics.json. Full-test time is extrapolated from the
measured seconds per pair (marked as an estimate) and added to the stage-1 timing of the B1 run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

CHAN = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CHAN / 'src'))
sys.path.insert(0, str(CHAN / 'scripts'))
from haiku_bl import data as D, metrics as M, rerankers  # noqa: E402
from haiku_bl.profiling import Profiler, pick_device  # noqa: E402
from run_exp import emb_cache_path, git_commit, top1_breakdown  # noqa: E402


def subset(es, spec):
    rng = np.random.default_rng(spec.get('seed', 0))
    out = []
    for tag in ('private', 'public'):
        idx = [i for i, q in enumerate(es.queries) if tag in q.tags]
        out += sorted(rng.choice(idx, min(spec.get(tag, 0), len(idx)), replace=False).tolist())
    return np.array(sorted(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    s1cfg = yaml.safe_load((CHAN / cfg['stage1_config']).read_text())
    paths = D.load_paths()
    es = D.build_eval_set('test', paths)
    f = emb_cache_path(s1cfg, es, paths)
    img = np.load(f).astype(np.float32)
    txt = np.load(f.with_name(f.stem + f'__text{len(es.queries)}.npy')).astype(np.float32)
    s1 = txt @ img.T
    qsel = subset(es, cfg['subset'])
    kmax = cfg['k_max']
    top = M.topk_from_scores(s1[qsel], kmax)

    name = cfg['exp_id'].split('/', 1)[1]
    store = paths['cache_dir'] / 'rerank' / f'{name}.npz'
    store.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    pair_sec = []
    if store.exists():
        z = np.load(store, allow_pickle=True)
        done = dict(zip(z['qi'].tolist(), z['scores']))
        pair_sec = z['pair_sec'].tolist()
        print(f'resuming: {len(done)}/{len(qsel)} queries done', flush=True)

    device = pick_device(a.device or cfg.get('device', 'auto'))
    prof = Profiler(device, cfg.get('vram_budget_gb', 24))
    with prof.stage('model_load'):
        rr = rerankers.build(cfg['reranker'], device)
    paths_pool = [im.path for im in es.pool]
    t_last = t_start = time.time()
    budget = cfg.get('time_budget_min')            # stop early; evaluate the queries done so far
    order = np.random.default_rng(cfg['subset'].get('seed', 0)).permutation(len(qsel))  # keeps strata mixed
    for n in order:
        qi, cand = qsel[n], top[n]
        if int(qi) in done:
            continue
        if budget and (time.time() - t_start) / 60 > budget:
            print(f'time budget {budget} min reached after {len(done)} queries', flush=True)
            break
        t0 = time.perf_counter()
        with prof.stage('rerank'):
            sc = rr.score(es.queries[qi].prompt, [paths_pool[j] for j in cand], prof)
        pair_sec.append((time.perf_counter() - t0) / kmax)
        done[int(qi)] = np.asarray(sc, dtype=np.float32)
        if time.time() - t_last > 60 or len(done) == len(qsel):
            keys = sorted(done)
            np.savez(store, qi=np.array(keys), scores=np.stack([done[k] for k in keys]), pair_sec=np.array(pair_sec))
            t_last = time.time()
            print(f'{len(done)}/{len(qsel)} queries, {np.median(pair_sec):.3f}s/pair, peak {prof.peak_gb:.1f}GB',
                  flush=True)
    keys = sorted(done)
    np.savez(store, qi=np.array(keys), scores=np.stack([done[k] for k in keys]), pair_sec=np.array(pair_sec))

    # stage-1 timing of the full test run (from the B1 metrics, measured without cache if available)
    b1 = json.loads((CHAN / 'results' / s1cfg['exp_id'].split('/', 1)[1] / 'metrics.json').read_text())
    t1 = b1['timing']
    sec_pair = float(np.median(pair_sec))
    keep = np.array([int(q) in done for q in qsel])
    qsel, top = qsel[keep], top[keep]
    rr_scores = np.stack([done[int(q)] for q in qsel])
    sub = D.EvalSet(es.name, [es.queries[i] for i in qsel], es.pool, es.public_provisional, es.public_source)
    masks = sub.masks()
    s1_sub = s1[qsel]
    s1_ranks = M.ranks_from_scores(s1_sub, sub.gt_index)
    base = M.summarize(s1_ranks)
    for K in cfg['k_list']:
        sc = s1_sub.copy()
        rows = np.arange(len(qsel))[:, None]
        sc[rows, top[:, :K]] = 1e3 + rr_scores[:, :K]
        ranks = M.ranks_from_scores(sc, sub.gt_index)
        by = {k: dict(M.summarize(ranks, m), ci95=M.bootstrap_ci(ranks, m)) for k, m in masks.items()}
        est_rerank_full = sec_pair * K * len(es.queries)
        res = {
            'exp_id': f"{cfg['exp_id']}_k{K}", 'author': cfg['exp_id'].split('/')[0], 'category': 'honest',
            'method': cfg.get('method', '') + f' (top-{K} of {s1cfg["exp_id"]})',
            'stage1_model': b1['stage1_model'], 'stage2_model': cfg['reranker']['model'], 'top_k': K,
            'uses_train_finetune': bool(s1cfg.get('uses_train_finetune', False)),
            'uses_train_hard_negatives': bool(s1cfg.get('uses_train_hard_negatives', False)),
            'violates_cross_query_rule': False,
            'eval_split': 'test_subset', 'n_queries': len(qsel), 'subset': cfg['subset'],
            'subset_truncated_by_time_budget': bool(len(qsel) < sum(cfg['subset'].get(t, 0) for t in ('private', 'public'))),
            'pool_size': len(es.pool), 'n_hn_in_pool': sum(im.kind == 'hn' for im in es.pool),
            'scores': dict(M.summarize(ranks), ci95=M.bootstrap_ci(ranks)),
            'scores_by_split': {k: by[k] for k in ('public', 'private') if k in by},
            'scores_by_type': {k: by[k] for k in ('easy', 'bundle') if k in by},
            'stage1_scores_same_subset': base,
            'stage1_recall_at_k_same_subset': float((s1_ranks <= K).mean()),
            'public_split_provisional': es.public_provisional, 'public_split_source': es.public_source,
            'diagnostics': {'top1_breakdown': top1_breakdown(M.topk_from_scores(sc, 1)[:, 0], sub)},
            'timing': {
                'device': prof.report()['device'],
                'measured_sec_per_pair': sec_pair, 'measured_pairs': len(pair_sec) * kmax,
                'model_load_sec': prof.times.get('model_load', 0.0) + (t1.get('model_load_sec') or 0.0),
                'image_encode_sec': t1.get('image_encode_sec'), 'text_encode_sec': t1.get('text_encode_sec'),
                'stage1_search_sec': t1.get('stage1_search_sec'),
                'rerank_sec': est_rerank_full,
                'rerank_sec_is_estimate': True,
                'total_sec': None if t1.get('image_encode_sec') is None else
                (t1['image_encode_sec'] + t1['text_encode_sec'] + t1['stage1_search_sec'] + est_rerank_full
                 + t1['model_load_sec'] + prof.times.get('model_load', 0.0)),
                'total_sec_is_estimate': True,
            },
            'peak_vram_gb': max(prof.peak_gb, b1.get('peak_vram_gb') or 0.0),
            'params_b': b1['params_b'] + rr.params_b, 'artifact_size_gb': (b1['params_b'] + rr.params_b) * 2,
            'git_commit': git_commit(), 'date': dt.datetime.now().isoformat(timespec='seconds'),
        }
        out = CHAN / 'results' / f'{name}_k{K}'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'metrics.json').write_text(json.dumps(res, indent=2, ensure_ascii=False))
        (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        if not (out / 'notes.md').exists():
            (out / 'notes.md').write_text(f"# {res['exp_id']}\n\n{res['method']}\n\n## 관찰\n\n- \n")
        s = res['scores']
        print(f"K={K}: score={s['score']:.4f} (stage1 same subset {base['score']:.4f}) "
              + ' '.join(f"{k}={v['score']:.4f}" for k, v in res['scores_by_split'].items())
              + f" | est. full rerank {est_rerank_full/60:.1f} min on {res['timing']['device']}", flush=True)


if __name__ == '__main__':
    main()
