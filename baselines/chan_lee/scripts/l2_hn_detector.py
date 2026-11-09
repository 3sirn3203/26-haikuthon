#!/usr/bin/env python
"""L2 (loophole): an image-level "is this a hard negative?" detector used as a ranking prior.

Trains L2-regularised logistic regression on frozen image embeddings of the train package
(originals vs the 500 train HN, val bundles held out), then ranks with
    score' = stage1 score - lambda * logit P(HN | image)
lambda is chosen on val; the oracle-best lambda on test is reported too. No cross-query
information is used, but it sidesteps the task (which HN looks generated, not which matches).

    python scripts/l2_hn_detector.py --config configs/exp/b1_siglip2_so400m384.yaml
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu

CHAN = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CHAN / 'src'))
sys.path.insert(0, str(CHAN / 'scripts'))
from haiku_bl import data as D, metrics as M, retrievers  # noqa: E402
from haiku_bl.profiling import pick_device  # noqa: E402
from run_exp import emb_cache_path, git_commit  # noqa: E402

LAMBDAS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2)


def fit_logreg(X, y, l2=1.0):
    X = np.hstack([X, np.ones((len(X), 1))])
    w_pos = len(y) / (2 * y.sum()); w_neg = len(y) / (2 * (len(y) - y.sum()))
    sw = np.where(y == 1, w_pos, w_neg)

    def f(w):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        loss = -(sw * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))).sum() + l2 * (w[:-1] ** 2).sum()
        g = X.T @ (sw * (p - y)) + 2 * l2 * np.r_[w[:-1], 0]
        return loss, g
    return minimize(f, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B').x


def logit(X, w):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def auc(pos, neg):
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    paths = D.load_paths()
    # train package embeddings (all 4,500 images), cached like the eval pools
    pkg = D.package('train', paths)
    src = D.hn_sources('train', paths)
    fit, val = D.val_split(paths)
    items = [(p, pkg['orig'][p], 0, p) for p in sorted(pkg['orig'])] + \
            [(h, pkg['hn'][h], 1, src[h]) for h in sorted(pkg['hn'])]
    r = cfg['retriever']
    name = r['model'] if r.get('pretrained') is None else f"{r['model']}:{r['pretrained']}"
    h = hashlib.sha1('\n'.join(str(x[1]) for x in items).encode()).hexdigest()[:10]
    f = paths['cache_dir'] / 'emb' / retrievers.cache_key(name, r.get('checkpoint')) / f'trainpkg_{h}.npy'
    ret = retrievers.build(r, pick_device('auto'))      # text encoder is needed for val prompts anyway
    if f.exists():
        X = np.load(f).astype(np.float32)
    else:
        X = ret.encode_images([x[1] for x in items])
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, X.astype(np.float16))
    y = np.array([x[2] for x in items], dtype=np.float64)
    in_fit = np.array([x[3] in fit for x in items])
    w = fit_logreg(X[in_fit].astype(np.float64), y[in_fit])
    val_auc = auc(logit(X[~in_fit], w)[y[~in_fit] == 1], logit(X[~in_fit], w)[y[~in_fit] == 0])

    def eval_with(es, lam_grid, img, txt):
        S = txt @ img.T
        z = logit(img.astype(np.float64), w).astype(np.float32)
        is_hn = np.array([im.kind == 'hn' for im in es.pool])
        out = {}
        for lam in lam_grid:
            ranks = M.ranks_from_scores(S - lam * z[None, :], es.gt_index)
            out[lam] = (ranks, M.summarize(ranks)['score'])
        return out, auc(z[is_hn], z[~is_hn])

    val_es = D.build_eval_set('val', paths)
    row_of = {('train/orig/' if x[2] == 0 else 'train/hn/') + x[0]: i for i, x in enumerate(items)}
    v_img = X[[row_of[im.iid] for im in val_es.pool]]
    v_txt = ret.encode_texts(val_es.prompts)
    vres, _ = eval_with(val_es, LAMBDAS, v_img, v_txt)
    lam_val = max(LAMBDAS, key=lambda l: vres[l][1])
    test_es = D.build_eval_set('test', paths)
    ef = emb_cache_path(cfg, test_es, paths)
    t_img = np.load(ef).astype(np.float32)
    t_txt = np.load(ef.with_name(ef.stem + f'__text{len(test_es.queries)}.npy')).astype(np.float32)
    tres, test_auc = eval_with(test_es, LAMBDAS, t_img, t_txt)
    lam_oracle = max(LAMBDAS, key=lambda l: tres[l][1])
    masks = test_es.masks()
    ranks = tres[lam_val][0]
    base = tres[0.0][0]
    b1 = json.loads((CHAN / 'results' / cfg['exp_id'].split('/', 1)[1] / 'metrics.json').read_text())
    res = {
        'exp_id': 'chan_lee/l2_hn_detector_' + cfg['exp_id'].split('/', 1)[1].removeprefix('b1_'),
        'author': 'chan_lee', 'category': 'loophole',
        'method': f'L2 image-level HN detector (logreg on frozen {name} embeddings, train HN) as ranking prior',
        'stage1_model': b1['stage1_model'], 'stage2_model': 'logreg_hn_prior', 'top_k': None,
        'uses_train_finetune': True, 'uses_train_hard_negatives': True, 'violates_cross_query_rule': False,
        'eval_split': 'test_full', 'n_queries': len(test_es.queries), 'pool_size': len(test_es.pool),
        'n_hn_in_pool': int(sum(im.kind == 'hn' for im in test_es.pool)),
        'scores': dict(M.summarize(ranks), ci95=M.bootstrap_ci(ranks)),
        'scores_by_split': {k: M.summarize(ranks, masks[k]) for k in ('public', 'private')},
        'baseline_scores_by_split': {k: M.summarize(base, masks[k]) for k in ('public', 'private')},
        'detector': {'val_auc': val_auc, 'test_auc': test_auc, 'n_train': int(in_fit.sum()),
                     'n_train_hn': int(y[in_fit].sum()), 'lambda_chosen_on_val': lam_val,
                     'lambda_oracle_on_test': lam_oracle, 'test_score_by_lambda': {str(l): tres[l][1] for l in LAMBDAS},
                     'val_score_by_lambda': {str(l): vres[l][1] for l in LAMBDAS}},
        'public_split_provisional': test_es.public_provisional, 'public_split_source': test_es.public_source,
        'timing': dict(b1['timing'], note='stage-1 timing of the B1 run; detector cost is negligible'),
        'peak_vram_gb': b1['peak_vram_gb'], 'params_b': b1['params_b'], 'artifact_size_gb': b1['artifact_size_gb'],
        'git_commit': git_commit(), 'date': dt.datetime.now().isoformat(timespec='seconds'),
    }
    out = CHAN / 'results' / res['exp_id'].split('/', 1)[1]
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(res, indent=2))
    (out / 'config.yaml').write_text(yaml.safe_dump({'stage1_config': a.config, 'lambdas': list(LAMBDAS)}))
    if not (out / 'notes.md').exists():
        (out / 'notes.md').write_text(f"# {res['exp_id']}\n\n{res['method']}\n\n## 관찰\n\n- \n")
    print(json.dumps({k: res[k] for k in ('scores', 'scores_by_split', 'baseline_scores_by_split', 'detector')}, indent=1))


if __name__ == '__main__':
    main()
