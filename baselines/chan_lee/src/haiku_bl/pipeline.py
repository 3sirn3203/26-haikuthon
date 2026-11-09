"""Experiment-config -> (Q, N) score matrix. Shared by run_exp.py (organizer eval) and
predict.py (scoring-server entry point) so timed and scored code paths are identical."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import loopholes, rerankers, retrievers
from .metrics import topk_from_scores

RERANK_OFFSET = 1e3   # reranked top-k are placed above every non-reranked candidate


def score(cfg: dict, prompts: list[str], image_paths: list, prof, device, *,
          emb_cache: Path | None = None, eval_set=None):
    """Returns (final scores, stage-1 scores, info). ``emb_cache``: .npy path for image
    embeddings (read if present, else written); None = always recompute (timing runs)."""
    info = {'image_embeddings_cached': False}
    with prof.stage('model_load'):
        ret = retrievers.build(cfg['retriever'], device)
    info['stage1_model'] = ret.name
    info['params_b'] = ret.params_b
    if emb_cache is not None and Path(emb_cache).exists():
        img = np.load(emb_cache).astype(np.float32)
        info['image_embeddings_cached'] = True
    else:
        with prof.stage('image_encode'):
            img = ret.encode_images(image_paths, prof)
        if emb_cache is not None:
            Path(emb_cache).parent.mkdir(parents=True, exist_ok=True)
            np.save(emb_cache, img.astype(np.float16))
    with prof.stage('text_encode'):
        txt = ret.encode_texts(prompts, prof)
    if emb_cache is not None:             # text embeddings next to the image cache (for ablations)
        np.save(Path(emb_cache).with_name(Path(emb_cache).stem + f'__text{len(prompts)}.npy'), txt.astype(np.float16))
    with prof.stage('stage1_search'):
        s1 = ret.similarity(txt, img.astype(np.float32)).astype(np.float32)
    scores = s1

    rr_cfg = cfg.get('reranker')
    if rr_cfg:
        if hasattr(ret, 'model'):          # free the retriever before loading the reranker
            del ret
            _empty_cache(device)
        with prof.stage('model_load'):
            rr = rerankers.build(rr_cfg, device)
        info['params_b'] += rr.params_b
        top = topk_from_scores(s1, rr_cfg['top_k'])
        scores = s1.copy()
        with prof.stage('rerank'):
            for qi, prompt in enumerate(prompts):
                s = rr.score(prompt, [image_paths[j] for j in top[qi]], prof)
                scores[qi, top[qi]] = RERANK_OFFSET + np.asarray(s, dtype=np.float32)
    if cfg.get('postprocess'):
        with prof.stage('postprocess'):
            scores = loopholes.apply(cfg['postprocess'], scores, eval_set)
    return scores, s1, info


def _empty_cache(device):
    import torch
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
