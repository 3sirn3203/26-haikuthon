#!/usr/bin/env python3
"""Run the shared evaluator with query-wise image-neighbor graph diffusion."""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix, diags


AUTHOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AUTHOR_ROOT.parents[1]
SHARED_RUNNER = REPO_ROOT / 'baselines' / 'chan_lee' / 'scripts' / 'run_exp.py'


def image_graph_diffusion(scores, image_embeddings, cfg):
    """Diffuse each query's scores over an image-only kNN graph.

    The graph is built solely from candidate image embeddings. Each query is propagated
    independently; no other query scores, IDs, or labels enter the adjustment.
    """
    if image_embeddings is None:
        raise RuntimeError('Image embeddings were not captured; disable embedding cache for this run.')
    x = np.asarray(image_embeddings, dtype=np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n = x.shape[0]
    k = min(int(cfg.get('k', 10)), n - 1)
    block_size = int(cfg.get('block_size', 256))
    temperature = float(cfg.get('temperature', 0.07))
    rows, cols, weights = [], [], []

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        sim = x[start:end] @ x.T
        sim[np.arange(end - start), np.arange(start, end)] = -np.inf
        idx = np.argpartition(sim, n - k, axis=1)[:, -k:]
        val = np.take_along_axis(sim, idx, axis=1)
        order = np.argsort(-val, axis=1, kind='stable')
        idx = np.take_along_axis(idx, order, axis=1)
        val = np.take_along_axis(val, order, axis=1)
        val = np.exp((val - val[:, :1]) / temperature).astype(np.float32)
        rows.append(np.repeat(np.arange(start, end, dtype=np.int32), k))
        cols.append(idx.astype(np.int32, copy=False).ravel())
        weights.append(val.ravel())

    graph = csr_matrix((np.concatenate(weights),
                        (np.concatenate(rows), np.concatenate(cols))), shape=(n, n), dtype=np.float32)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    graph = (diags(1.0 / np.maximum(degree, 1e-12), format='csr') @ graph).tocsr()

    base = np.asarray(scores, dtype=np.float32)
    current = base.copy()
    alpha = float(cfg.get('alpha', 0.15))
    steps = int(cfg.get('steps', 2))
    for _ in range(steps):
        propagated = graph.T.dot(current.T).T
        current = (1.0 - alpha) * base + alpha * np.asarray(propagated, dtype=np.float32)
    return current


def main():
    if not any(arg == '--paths' or arg.startswith('--paths=') for arg in sys.argv[1:]):
        sys.argv.extend(['--paths', str(AUTHOR_ROOT / 'configs' / 'paths.yaml')])

    spec = spec_from_file_location('_shared_graph_run_exp', SHARED_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load shared evaluator: {SHARED_RUNNER}')
    shared = module_from_spec(spec)
    spec.loader.exec_module(shared)
    shared.CHAN = AUTHOR_ROOT

    state = {'image_embeddings': None}
    original_build = shared.retrievers.build

    def build_and_capture(cfg, device):
        ret = original_build(cfg, device)
        encode_images = ret.encode_images

        def capture(paths, prof=None):
            embeddings = encode_images(paths, prof)
            state['image_embeddings'] = np.asarray(embeddings, dtype=np.float32)
            return embeddings

        ret.encode_images = capture
        return ret

    original_apply = shared.pipeline.loopholes.apply

    def apply_postprocess(cfg, scores, eval_set):
        if cfg.get('type') == 'image_graph_diffusion':
            return image_graph_diffusion(scores, state['image_embeddings'], cfg)
        return original_apply(cfg, scores, eval_set)

    shared.retrievers.build = build_and_capture
    shared.pipeline.loopholes.apply = apply_postprocess
    shared.main()


if __name__ == '__main__':
    main()
