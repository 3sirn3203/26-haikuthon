#!/usr/bin/env python
"""L0 leak audit of the data packages (organizer-side). Writes only aggregate numbers to
results/l0_leak_audit/audit.json; no prompt text, no id lists.

    python scripts/check_leaks.py
"""
import json
import struct
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import mannwhitneyu

CHAN = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CHAN / 'src'))
from haiku_bl import data as D  # noqa: E402


def png_info(path):
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:8] == b'\x89PNG\r\n\x1a\n', path
    chunks, i = [], 8
    while i < len(data):
        n, = struct.unpack('>I', data[i:i + 4])
        chunks.append(data[i + 4:i + 8].decode('latin1'))
        i += 12 + n
    ihdr = struct.unpack('>IIBBBBB', data[16:29])
    return {'size': len(data), 'chunks': tuple(sorted(set(chunks))), 'w': ihdr[0], 'h': ihdr[1],
            'bitdepth': ihdr[2], 'colortype': ihdr[3]}


def auc(pos, neg):
    """P(score(pos) > score(neg)); 0.5 = no signal. Reported as max(auc, 1-auc)."""
    u = mannwhitneyu(pos, neg).statistic
    a = u / (len(pos) * len(neg))
    return float(max(a, 1 - a))


def mean_rgb(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert('RGB').resize((64, 64)), dtype=np.float32)
    return [a.mean(), a.std(), a[..., 0].mean() - a[..., 2].mean()]


def main():
    cfg = D.load_paths()
    out = {}
    for split in ('train', 'test'):
        pkg = D.package(split, cfg)
        src = D.hn_sources(split, cfg)
        o_paths = [pkg['orig'][p] for p in sorted(pkg['orig'])]
        h_paths = [pkg['hn'][h] for h in sorted(pkg['hn'])]
        with ThreadPoolExecutor(12) as ex:
            oi = list(ex.map(png_info, o_paths))
            hi = list(ex.map(png_info, h_paths))
            orgb = np.array(list(ex.map(mean_rgb, o_paths)))
            hrgb = np.array(list(ex.map(mean_rgb, h_paths)))
        stems = [int(Path(p).stem) for p in o_paths]
        n_o = len(o_paths)
        out[split] = {
            'n_orig': n_o, 'n_hn': len(h_paths),
            'orig_file_id_equals_prompt_id': float(np.mean([s == int(p) for s, p in zip(stems, sorted(pkg['orig']))])),
            'hn_ids_all_after_originals': bool(min(int(h) for h in pkg['hn']) >= n_o),
            'hn_rows_labelled_in_prompts_jsonl': True,
            'hn_file_order_mirrors_source_order': bool(src) and all(
                int(h) - n_o == int(s) for h, s in src.items()),
            'hn_order_rule_pixel_rank1_rate': float(np.mean(
                [v['pixel_rank_of_source'] == 1 for v in json.loads(
                    (cfg['cache_dir'] / 'meta' / f'{split}_hn_sources.json').read_text()).values()])) if src else None,
            'png_chunk_sets': {','.join(k): v for k, v in Counter(i['chunks'] for i in oi + hi).items()},
            'image_formats': {f'{k[0]}x{k[1]} depth{k[2]} type{k[3]}': v for k, v in Counter(
                (i['w'], i['h'], i['bitdepth'], i['colortype']) for i in oi + hi).items()},
            'file_size_mb': {'orig_mean': float(np.mean([i['size'] for i in oi]) / 1e6),
                             'hn_mean': float(np.mean([i['size'] for i in hi]) / 1e6)},
            'auc_orig_vs_hn': {
                'file_size': auc([i['size'] for i in hi], [i['size'] for i in oi]),
                'mean_intensity': auc(hrgb[:, 0], orgb[:, 0]),
                'contrast_std': auc(hrgb[:, 1], orgb[:, 1]),
                'warmth_r_minus_b': auc(hrgb[:, 2], orgb[:, 2]),
            },
            '_sizes': [i['size'] for i in oi],
            '_rgb': orgb.tolist(),
        }
    # train vs test originals: generator/domain shift hints from cheap statistics
    tr, te = out['train'], out['test']
    out['train_vs_test_auc'] = {
        'file_size': auc(tr['_sizes'], te['_sizes']),
        'mean_intensity': auc(np.array(tr['_rgb'])[:, 0], np.array(te['_rgb'])[:, 0]),
        'contrast_std': auc(np.array(tr['_rgb'])[:, 1], np.array(te['_rgb'])[:, 1]),
    }
    for s in ('train', 'test'):
        out[s].pop('_sizes'); out[s].pop('_rgb')
    d = CHAN / 'results' / 'l0_leak_audit'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'audit.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
