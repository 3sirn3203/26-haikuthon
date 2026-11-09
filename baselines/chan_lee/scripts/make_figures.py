#!/usr/bin/env python
"""Figures for REPORT.md from results/*/metrics.json -> results/figures/*.png

    python scripts/make_figures.py

Palette: first three categorical slots of the dataviz reference palette (validated all-pairs,
light mode); every figure has a matching table in REPORT.md (contrast relief for slot 3).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

CHAN = Path(__file__).resolve().parents[1]
R = CHAN / 'results'
OUT = R / 'figures'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'legend.frameon': False, 'lines.linewidth': 2, 'lines.markersize': 8,
})


def load(name):
    return json.loads((R / name / 'metrics.json').read_text())


def b1_models():
    names = {'b1_pecore_l14_336': 'PE-Core-L/14@336', 'b1_dfn5b_h14_378': 'DFN5B ViT-H/14@378',
             'b1_siglip2_so400m384': 'SigLIP2 so400m@384', 'b1_qwen3vlemb_2b': 'Qwen3-VL-Embedding-2B',
             'b1_clip_l14_336': 'OpenAI CLIP-L/14@336'}
    rows = [(lbl, load(n)['scores']) for n, lbl in names.items() if (R / n).exists()]
    rows.sort(key=lambda r: r[1]['score'])
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y = range(len(rows))
    sc = [r[1]['score'] for r in rows]
    lo = [r[1]['score'] - r[1]['ci95'][0] for r in rows]
    hi = [r[1]['ci95'][1] - r[1]['score'] for r in rows]
    ax.barh(list(y), sc, height=0.55, color=S1, zorder=2)
    ax.errorbar(sc, list(y), xerr=[lo, hi], fmt='none', ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
    for i, r in zip(y, rows):
        ax.text(r[1]['ci95'][1] + 0.008, i, f"{r[1]['score']:.3f}", va='center', color=INK, fontsize=9)
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.set_xlim(0, 0.55)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('score = (R@1 + NDCG@5) / 2  ·  8,000 queries, 15,000 images')
    ax.set_title('B1: zero-shot dual encoders (whiskers: 95% CI)')
    fig.tight_layout()
    fig.savefig(OUT / 'b1_models.png', dpi=160)


def b3_k_curve():
    series = [('b3_siglip2__qwen35_4b', 'Qwen3.5-4B (yes/no)', S1),
              ('b3_siglip2__qwen35_2b', 'Qwen3.5-2B (yes/no)', S2),
              ('b3_siglip2__qwen3vlrr_2b', 'Qwen3-VL-Reranker-2B', S3)]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    base = None
    for name, lbl, c in series:
        ks = [k for k in (5, 10, 20) if (R / f'{name}_k{k}').exists()]
        if not ks:
            continue
        ms = [load(f'{name}_k{k}') for k in ks]
        base = ms[0]['stage1_scores_same_subset']['score']
        xs, ys = [0] + ks, [base] + [m['scores']['score'] for m in ms]
        ax.plot(xs, ys, color=c, marker='o', label=lbl, zorder=3)
        ax.annotate(f'{lbl}  {ys[-1]:.3f}', (xs[-1], ys[-1]), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=9, color=INK)
    if base is not None:
        ax.axhline(base, color=INK2, linestyle='--', linewidth=1, zorder=1)
        ax.text(20.5, base - 0.02, f'stage 1 only (SigLIP2) {base:.3f}', fontsize=9, color=INK2)
    ax.set_xticks([0, 5, 10, 20], ['0\n(no rerank)', '5', '10', '20'])
    ax.set_xlim(-1, 31)
    ax.set_ylim(0.4, 0.8)
    ax.set_xlabel('K = number of stage-1 candidates reranked (top-K)')
    ax.set_ylabel('score')
    ax.set_title('B3: two-stage reranking vs K (stratified 400-query test subset)')
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'b3_k_curve.png', dpi=160)


def pareto():
    pts = []
    for n in ('b1_pecore_l14_336', 'b1_dfn5b_h14_378', 'b1_siglip2_so400m384', 'b1_qwen3vlemb_2b', 'b1_clip_l14_336'):
        if (R / n).exists():
            m = load(n)
            pts.append(('B1 stage 1 only (measured)', m['timing']['total_sec'] / 60, m['scores']['score'], n[3:]))
    for name, fam in (('b3_siglip2__qwen35_4b', 'B3 + Qwen3.5 2B/4B (estimated)'), ('b3_siglip2__qwen35_2b', 'B3 + Qwen3.5 2B/4B (estimated)'),
                      ('b3_siglip2__qwen3vlrr_2b', 'B3 + Qwen3-VL-Reranker-2B (estimated)')):
        for k in (5, 10, 20):
            if (R / f'{name}_k{k}').exists():
                m = load(f'{name}_k{k}')
                tag = name.split('__')[1].replace('qwen35_', '').replace('qwen3vlrr_', 'rr')
                pts.append((fam, m['timing']['total_sec'] / 60, m['scores']['score'], f'{tag} K{k}'))
    colors = {'B1 stage 1 only (measured)': S1, 'B3 + Qwen3.5 2B/4B (estimated)': S2, 'B3 + Qwen3-VL-Reranker-2B (estimated)': S3}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for fam, c in colors.items():
        P = [p for p in pts if p[0] == fam]
        ax.scatter([p[1] for p in P], [p[2] for p in P], s=56, color=c, edgecolor=SURFACE, linewidth=2,
                   label=fam, zorder=3)
    offsets = {'siglip2_so400m384': (5, -12)}
    for fam, t, s, lbl in pts:
        if fam.startswith('B1') or lbl.endswith('K20') or lbl.endswith('K5'):
            ax.annotate(lbl, (t, s), xytext=offsets.get(lbl, (5, 4)), textcoords='offset points', fontsize=8,
                        color=INK2)
    ax.set_xscale('log')
    ax.set_xticks([10, 30, 60, 120, 240, 480, 960], ['10', '30', '60', '120', '240', '480', '960'])
    ax.set_xlabel('full-test inference time (minutes, Mac Studio M4 Max MPS, log scale)')
    ax.set_ylabel('score')
    ax.set_title('Score vs inference time (MPS, not CUDA)')
    ax.legend(loc='lower right', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / 'pareto_mps.png', dpi=160)


def hn_ablation():
    f = R / 'a1_hn_ablation' / 'b1_siglip2_so400m384.json'
    if not f.exists():
        return
    g = json.loads(f.read_text())['grid']
    xs = sorted(int(k) for k in g)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for key, lbl, c in (('all', 'all queries', S1), ('own_hn_absent', 'queries whose own HN is not in the pool', S2),
                        ('own_hn_in_pool', 'queries whose own HN is in the pool', S3)):
        pts = [(x, g[str(x)][key]) for x in xs if g[str(x)][key] is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, marker='o', label=lbl, zorder=3)
        ax.annotate(f'{pts[-1][1]:.3f}', pts[-1], xytext=(6, 0), textcoords='offset points', va='center',
                    fontsize=9, color=INK)
    ax.set_xlabel('number of hard-negative images in the test candidate pool')
    ax.set_ylabel('score (SigLIP2 zero-shot)')
    ax.set_xlim(-300, 7900)
    ax.set_title('A1: difficulty vs number of hard negatives')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'hn_ablation.png', dpi=160)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (b1_models, b3_k_curve, pareto, hn_ablation):
        f()
        print('ok', f.__name__)


if __name__ == '__main__':
    main()
