#!/usr/bin/env python
"""One overview figure of every organizer's honest configurations -> results/figures/all_organizers.png

    python scripts/make_overview_figure.py

Scores are read from baselines/*/results/*/metrics.json. Variants that exist only as rows in a
README table (seohyun_jang B6/B7 variants) are listed in README_ONLY with their source.
Loophole runs and the B0 random baseline are excluded.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

CHAN = Path(__file__).resolve().parents[1]
BASE = CHAN.parent
OUT = CHAN / 'results' / 'figures' / 'all_organizers.png'
ZS, FT, RR = '#2a78d6', '#eb6834', '#1baf7a'      # dataviz reference palette slots 1-3 (validated all-pairs)
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
STAGE = {'zs': ('zero-shot stage 1', ZS), 'ft': ('fine-tuned stage 1', FT), 'rr': ('two-stage (stage 1 + VLM rerank)', RR)}

# (author, results dir name, stage, configuration text)
FROM_JSON = [
    ('chan_lee', 'b1_clip_l14_336', 'zs', 'OpenAI CLIP-L/14@336'),
    ('chan_lee', 'b1_qwen3vlemb_2b', 'zs', 'Qwen3-VL-Embedding-2B'),
    ('chan_lee', 'b1_siglip2_so400m384', 'zs', 'SigLIP2 so400m@384 (MPS)'),
    ('chan_lee', 'b1_dfn5b_h14_378', 'zs', 'DFN5B ViT-H/14@378'),
    ('chan_lee', 'b1_pecore_l14_336', 'zs', 'PE-Core-L/14@336'),
    ('chan_lee', 'b2_siglip2_inbatch', 'ft', 'SigLIP2 FT, random batches'),
    ('chan_lee', 'b2_siglip2_bundle_hn', 'ft', 'SigLIP2 FT, bundle batches + train HN images'),
    ('chan_lee', 'b2_siglip2_bundle', 'ft', 'SigLIP2 FT, bundle batches (MPS)'),
    ('chan_lee', 'b3_siglip2__qwen3vlrr_2b_k20', 'rr', 'zero-shot SigLIP2 + Qwen3-VL-Reranker-2B, K=20'),
    ('chan_lee', 'b3_siglip2__qwen35_2b_k20', 'rr', 'zero-shot SigLIP2 + Qwen3.5-2B yes/no, K=20'),
    ('chan_lee', 'b3_siglip2__qwen35_4b_k20', 'rr', 'zero-shot SigLIP2 + Qwen3.5-4B yes/no, K=20'),
    ('chan_lee', 'b3_siglip2_bundle__qwen35_4b_k5', 'rr', 'FT SigLIP2 (bundle) + Qwen3.5-4B yes/no, K=5'),
    ('ichan_jung', 'b1_pickscore_zs', 'zs', 'PickScore v1 (CLIP-H/14)'),
    ('junhyun_park', 'b1_fgclip_base_zs', 'zs', 'FG-CLIP base (0.15B), 77 tokens'),
    ('junhyun_park', 'b1_fgclip_base_long_zs', 'zs', 'FG-CLIP base, long-caption mode (248 tokens)'),
    ('junhyun_park', 'b1_fgclip_large_zs', 'zs', 'FG-CLIP large (0.43B)'),
    ('junhyun_park', 'b2_ft_hn2', 'ft', 'FG-CLIP base FT, random batches + rule-based text HN'),
    ('junhyun_park', 'b2_ft_ctrl', 'ft', 'FG-CLIP base FT, random batches'),
    ('junhyun_park', 'b3_ft_bundle_bval', 'ft', 'FG-CLIP base FT, bundle batches'),
    ('seohyun_jang', 'b1_siglip2_so400m384_l4', 'zs', 'SigLIP2 so400m@384 (L4 replication)'),
    ('seohyun_jang', 'b2_siglip2_bundle_l4', 'ft', 'SigLIP2 FT, bundle batches (L4 replication)'),
    ('seohyun_jang', 'b3_siglip2_bundle_l4__qwen35_4b_full_k5', 'rr', 'FT SigLIP2 + Qwen3.5-4B yes/no, K=5'),
]
# (author, stage, configuration, score, n_queries, source) -- values that are not a metrics.json headline
README_ONLY = [
    ('seohyun_jang', 'ft', 'WiSE-FT weight mix zero-shot:FT = 0.5:0.5', 0.5813, 8000, 'README B6'),
    ('seohyun_jang', 'ft', 'score ensemble zero-shot + FT', 0.6141, 8000, 'README B6'),
    ('seohyun_jang', 'ft', 'score ensemble WiSE-FT + FT', 0.6507, 8000, 'README B6'),
    ('seohyun_jang', 'rr', 'FT SigLIP2 + 4B, K=10, fused with stage-1 score', 0.8409, 1183, 'README B5'),
    ('seohyun_jang', 'rr', 'FT SigLIP2 + 4B, K=5, checklist-style question', 0.8276, 727, 'README B7'),
    ('seohyun_jang', 'rr', 'FT SigLIP2 + 4B, K=5, two questions + stage-1 score', 0.8390, 727, 'README B7'),
]
# val-only result: different (easier) pool, shown hatched and not comparable with test bars
VAL_ONLY = [('junhyun_park', 'rr', 'FT FG-CLIP + Qwen3.5-2B LoRA-finetuned reranker, K=10', 0.9386, 500, 'notes.md (val)')]


def letter(i):
    return chr(65 + i) if i < 26 else 'A' + chr(65 + i - 26)


def build_rows():
    """All configurations with stable letter IDs (assigned by ascending headline score)."""
    rows = []
    for author, name, stage, text in FROM_JSON:
        m = json.loads((BASE / author / 'results' / name / 'metrics.json').read_text())
        rows.append(dict(author=author, stage=stage, text=text, score=m['scores']['score'],
                         n=m.get('n_queries', 8000), val=False, name=name,
                         s1_same=(m.get('stage1_scores_same_subset') or {}).get('score')))
    for author, stage, text, score, n, _ in README_ONLY:
        rows.append(dict(author=author, stage=stage, text=text, score=score, n=n, val=False))
    for author, stage, text, score, n, _ in VAL_ONLY:
        rows.append(dict(author=author, stage=stage, text=text, score=score, n=n, val=True))
    rows.sort(key=lambda r: (r['val'], r['score']))
    for i, r in enumerate(rows):
        r['id'] = letter(i)
    return rows


def main():
    rows = build_rows()
    plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                         'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK,
                         'axes.edgecolor': GRID, 'font.size': 10, 'hatch.color': SURFACE, 'hatch.linewidth': 1.5})
    n = len(rows)
    fig = plt.figure(figsize=(19, 0.36 * n + 2.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.04)
    ax = fig.add_subplot(gs[0])
    y = list(range(n))
    for i, r in enumerate(rows):
        ax.barh(i, r['score'], height=0.62, color=STAGE[r['stage']][1], zorder=2,
                hatch='///' if r['val'] else None, edgecolor=SURFACE if r['val'] else 'none')
        tag = f"{r['score']:.3f}" + ('  (val)' if r['val'] else '' if r['n'] == 8000 else f"  (n={r['n']:,})")
        ax.text(r['score'] + 0.008, i, tag, va='center', fontsize=9, color=INK)
    ax.set_yticks(y, [r['id'] for r in rows], fontweight='bold', fontsize=10.5)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('score = (Recall@1 + NDCG@5) / 2')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in STAGE.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=RR, hatch='///', edgecolor=SURFACE))
    ax.legend(handles, [s for s, _ in STAGE.values()] + ['val set only (easier pool, not comparable)'],
              loc='lower right', frameon=False, fontsize=9.5)

    tx = fig.add_subplot(gs[1])
    tx.set_ylim(-0.7, n - 0.3)
    tx.set_xlim(0, 1)
    tx.axis('off')
    cols = [(0.03, 'ID'), (0.09, 'by'), (0.25, 'configuration'), (0.86, 'evaluated on')]
    for x, h in cols:
        tx.text(x, n - 0.25, h, fontsize=9.5, color=INK2, fontweight='bold', va='bottom')
    for i, r in enumerate(rows):
        if i % 2 == 0:
            tx.axhspan(i - 0.5, i + 0.5, color='#f1f0ec', zorder=0)
        scope = 'val 500' if r['val'] else 'test 8,000' if r['n'] == 8000 else f"test subset {r['n']:,}"
        tx.text(0.03, i, r['id'], fontweight='bold', va='center', fontsize=10.5)
        tx.text(0.045 + 0.03, i, '■', color=STAGE[r['stage']][1], va='center', ha='right', fontsize=11)
        tx.text(0.09, i, r['author'], va='center', fontsize=9.5, color=INK2)
        tx.text(0.25, i, r['text'], va='center', fontsize=9.8)
        tx.text(0.86, i, scope, va='center', fontsize=9.5, color=INK2)
    fig.suptitle('HAIKUTHON baselines: every honest configuration from all organizers (sorted by score)',
                 x=0.012, ha='left', fontsize=14, fontweight='bold', y=0.985)
    fig.text(0.012, 0.012, 'Test pool: 8,000 queries, 15,000 candidate images. Subset scores (n<8,000) carry wider '
             'confidence intervals and are not strictly comparable with full-test scores. '
             'Rule-violating (loophole) runs and the random baseline are excluded.', fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.94, bottom=0.075)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    for r in rows:
        print(r['id'], f"{r['score']:.4f}", r['n'], r['author'], '|', r['text'])


if __name__ == '__main__':
    main()
