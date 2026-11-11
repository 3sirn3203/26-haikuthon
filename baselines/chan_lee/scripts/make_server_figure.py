#!/usr/bin/env python
"""Overview figure rescaled to the submission server (NVIDIA L4 24GB, 4 CPU cores, 15-minute GPU limit).

    python scripts/make_server_figure.py   -> results/figures/all_organizers_server15min.png

For every configuration of make_overview_figure.py (same letter IDs):
  * full-test inference time on the L4 (8,000 queries x 15,000 images), measured where an organizer
    measured it on an L4, otherwise estimated;
  * the score that configuration is estimated to reach inside the 15-minute limit.

ESTIMATION RULES (all labelled "est." in the figure)
  stage-1 time   L4 measurements are used as-is (seohyun_jang, junhyun_park). Models timed only on the
                 M4 Max (MPS) are scaled by 0.37 = 294 s / 790 s, the L4/MPS ratio of SigLIP2 so400m image
                 encoding -- a single calibration point.
  rerank time    8,000 x K x t_pair + 50 s reranker load. t_pair on L4: Qwen3.5-4B 0.153 s (measured,
                 seohyun_jang), Qwen3.5-2B 0.067 s (measured, junhyun_park), checklist question +0.149 s
                 (measured, seohyun_jang), Qwen3-VL-Reranker-2B 0.078 s (MPS 0.153 s x 0.51, the L4/MPS
                 ratio of Qwen3.5-4B).
  15-min score   stage-1-only: unchanged if the run fits, otherwise not achievable.
                 two-stage: only f = (900 - stage-1 time - 50) / (8,000 x K x t_pair) of the queries can be
                 reranked; score = stage-1 full-test score + f x (reranked - stage-1 score on the same
                 subset). Assumes the reranked queries are a random sample and K is kept as configured.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_overview_figure import CHAN, GRID, INK, INK2, RR, STAGE, SURFACE, build_rows  # noqa: E402

OUT = CHAN / 'results' / 'figures' / 'all_organizers_server15min.png'
LIMIT, LOAD = 900.0, 50.0
MPS_TO_L4 = 294 / 790          # SigLIP2 so400m image encoding, L4 (seohyun_jang) / M4 Max (chan_lee)
SIGLIP2_ZS, SIGLIP2_FT, FGCLIP = 366.0, 345.0, 80.0     # measured stage-1 totals on L4 (s)
T4B, T2B, TQ2, TRR2B = 0.153, 0.067, 0.149, 0.153 * (0.153 / 0.300)

# ID -> (stage-1 seconds on L4, basis)
STAGE1 = {
    'A': (615 * MPS_TO_L4, 'est'), 'B': (None, 'site'), 'C': (6408 * MPS_TO_L4, 'est'),
    'D': (81.4, 'meas'), 'E': (92.6, 'meas'), 'F': (304.9, 'meas'), 'G': (366, 'meas'), 'H': (SIGLIP2_ZS, 'meas*'),
    'I': (1695 * MPS_TO_L4, 'est'), 'J': (727 * MPS_TO_L4, 'est'),
    'K': (103.2, 'meas'), 'L': (80.2, 'meas'), 'M': (SIGLIP2_FT, 'meas*'), 'O': (2 * SIGLIP2_FT, 'est'),
    'P': (SIGLIP2_FT, 'meas*'), 'Q': (2 * SIGLIP2_FT, 'est'), 'R': (80.1, 'meas'), 'S': (345, 'meas'),
    'T': (SIGLIP2_FT, 'meas*'), 'U': (SIGLIP2_FT, 'meas*'),
}
# ID -> (stage-1 seconds, K, seconds per pair, full-test stage-1 score, stage-1 score on the rerank subset)
RERANK = {
    'N': (SIGLIP2_ZS, 20, TRR2B, 0.4263, None), 'V': (SIGLIP2_ZS, 20, T2B, 0.4263, None),
    'W': (SIGLIP2_ZS, 20, T4B, 0.4263, None), 'X': (SIGLIP2_FT, 5, T4B, 0.6771, None),
    'Y': (SIGLIP2_FT, 5, T4B, 0.6771, None), 'Z': (SIGLIP2_FT, 5, TQ2, 0.6771, 0.6786),
    'AA': (SIGLIP2_FT, 5, T4B + TQ2, 0.6771, 0.6786), 'AB': (SIGLIP2_FT, 10, T4B, 0.6771, 0.6786),
    'AC': (FGCLIP, 10, T2B, 0.6731, 0.7386),
}


def fmt_time(sec):
    return f'{sec / 60:.1f} min' if sec < 5400 else f'{sec / 3600:.1f} h'


def main():
    rows = build_rows()
    for r in rows:
        i = r['id']
        if i in RERANK:
            s1_sec, k, tp, s1_full, s1_sub = RERANK[i]
            s1_sub = s1_sub if s1_sub is not None else r['s1_same']
            pairs = 8000 * k * tp
            r['time'] = s1_sec + LOAD + pairs
            r['basis'] = 'est'
            r['frac'] = max(0.0, min(1.0, (LIMIT - s1_sec - LOAD) / pairs))
            r['score15'] = s1_full + r['frac'] * (r['score'] - s1_sub)
            r['note'] = f"{r['frac'] * 100:.0f}% of queries reranked"
        else:
            sec, basis = STAGE1[i]
            r['time'], r['basis'], r['frac'] = sec, basis, None
            fits = sec is None or sec <= LIMIT
            r['score15'] = r['score'] if fits else None
            r['note'] = 'fits' if fits else 'does not fit'
    rows.sort(key=lambda r: (r['score15'] is not None, r['score15'] or 0))

    plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                         'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK,
                         'axes.edgecolor': GRID, 'font.size': 10, 'hatch.color': SURFACE, 'hatch.linewidth': 1.5})
    n = len(rows)
    fig = plt.figure(figsize=(22, 0.36 * n + 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.03)
    ax = fig.add_subplot(gs[0])
    for y, r in enumerate(rows):
        c = STAGE[r['stage']][1]
        if r['score15'] is not None:
            ax.barh(y, r['score15'], height=0.62, color=c, zorder=2, hatch='///' if r['val'] else None,
                    edgecolor=SURFACE if r['val'] else 'none')
        # score without a time limit, as reported by the organizer
        ax.plot([r['score']] * 2, [y - 0.36, y + 0.36], color=INK, linewidth=1.6, zorder=4)
        if r['score15'] is None:
            label = 'over 15 min'
        else:
            label = f"{r['score15']:.3f}" + ('' if r['frac'] is None else ' est.')
        x_text = max(r['score'], r['score15'] or 0) + 0.012
        ax.text(x_text, y, label, va='center', fontsize=9, color=INK)
    ax.set_yticks(range(n), [r['id'] for r in rows], fontweight='bold', fontsize=10.5)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('score = (Recall@1 + NDCG@5) / 2')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in STAGE.values()]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor=RR, hatch='///', edgecolor=SURFACE),
                plt.Line2D([0], [0], color=INK, linewidth=1.6)]
    ax.legend(handles, [s for s, _ in STAGE.values()] +
              ['rerank gain measured on val only (optimistic)', 'score with no time limit (as reported)'],
              loc='lower right', frameon=False, fontsize=9.5, title='bar = estimated score within 15 min',
              title_fontsize=9.5, alignment='left')

    tx = fig.add_subplot(gs[1])
    tx.set_ylim(-0.7, n - 0.3)
    tx.set_xlim(0, 1)
    tx.axis('off')
    X = dict(id=0.015, by=0.06, cfg=0.165, time=0.585, basis=0.675, lim=0.755, nolim=0.895)
    for key, h in (('id', 'ID'), ('by', 'by'), ('cfg', 'configuration'), ('time', 'full-test time\non L4'),
                   ('basis', 'time is'), ('lim', 'within the 15-min limit'), ('nolim', 'score, no\ntime limit')):
        tx.text(X[key], n - 0.25, h, fontsize=9.5, color=INK2, fontweight='bold', va='bottom', linespacing=1.1)
    basis_text = {'meas': 'measured', 'meas*': 'measured*', 'est': 'estimated', 'site': 'not reported'}
    for y, r in enumerate(rows):
        if y % 2 == 0:
            tx.axhspan(y - 0.5, y + 0.5, color='#f1f0ec', zorder=0)
        tx.text(X['id'], y, r['id'], fontweight='bold', va='center', fontsize=10.5)
        tx.text(X['by'] - 0.012, y, '■', color=STAGE[r['stage']][1], va='center', ha='right', fontsize=11)
        tx.text(X['by'], y, r['author'], va='center', fontsize=9.3, color=INK2)
        tx.text(X['cfg'], y, r['text'], va='center', fontsize=9.6)
        over = r['time'] is not None and r['time'] > LIMIT
        tx.text(X['time'], y, 'n/a' if r['time'] is None else fmt_time(r['time']), va='center', fontsize=9.6,
                fontweight='bold' if over else 'normal')
        tx.text(X['basis'], y, basis_text[r['basis']], va='center', fontsize=8.8, color=INK2)
        note = r['note'] if r['basis'] != 'site' else 'fits (ran on the site)'
        tx.text(X['lim'], y, note, va='center', fontsize=9.3, color=INK if over else INK2)
        scope = '' if r['n'] == 8000 else ' (val)' if r['val'] else f" (n={r['n']:,})"
        tx.text(X['nolim'], y, f"{r['score']:.3f}{scope}", va='center', fontsize=9.3, color=INK2)
    fig.suptitle('HAIKUTHON baselines rescaled to the submission server: NVIDIA L4 24GB, 4 CPU cores, '
                 '15-minute GPU limit (estimates)', x=0.012, ha='left', fontsize=14, fontweight='bold', y=0.988)
    fig.text(0.012, 0.012,
             'Time = full test inference (8,000 queries x 15,000 images), setup excluded. "measured" = timed on an L4 by '
             'seohyun_jang / junhyun_park; "estimated" = MPS time x 0.37 for stage 1, or 8,000 x K x seconds-per-pair '
             '+ 50 s load for reranking.\n'
             'Two-stage bars assume only the affordable share of queries (random sample, K as configured) is reranked: '
             'score = stage-1 score + share x measured rerank gain. Participants can do better by lowering K, '
             'reranking only uncertain queries, or batching.\n'
             'AC uses a rerank gain measured on the easier val set, so its bar is optimistic. FG-CLIP timings are warm '
             'runs; a cold first run of D took 250 s. measured* = the same model was timed on an L4 by another organizer.',
             fontsize=8.5, color=INK2, linespacing=1.5, va='bottom')
    fig.subplots_adjust(left=0.03, right=0.995, top=0.925, bottom=0.115)
    fig.savefig(OUT, dpi=150)
    for r in sorted(rows, key=lambda r: r['id'] if len(r['id']) == 1 else 'Z' + r['id']):
        print(f"{r['id']:>2} {r['score']:.3f} -> {'  n/a' if r['score15'] is None else format(r['score15'], '.3f')}  "
              f"time {'n/a' if r['time'] is None else fmt_time(r['time']):>9} [{r['basis']}] {r['note']}")


if __name__ == '__main__':
    main()
