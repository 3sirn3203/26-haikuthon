"""Prompt/image loading and the splits used for evaluation.

Vocabulary
- query: one prompt with exactly one ground-truth image.
- pool: the candidate images every query in an EvalSet is ranked against.
- bundle: stage1's 8-prompt contrast group (same background/rendering/total count).
  Prompts outside any bundle are stage1 "baseline" (easy) items.

Internal image ids look like ``test/orig/0123`` or ``test/hn/0123``. They exist only
for bookkeeping and are never shown to a model: models only see pixels and prompt text.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

CHAN_ROOT = Path(__file__).resolve().parents[2]          # baselines/chan_lee
REPO_ROOT = CHAN_ROOT.parents[1]                          # haikuthon/
SPLITS = ('train', 'test')


# --------------------------------------------------------------------- config
_ENV = re.compile(r'\$\{(\w+)(?::-([^}]*))?\}')


def _expand(value):
    if isinstance(value, str):
        value = _ENV.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ''), value)
        return os.path.expanduser(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_paths(path: str | Path | None = None) -> dict:
    path = Path(path or CHAN_ROOT / 'configs' / 'paths.yaml')
    cfg = _expand(yaml.safe_load(path.read_text(encoding='utf-8')))
    cfg['data_root'] = repo_path(cfg['data_root'])
    cfg['cache_dir'] = repo_path(cfg['cache_dir'])
    return cfg


def repo_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def read_prompts(path) -> dict[str, str]:
    out = {}
    with open(repo_path(path), encoding='utf-8') as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row['id']] = row['prompt']
    return out


# -------------------------------------------------------------------- bundles
def bundles(split: str, cfg: dict) -> dict[str, str | None]:
    """prompt id -> bundle id (``None`` for stage1 baseline/easy items).

    Uses stage1's own reconstruction (validation.partition) read-only. Bytecode writing is
    disabled so importing does not create files under stage1_components/.
    """
    cache = cfg['cache_dir'] / 'meta' / f'{split}_bundles.json'
    if cache.exists():
        return json.loads(cache.read_text())
    sys.dont_write_bytecode = True
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from stage1_components.model import default_catalog
    from stage1_components.validation import partition, read_jsonl
    rows = read_jsonl(repo_path(cfg['components'][split]))
    easy, groups = partition(rows, default_catalog())
    out = {s['prompt_id']: None for s in easy}
    for g in groups:
        gid = f'{split}:b' + min(s['prompt_id'] for s in g)
        for s in g:
            out[s['prompt_id']] = gid
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, sort_keys=True))
    return out


# --------------------------------------------------------------------- images
def find_image(cfg: dict, kind: str, pid: str) -> Path | None:
    for pattern in cfg['images'][kind]:
        p = cfg['data_root'] / pattern.format(id=pid)
        if p.exists():
            return p
    return None


@dataclass
class Image:
    iid: str            # internal id, e.g. "test/hn/0123"
    path: Path | None
    kind: str           # "orig" | "hn"
    source_pid: str     # prompt id the image was generated for (orig) or perturbed from (hn)


@dataclass
class Query:
    qid: str            # "test:0123"
    pid: str
    prompt: str
    gt: str             # internal image id of the ground truth
    bundle: str | None
    tags: set = field(default_factory=set)   # e.g. {"public"} / {"private"} / {"val"}


@dataclass
class EvalSet:
    name: str
    queries: list[Query]
    pool: list[Image]
    public_provisional: bool = False   # True when Public/Private uses the provisional rule

    def __post_init__(self):
        index = {im.iid: i for i, im in enumerate(self.pool)}
        self.gt_index = np.array([index[q.gt] for q in self.queries], dtype=np.int64)
        self.image_index = index

    @property
    def prompts(self):
        return [q.prompt for q in self.queries]

    def masks(self) -> dict[str, np.ndarray]:
        """Boolean masks over queries for every sub-split we report."""
        tags = sorted({t for q in self.queries for t in q.tags})
        out = {t: np.array([t in q.tags for q in self.queries]) for t in tags}
        out['easy'] = np.array([q.bundle is None for q in self.queries])
        out['bundle'] = ~out['easy']
        return out

    def missing_images(self) -> list[str]:
        return [im.iid for im in self.pool if im.path is None or not Path(im.path).exists()]


# --------------------------------------------------------------------- splits
def _public_ids(cfg, test_bundles) -> tuple[set[str], bool]:
    """Returns (public ids, provisional?)."""
    f = cfg.get('public_ids')
    if f:
        ids = {l.strip() for l in Path(f).read_text().splitlines() if l.strip()}
        return ids, False
    rng = np.random.default_rng(cfg['split_seed'])
    easy = sorted(p for p, b in test_bundles.items() if b is None)
    rest = sorted(p for p, b in test_bundles.items() if b is not None)
    extra = rng.choice(rest, size=max(0, cfg['public_size'] - len(easy)), replace=False)
    return set(easy) | set(extra.tolist()), True


def val_split(cfg) -> tuple[set[str], set[str]]:
    """Split train prompt ids into (fit, val) by whole bundles so no bundle straddles both.

    Groups whose ids have a train HN image are taken first until ``val_hn_target`` HN images
    land in val; the remaining HN images stay available for fine-tuning (B2b / L2).
    """
    b = bundles('train', cfg)
    hn_ids = set(read_prompts(cfg['prompts']['train_hn']))
    groups: dict[str, list[str]] = {}
    for pid, gid in b.items():
        groups.setdefault(gid or f'easy:{pid}', []).append(pid)
    keys = sorted(groups)
    rng = np.random.default_rng(cfg['split_seed'])
    rng.shuffle(keys)
    with_hn = [k for k in keys if any(p in hn_ids for p in groups[k])]
    without = [k for k in keys if k not in set(with_hn)]
    val: list[str] = []
    n_hn = 0
    for k in with_hn:
        if n_hn >= cfg['val_hn_target'] or len(val) + len(groups[k]) > cfg['val_size']:
            break
        val += groups[k]
        n_hn += sum(p in hn_ids for p in groups[k])
    for k in without:
        if len(val) + len(groups[k]) > cfg['val_size']:
            continue
        val += groups[k]
        if len(val) == cfg['val_size']:
            break
    val_set = set(val)
    return set(b) - val_set, val_set


def _images(cfg, kind, pids, internal_kind, split):
    return [Image(f'{split}/{internal_kind}/{p}', find_image(cfg, kind, p), internal_kind, p)
            for p in sorted(pids)]


def build_eval_set(name: str, cfg: dict | None = None, *, include_hn: bool = True) -> EvalSet:
    """name: 'test' (8,000 queries, orig + HN pool) or 'val' (held-out train bundles)."""
    cfg = cfg or load_paths()
    if name == 'test':
        prompts = read_prompts(cfg['prompts']['test'])
        b = bundles('test', cfg)
        public, provisional = _public_ids(cfg, b)
        hn_pids = set(read_prompts(cfg['prompts']['test_hn'])) - public if include_hn else set()
        queries = [Query(f'test:{p}', p, prompts[p], f'test/orig/{p}', b[p],
                         {'public' if p in public else 'private'}) for p in sorted(prompts)]
        pool = _images(cfg, 'test', prompts, 'orig', 'test') + _images(cfg, 'test_hn', hn_pids, 'hn', 'test')
        return EvalSet('test', queries, pool, provisional)
    if name == 'val':
        prompts = read_prompts(cfg['prompts']['train'])
        b = bundles('train', cfg)
        _, val = val_split(cfg)
        hn_pids = (set(read_prompts(cfg['prompts']['train_hn'])) & val) if include_hn else set()
        queries = [Query(f'train:{p}', p, prompts[p], f'train/orig/{p}', b[p], {'val'}) for p in sorted(val)]
        pool = _images(cfg, 'train', val, 'orig', 'train') + _images(cfg, 'train_hn', hn_pids, 'hn', 'train')
        return EvalSet('val', queries, pool)
    raise ValueError(f'unknown eval set {name!r}')


def train_pairs(cfg: dict | None = None, *, exclude_val: bool = True):
    """(prompt, image path, bundle, hn image path or None) for fine-tuning; val bundles excluded."""
    cfg = cfg or load_paths()
    prompts = read_prompts(cfg['prompts']['train'])
    b = bundles('train', cfg)
    fit, _ = val_split(cfg)
    hn = set(read_prompts(cfg['prompts']['train_hn']))
    keep = fit if exclude_val else set(prompts)
    return [dict(pid=p, prompt=prompts[p], image=find_image(cfg, 'train', p), bundle=b[p],
                 hn_image=find_image(cfg, 'train_hn', p) if p in hn else None)
            for p in sorted(keep)]
