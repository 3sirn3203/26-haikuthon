#!/usr/bin/env python
"""B2: contrastive fine-tuning of an open_clip dual encoder on train (val bundles held out).

    python scripts/finetune.py --config configs/exp/b2_siglip2_ft_inbatch.yaml

Config keys (under ``finetune``):
  epochs, lr, batch_size, train_last_blocks (unfreeze the last N blocks of each tower + heads),
  batching: random | bundle   (bundle = whole stage1 8-prompt groups per batch: hard in-batch negatives)
  use_hard_negatives: bool    (train HN images appended as extra negative columns, no matching text)
The checkpoint goes to cache/ckpt/<exp>.pt; the eval config then points retriever.checkpoint at it.
After training, evaluates on val and writes results/<name>/ like run_exp.py (eval_split=val) unless
--no-eval. Test evaluation is a separate run_exp call with the same config.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset

CHAN = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CHAN / 'src'))

from haiku_bl import data as D  # noqa: E402
from haiku_bl.profiling import Profiler, pick_device  # noqa: E402


class Pairs(Dataset):
    def __init__(self, rows, transform, tokenizer):
        self.rows, self.t, self.tok = rows, transform, tokenizer

    def __len__(self):
        return len(self.rows)

    def _img(self, p):
        with Image.open(p) as im:
            return self.t(im.convert('RGB'))

    def __getitem__(self, i):
        r = self.rows[i]
        hn = self._img(r['hn_image']) if r.get('hn_image') else None
        return self._img(r['image']), self.tok([r['prompt']])[0], hn


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    toks = torch.stack([b[1] for b in batch])
    hns = [b[2] for b in batch if b[2] is not None]
    return imgs, toks, (torch.stack(hns) if hns else None)


def threaded_batches(ds, batch_index_lists, n_threads):
    """Thread-pool decoding with one batch of prefetch (DataLoader workers are very slow on macOS)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(n_threads) as pool, ThreadPoolExecutor(1) as pre:
        make = lambda idx: collate(list(pool.map(ds.__getitem__, idx)))  # noqa: E731
        nxt = pre.submit(make, batch_index_lists[0]) if batch_index_lists else None
        for k in range(len(batch_index_lists)):
            cur = nxt.result()
            if k + 1 < len(batch_index_lists):
                nxt = pre.submit(make, batch_index_lists[k + 1])
            yield cur


def batches(rows, bs, mode, rng):
    idx = list(range(len(rows)))
    if mode == 'bundle':                       # keep each 8-prompt bundle inside one batch
        groups = {}
        for i, r in enumerate(rows):
            groups.setdefault(r['bundle'] or f'easy:{r["pid"]}', []).append(i)
        gl = list(groups.values())
        rng.shuffle(gl)
        out, cur = [], []
        for g in gl:
            if cur and len(cur) + len(g) > bs:
                out.append(cur); cur = []
            cur += g
        if cur:
            out.append(cur)
        return out
    rng.shuffle(idx)
    return [idx[s:s + bs] for s in range(0, len(idx), bs)]


def unfreeze(model, n_last):
    for p in model.parameters():
        p.requires_grad = False
    towers = []
    v = getattr(model, 'visual', None)
    if v is not None:
        blocks = getattr(getattr(v, 'trunk', v), 'blocks', None) or getattr(getattr(v, 'transformer', None), 'resblocks', None)
        towers.append((v, blocks))
    t = getattr(model, 'text', model)
    tblocks = getattr(getattr(t, 'transformer', None), 'resblocks', None) or \
        getattr(getattr(getattr(t, 'transformer', None), 'encoder', None), 'layer', None)
    towers.append((t, tblocks))
    for _, blocks in towers:
        if blocks is not None:
            for b in list(blocks)[-n_last:]:
                for p in b.parameters():
                    p.requires_grad = True
    # output-side heads/norms/projections/logit scale
    for name, p in model.named_parameters():
        if any(k in name for k in ('proj', 'head', 'ln_post', 'ln_final', 'norm', 'logit_scale', 'logit_bias',
                                   'text_projection', 'attn_pool')) and not any(k in name for k in ('blocks.', 'resblocks.')) \
                and not any(k in name for k in ('patch_embed', 'token_embedding', 'pos_embed', 'positional_embedding')):
            # input-side embeddings stay frozen: unfreezing them makes autograd keep every block's activations
            p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def loss_fn(model, img_f, txt_f, hn_f):
    """SigLIP sigmoid loss if the model has a logit bias, else CLIP InfoNCE. HN columns are
    negatives for every text in the batch."""
    cols = img_f if hn_f is None else torch.cat([img_f, hn_f])
    scale = model.logit_scale.exp()
    logits = scale * txt_f @ cols.T
    n = len(txt_f)
    if getattr(model, 'logit_bias', None) is not None:
        logits = logits + model.logit_bias
        labels = -torch.ones_like(logits)
        labels[torch.arange(n), torch.arange(n)] = 1
        return -F.logsigmoid(labels * logits).sum() / n
    tgt = torch.arange(n, device=logits.device)
    l_t2i = F.cross_entropy(logits, tgt)
    l_i2t = F.cross_entropy(scale * img_f @ txt_f.T, tgt)
    return (l_t2i + l_i2t) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--no-eval', action='store_true')
    ap.add_argument('--max-steps', type=int, default=None, help='smoke test')
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    ft = cfg['finetune']
    paths = D.load_paths()
    device = pick_device(cfg.get('device', 'auto'))
    seed = ft.get('seed', 0)
    torch.manual_seed(seed); rng = random.Random(seed)

    import open_clip
    r = cfg['retriever']
    model, pre_train, pre_val = open_clip.create_model_and_transforms(r['model'], pretrained=r.get('pretrained'),
                                                                     device=device)
    tok = open_clip.get_tokenizer(r['model'])
    rows = D.train_pairs(paths, exclude_val=True)
    if not ft.get('use_hard_negatives'):
        for x in rows:
            x['hn_image'] = None
    n_hn = sum(x['hn_image'] is not None for x in rows)
    params = unfreeze(model, ft.get('train_last_blocks', 4))
    print(f'train pairs {len(rows)} (HN {n_hn}), trainable {sum(p.numel() for p in params)/1e6:.1f}M', flush=True)
    opt = torch.optim.AdamW(params, lr=ft.get('lr', 1e-5), weight_decay=ft.get('weight_decay', 0.05))
    ds = Pairs(rows, pre_val, tok)            # eval transform: no augmentation (synthetic, fixed framing)
    steps_per_epoch = math.ceil(len(rows) / ft['batch_size'])
    total = min(a.max_steps or 10**9, ft['epochs'] * steps_per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=ft.get('lr', 1e-5), total_steps=total,
                                                pct_start=0.1, anneal_strategy='cos')
    dtype = torch.bfloat16 if device.type in ('cuda', 'mps') else torch.float32
    model.train()
    prof = Profiler(device)                    # peak training memory (CUDA: max allocated; MPS: sampled)
    step, t0, log = 0, time.time(), []
    for ep in range(ft['epochs']):
        bl = batches(rows, ft['batch_size'], ft.get('batching', 'random'), rng)
        for imgs, toks, hns in threaded_batches(ds, bl, ft.get('num_workers', 8)):
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type != 'cpu'):
                allimg = imgs if hns is None else torch.cat([imgs, hns])
                f_img = F.normalize(model.encode_image(allimg.to(device)).float(), dim=-1)
                f_txt = F.normalize(model.encode_text(toks.to(device)).float(), dim=-1)
                loss = loss_fn(model, f_img[:len(imgs)], f_txt, f_img[len(imgs):] if hns is not None else None)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            prof.sample()
            if device.type == 'mps':
                # bundle batches vary in size; without this the MPS allocator cache keeps growing
                # (14.8 -> 29.5 GB in 90 steps) until the machine swaps
                torch.mps.empty_cache()
            if step % 10 == 0 or step == 1:
                print(f'ep {ep} step {step}/{total} loss {loss.item():.4f} {time.time()-t0:.0f}s '
                      f'peak {prof.peak_gb:.1f}GB', flush=True)
            log.append(float(loss.item()))
            if step >= total:
                break
        if step >= total:
            break
    train_sec = time.time() - t0
    ck = paths['cache_dir'] / 'ckpt' / f"{cfg['exp_id'].split('/', 1)[1]}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    # only the trained tensors (retriever loads pretrained weights, then this with strict=False)
    trained = {n for n, p_ in model.named_parameters() if p_.requires_grad}
    torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trained}}, ck)
    meta = {'train_sec': train_sec, 'steps': step, 'device': str(device), 'final_loss': float(np.mean(log[-10:])),
            'train_peak_mem_gb': prof.peak_gb, 'batch_size': ft['batch_size'],
            'n_train_pairs': len(rows), 'n_train_hn': n_hn, 'checkpoint': str(ck)}
    (ck.with_suffix('.json')).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    del model
    if not a.no_eval:
        for split in ('val', 'test'):
            ecfg = dict(cfg)
            ecfg.pop('finetune', None)
            ecfg['retriever'] = dict(r, checkpoint=str(ck))
            ecfg['eval_split'] = split
            ecfg['exp_id'] = cfg['exp_id'] + ('__val' if split == 'val' else '')
            ecfg['finetune_meta'] = meta
            p = paths['cache_dir'] / 'ckpt' / f'eval_{split}.yaml'
            p.write_text(yaml.safe_dump(ecfg))
            subprocess.run([sys.executable, str(CHAN / 'scripts' / 'run_exp.py'), '--config', str(p)], check=True)


if __name__ == '__main__':
    main()
