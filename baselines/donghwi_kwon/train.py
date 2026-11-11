"""Train the existing ResNet18 + bidirectional GRU on train_v1 only."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
from zipfile import ZipFile, ZIP_STORED

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import load_file, save_file
from model import DualEncoder, encode_text_batch, image_to_tensor, tokenize

TRAIN_FILE_ID = '1CgLYOn8eZ9eORVdoWTRpjFfY3AQKSwtN'
CONFIG = dict(embedding_dim=256, text_hidden=192, max_length=80,
              image_batch_size=96, text_batch_size=16)
REQUIRED = {'model.py', 'weights.safetensors', 'vocab.json', 'config.json'}

def read_records(root):
    root = Path(root).resolve()
    if root.name != 'train_v1':
        raise ValueError('data-root must be the extracted train_v1 directory')
    source = root / 'prompts.jsonl'
    positives, negatives, seen = [], [], set()
    for lineno, line in enumerate(source.read_text(encoding='utf-8').splitlines(), 1):
        row = json.loads(line)
        image_id, caption = row['id'], row['prompts']
        if not isinstance(image_id, str) or not image_id.isdigit() or image_id in seen:
            raise ValueError(f'invalid or duplicate id on line {lineno}')
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f'invalid caption on line {lineno}')
        path = root / 'images' / f'{image_id}.png'
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(image_id)
        item = dict(id=image_id, caption=caption.strip(), path=str(path))
        (negatives if caption.strip().lower() == 'hard_negative' else positives).append(item)
    if len(positives) < 6:
        raise ValueError('at least six captioned train pairs required')
    return positives, negatives, hashlib.sha256(source.read_bytes()).hexdigest()

def build_vocab(rows, initial=None):
    vocab = dict(initial or {'<pad>': 0, '<unk>': 1})
    if vocab.get('<pad>') != 0 or vocab.get('<unk>') != 1 or sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError('vocabulary IDs must be contiguous, pad=0, unk=1')
    counts = Counter(t for row in rows for t in tokenize(row['caption']))
    for token, _ in sorted(counts.items(), key=lambda p: (-p[1], p[0])):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab

class Pairs(Dataset):
    def __init__(self, rows, vocab):
        self.rows, self.tokens = rows, encode_text_batch([r['caption'] for r in rows], vocab, CONFIG['max_length'])
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        with Image.open(self.rows[index]['path']) as im:
            image = image_to_tensor(im)
        return image, self.tokens[index]

def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)

@torch.inference_mode()
def validate(net, rows, distractors, vocab, device, batch_size):
    net.eval()
    candidates = rows + distractors
    image_features, text_features = [], []
    for start in range(0, len(candidates), batch_size):
        images = []
        for item in candidates[start:start+batch_size]:
            with Image.open(item['path']) as im:
                images.append(image_to_tensor(im))
        image_features.append(net.encode_images(torch.stack(images).to(device)).float())
    features = torch.cat(image_features)
    for start in range(0, len(rows), batch_size):
        tokens = encode_text_batch([r['caption'] for r in rows[start:start+batch_size]], vocab, net.max_length)
        text_features.append(net.encode_tokens(tokens.to(device)).float())
    scores = torch.cat(text_features) @ features.T
    top = scores.topk(min(5, len(candidates)), dim=1).indices
    hits = top.eq(torch.arange(len(rows), device=device)[:, None])
    r1 = hits[:, 0].float().mean().item()
    r5 = hits.any(dim=1).float().mean().item()
    discounts = 1 / torch.log2(torch.arange(top.shape[1], device=device).float()+2)
    ndcg5 = (hits.float() * discounts).sum(dim=1).mean().item()
    return dict(recall_at_1=r1, recall_at_5=r5, ndcg_at_5=ndcg5,
                local_score=(r1+ndcg5)/2, queries=len(rows), candidates=len(candidates))

def cpu_state(net):
    return {k: v.detach().cpu().contiguous().clone() for k, v in net.state_dict().items()}

def package(out):
    with ZipFile(out / 'submission.zip', 'w', compression=ZIP_STORED) as archive:
        for name in sorted(REQUIRED):
            archive.write(out / name, arcname=name)
    with ZipFile(out / 'submission.zip') as archive:
        if set(archive.namelist()) != REQUIRED or archive.testzip() is not None:
            raise RuntimeError('ZIP integrity/structure validation failed')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=.1)
    parser.add_argument('--negative-images-per-batch', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--init-dir', help='trusted old submission unpacked by notebook; optional warm start')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-cpu', action='store_true', help='small local verification only')
    parser.add_argument('--no-imagenet', action='store_true', help='offline verification or explicit scratch training')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or not 0 < args.val_fraction < .5 or args.negative_images_per_batch < 0:
        parser.error('invalid training settings')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not args.allow_cpu:
        raise RuntimeError('No GPU. Select a Colab T4 GPU runtime before training.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not args.resume:
        raise ValueError('output directory is not empty; choose a new run directory or --resume')
    positives, negatives, data_hash = read_records(args.data_root)
    order = list(positives)
    random.Random(args.seed).shuffle(order)
    val_count = max(1, int(len(order)*args.val_fraction))
    val_rows, train_rows = order[:val_count], order[val_count:]
    if len(train_rows) < 2:
        raise ValueError('not enough training pairs')
    initial_vocab = None
    if args.init_dir:
        initial = Path(args.init_dir)
        initial_config = json.loads((initial/'config.json').read_text(encoding='utf-8'))
        if any(initial_config[k] != CONFIG[k] for k in ('embedding_dim','text_hidden','max_length')):
            raise ValueError('initial model configuration differs from this CNN+GRU')
        initial_vocab = json.loads((initial/'vocab.json').read_text(encoding='utf-8'))
    vocab = build_vocab(train_rows, initial_vocab)
    net = DualEncoder(len(vocab), **{k: CONFIG[k] for k in ('embedding_dim','text_hidden','max_length')})
    if args.init_dir:
        state = load_file(str(Path(args.init_dir)/'weights.safetensors'), device='cpu')
        expanded = net.word_embedding.weight.detach().clone()
        expanded[:len(initial_vocab)] = state['word_embedding.weight']
        state['word_embedding.weight'] = expanded
        net.load_state_dict(state, strict=True)
    elif not args.no_imagenet and not args.resume:
        state = torch.hub.load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet18-f37072fd.pth',
            map_location='cpu', check_hash=True, weights_only=True)
        state.pop('fc.weight'); state.pop('fc.bias')
        net.visual.load_state_dict(state, strict=True)
    net.to(device)
    param_count = sum(p.numel() for p in net.parameters())
    if param_count > 500_000_000:
        raise ValueError('parameter count exceeds 0.5B')
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    generator = torch.Generator().manual_seed(args.seed)
    batch_size = min(args.batch_size, len(train_rows))
    # Avoid single-item batches (BatchNorm at tiny spatial sizes / degenerate contrastive target).
    drop_last = len(train_rows) % batch_size == 1
    loader = DataLoader(Pairs(train_rows, vocab), batch_size=batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=amp, drop_last=drop_last,
                        generator=generator, worker_init_fn=seed_worker)
    metadata = dict(dataset='train_v1', drive_file_id=TRAIN_FILE_ID, prompts_sha256=data_hash,
                    positive_pairs=len(positives), hard_negative_images=len(negatives),
                    train_pairs=len(train_rows), val_pairs=len(val_rows),
                    params=param_count, device=str(device), arguments=vars(args),
                    initialization='previous_submission' if args.init_dir else ('random' if args.no_imagenet else 'ImageNet_ResNet18'),
                    versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
                    gpu=torch.cuda.get_device_name(0) if amp else None,
                    validation_note='held-out train only; not official test score',
                    image_preprocessing='160x160 RGB resize, ImageNet normalization; no flips/crops')
    split = dict(train_ids=[r['id'] for r in train_rows], validation_ids=[r['id'] for r in val_rows],
                 hard_negative_ids=[r['id'] for r in negatives])
    start_epoch, best, history = 0, -1.0, []
    checkpoint = out/'last_checkpoint.pt'
    if args.resume:
        # Only load checkpoints created by this script, never third-party pickle files.
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if ckpt['data_hash'] != data_hash or ckpt['split'] != split or ckpt['vocab'] != vocab:
            raise ValueError('resume dataset, split, or vocabulary changed')
        settings = {k: v for k, v in vars(args).items() if k not in ('epochs','resume','workers')}
        if ckpt['resume_settings'] != settings:
            raise ValueError('resume training settings changed')
        net.load_state_dict(ckpt['network'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        generator.set_state(ckpt['generator_state'])
        torch.set_rng_state(ckpt['torch_rng'])
        random.setstate(ckpt['python_rng'])
        if amp:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        start_epoch, best, history = ckpt['epoch'], ckpt['best'], ckpt['history']
    for name, value in [('config.json',CONFIG),('vocab.json',vocab),('split.json',split),('training_metadata.json',metadata)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'model.py').write_bytes(Path(__file__).with_name('model.py').read_bytes())
    print(json.dumps(metadata,indent=2),flush=True)
    for epoch in range(start_epoch, args.epochs):
        begun = time.monotonic(); net.train(); total_loss, total = 0., 0
        for step, (images,tokens) in enumerate(loader,1):
            images, tokens = images.to(device, non_blocking=amp), tokens.to(device, non_blocking=amp)
            count = len(images)
            # Optional negatives have images only: NEVER encode "hard_negative" as a caption.
            chosen = random.sample(negatives,min(args.negative_images_per_batch,len(negatives)))
            if chosen:
                extra = []
                for row in chosen:
                    with Image.open(row['path']) as im:
                        extra.append(image_to_tensor(im))
                images = torch.cat([images,torch.stack(extra).to(device)])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16) if amp else nullcontext():
                image_features, text_features = net.encode_images(images), net.encode_tokens(tokens)
                logits = net.logit_scale.exp().clamp(max=100).float() * (text_features.float() @ image_features.float().T)
                labels = torch.arange(count, device=device)
                loss = (nn.functional.cross_entropy(logits,labels) + nn.functional.cross_entropy(logits[:,:count].T,labels))/2
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite loss; no invalid checkpoint saved')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            scaler.step(optimizer); scaler.update()
            with torch.no_grad(): net.logit_scale.clamp_(math.log(.01),math.log(100))
            total_loss += loss.item()*count; total += count
            if step == 1 or step % 10 == 0 or step == len(loader):
                elapsed = time.monotonic()-begun
                print(f'epoch {epoch+1}/{args.epochs} batch {step}/{len(loader)} loss={loss.item():.4f} elapsed={elapsed:.1f}s',flush=True)
        metrics = validate(net,val_rows,negatives,vocab,device,batch_size)
        elapsed = time.monotonic()-begun
        result = dict(epoch=epoch+1, train_loss=total_loss/total, seconds=elapsed, **metrics)
        history.append(result)
        if metrics['local_score'] > best:
            best = metrics['local_score']
            save_file(cpu_state(net), str(out/'weights.safetensors'))
        checkpoint_tmp = checkpoint.with_suffix('.tmp')
        torch.save(dict(epoch=epoch+1, network=cpu_state(net), optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(), best=best, history=history, data_hash=data_hash,
                        split=split, vocab=vocab,
                        resume_settings={k:v for k,v in vars(args).items() if k not in ('epochs','resume','workers')},
                        generator_state=generator.get_state(), torch_rng=torch.get_rng_state(),
                        python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if amp else []),checkpoint_tmp)
        checkpoint_tmp.replace(checkpoint)
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps(result),flush=True)
        print(f'Estimated remaining based on last epoch: {elapsed*(args.epochs-epoch-1)/60:.1f} minutes',flush=True)
        # Keep a usable best-model ZIP after every completed epoch.
        package(out)
    # Safetensors import and API calls are verified separately by verify_submission.py.
    package(out)
    print(f'SAVED: {out / "submission.zip"}',flush=True)

if __name__ == '__main__':
    main()
