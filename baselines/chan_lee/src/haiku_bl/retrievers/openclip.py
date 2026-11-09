"""Dual encoders through open_clip (OpenAI CLIP, LAION OpenCLIP, SigLIP/SigLIP2, EVA, DFN...)."""
from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..profiling import Profiler, param_count_b


class ImageFiles(Dataset):
    def __init__(self, paths, transform):
        self.paths = [str(p) for p in paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert('RGB'))


def _autocast(device, precision):
    if precision == 'fp32' or device.type == 'cpu':
        return nullcontext()
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


class OpenCLIPRetriever:
    """``model``: open_clip architecture name or ``hf-hub:org/repo``; ``pretrained``: tag or None."""

    def __init__(self, model: str, pretrained: str | None = None, *, device: torch.device,
                 precision: str = 'fp16', batch_size: int = 64, text_batch_size: int = 256,
                 num_workers: int = 8, checkpoint: str | None = None, **_):
        import open_clip
        self.name = model if pretrained is None else f'{model}:{pretrained}'
        self.device, self.precision = device, precision
        self.batch_size, self.text_batch_size, self.num_workers = batch_size, text_batch_size, num_workers
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained, device=device)
        if checkpoint:                       # fine-tuned weights (B2)
            state = torch.load(checkpoint, map_location='cpu')
            self.model.load_state_dict(state.get('model', state), strict=False)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)

    @property
    def params_b(self) -> float:
        return param_count_b(self.model)

    @torch.inference_mode()
    def encode_images(self, paths, prof: Profiler | None = None) -> np.ndarray:
        loader = DataLoader(ImageFiles(paths, self.preprocess), batch_size=self.batch_size,
                            num_workers=self.num_workers, pin_memory=self.device.type == 'cuda',
                            persistent_workers=False, prefetch_factor=4 if self.num_workers else None)
        out, it = [], iter(loader)
        while True:
            t = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            if prof:                         # time the main loop waits on decode/IO
                prof.add('image_io_wait', time.perf_counter() - t)
            ctx = prof.stage('image_forward') if prof else nullcontext()
            with ctx, _autocast(self.device, self.precision):
                f = self.model.encode_image(batch.to(self.device, non_blocking=True))
                out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    @torch.inference_mode()
    def encode_texts(self, texts, prof: Profiler | None = None) -> np.ndarray:
        out = []
        for s in range(0, len(texts), self.text_batch_size):
            tokens = self.tokenizer(list(texts[s:s + self.text_batch_size])).to(self.device)
            ctx = prof.stage('text_forward') if prof else nullcontext()
            with ctx, _autocast(self.device, self.precision):
                f = self.model.encode_text(tokens)
                out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    def similarity(self, text_emb: np.ndarray, image_emb: np.ndarray) -> np.ndarray:
        # Ranking only needs the dot product; logit scale/bias are monotone.
        return text_emb @ image_emb.T


class RandomRetriever:
    """B0: uniform random scores. Used to check the metric implementation."""

    name = 'random'
    params_b = 0.0

    def __init__(self, seed: int = 0, **_):
        self.rng = np.random.default_rng(seed)

    def encode_images(self, paths, prof=None):
        return np.zeros((len(paths), 1), dtype=np.float32)

    def encode_texts(self, texts, prof=None):
        return np.zeros((len(texts), 1), dtype=np.float32)

    def similarity(self, text_emb, image_emb):
        return self.rng.random((len(text_emb), len(image_emb)), dtype=np.float32)


def cache_key(name: str, ckpt: str | None = None) -> str:
    key = name.replace('/', '_').replace(':', '__')
    if ckpt:
        key += '__' + Path(ckpt).stem
    return key
