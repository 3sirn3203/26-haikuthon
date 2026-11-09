"""VLM-based embedding models through sentence-transformers (e.g. Qwen/Qwen3-VL-Embedding-2B/8B)."""
from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

from ..profiling import log_progress, param_count_b


class STEmbedRetriever:
    def __init__(self, model: str, *, device: torch.device, dtype: str = 'bf16', batch_size: int = 16,
                 text_batch_size: int = 64, max_pixels: int | None = None,
                 query_prompt: str | None = 'Retrieve images that match the description.', **_):
        from sentence_transformers import SentenceTransformer
        torch_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[dtype]
        kw = {'model_kwargs': {'dtype': torch_dtype}}
        if max_pixels:
            kw['processor_kwargs'] = {'max_pixels': max_pixels}
        self.model = SentenceTransformer(model, device=str(device), **kw)
        self.name, self.device = model, device
        self.batch_size, self.text_batch_size, self.query_prompt = batch_size, text_batch_size, query_prompt

    @property
    def params_b(self):
        return param_count_b(self.model)

    @torch.inference_mode()
    def encode_images(self, paths, prof=None):
        out = []
        for s in range(0, len(paths), self.batch_size):
            t = time.perf_counter()
            ims = []
            for p in paths[s:s + self.batch_size]:
                with Image.open(p) as im:
                    ims.append(im.convert('RGB'))
            if prof:
                prof.add('image_io_wait', time.perf_counter() - t)
            out.append(self.model.encode(ims, batch_size=self.batch_size, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=False))
            if prof:
                prof.sample()
            log_progress(s + len(ims), len(paths))
        return np.concatenate(out).astype(np.float32)

    @torch.inference_mode()
    def encode_texts(self, texts, prof=None):
        kw = {'prompt': self.query_prompt} if self.query_prompt else {}
        return self.model.encode(list(texts), batch_size=self.text_batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False, **kw).astype(np.float32)

    def similarity(self, t, i):
        return t @ i.T
