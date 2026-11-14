"""Multimodal cross-encoder rerankers through sentence-transformers (e.g. Qwen/Qwen3-VL-Reranker-2B/8B)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..profiling import param_count_b


class STCrossReranker:
    def __init__(self, model: str, *, device: torch.device, dtype: str = 'bf16', batch_size: int = 8,
                 max_pixels: int | None = None,
                 instruction: str = 'Retrieve the image that exactly matches every object, count, '
                                    'attribute and spatial relation in the description.', **_):
        from sentence_transformers import CrossEncoder
        torch_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[dtype]
        kw = {'model_kwargs': {'dtype': torch_dtype}}
        if max_pixels:
            kw['processor_kwargs'] = {'max_pixels': max_pixels}
        self.model = CrossEncoder(model, device=str(device), **kw)
        self.batch_size, self.instruction = batch_size, instruction

    @property
    def params_b(self):
        return param_count_b(self.model)

    @torch.inference_mode()
    def score(self, prompt, image_paths, prof=None):
        ims = []
        for p in image_paths:
            with Image.open(p) as im:
                ims.append(im.convert('RGB'))
        s = self.model.predict([(prompt, im) for im in ims], batch_size=self.batch_size,
                               prompt=self.instruction, show_progress_bar=False)
        if prof:
            prof.sample()
        return np.asarray(s, dtype=np.float32)
