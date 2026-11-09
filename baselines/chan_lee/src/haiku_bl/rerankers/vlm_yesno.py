"""VQAScore-style reranker: score = logit(Yes) - logit(No) for
"Does this image match the description ...? Answer yes or no." on a chat VLM (Qwen-VL family).

Each query is scored against its own K candidates only (no cross-query information).
``max_pixels`` caps the vision tokens per image and is the main speed/accuracy knob.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..profiling import param_count_b

QUESTION = ('Does this image accurately match the following description, including every '
            'object count, color, material, pattern, pose and spatial relation?\n'
            'Description: "{prompt}"\nAnswer with Yes or No.')


class VLMYesNoReranker:
    def __init__(self, model: str, *, device: torch.device, dtype: str = 'bf16',
                 batch_size: int = 8, max_pixels: int = 448 * 448, load_in_8bit: bool = False,
                 attn_implementation: str | None = None, **_):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.device = device
        self.batch_size = batch_size
        torch_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[dtype]
        kw = dict(dtype=torch_dtype)
        if attn_implementation:
            kw['attn_implementation'] = attn_implementation
        if load_in_8bit:                      # CUDA only (bitsandbytes)
            from transformers import BitsAndBytesConfig
            kw['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
            kw['device_map'] = {'': device.index or 0}
        self.model = AutoModelForImageTextToText.from_pretrained(model, **kw).eval()
        if not load_in_8bit:
            self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(model, min_pixels=64 * 64, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = 'left'
        tok = self.processor.tokenizer
        self.yes = [tok.encode(w, add_special_tokens=False)[0] for w in ('Yes', 'yes', ' Yes')]
        self.no = [tok.encode(w, add_special_tokens=False)[0] for w in ('No', 'no', ' No')]

    @property
    def params_b(self) -> float:
        return param_count_b(self.model)

    def _chat(self, prompt):
        msgs = [{'role': 'user', 'content': [{'type': 'image'},
                                             {'type': 'text', 'text': QUESTION.format(prompt=prompt)}]}]
        return self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                  enable_thinking=False)

    @torch.inference_mode()
    def score(self, prompt: str, image_paths, prof=None) -> np.ndarray:
        text = self._chat(prompt)
        out = []
        for s in range(0, len(image_paths), self.batch_size):
            ims = []
            for p in image_paths[s:s + self.batch_size]:
                with Image.open(p) as im:
                    ims.append(im.convert('RGB'))
            inputs = self.processor(text=[text] * len(ims), images=ims, return_tensors='pt',
                                    padding=True).to(self.device)
            logits = self.model(**inputs, logits_to_keep=1).logits[:, -1].float()
            lse = lambda ids: torch.logsumexp(logits[:, ids], dim=-1)  # noqa: E731
            out.append((lse(self.yes) - lse(self.no)).cpu())
            if prof:
                prof.sample()
        return torch.cat(out).numpy()
