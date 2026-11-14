"""HAIKUTHON participant model: pure PyTorch ResNet18 + GRU dual encoder.

Server dependencies: torch, numpy, Pillow, safetensors (no torchvision/transformers).
The evaluator calls predict(model, prompts: list[str], candidates: list[dict])
and expects a list of five image IDs per prompt.
"""
from __future__ import annotations

import json
import time
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file


# Fixed for both training and evaluation. Inference never uses online downloads.
IMAGE_SIZE = 160
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
TOP_K = 5


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+|[^\w\s]", text.lower())


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18(nn.Module):
    """State dict names are compatible with the official torchvision ResNet18 weights."""

    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return torch.flatten(self.avgpool(x), 1)


class DualEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 256,
                 text_hidden: int = 192, max_length: int = 80):
        super().__init__()
        self.visual = ResNet18()
        self.image_proj = nn.Linear(512, embedding_dim)
        self.word_embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.gru = nn.GRU(128, text_hidden, batch_first=True, bidirectional=True)
        self.text_proj = nn.Linear(text_hidden * 2, embedding_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.max_length = max_length

    def encode_images(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_proj(self.visual(x)), dim=-1)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # Packing stops padding from contaminating the GRU's final state.
        lengths = tokens.ne(0).sum(dim=1).clamp(min=1).cpu()
        embedded = self.word_embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        output = torch.cat((h[-2], h[-1]), dim=-1)
        return F.normalize(self.text_proj(output), dim=-1)


def encode_text_batch(prompts: list[str], vocab: dict[str, int], max_length: int) -> torch.Tensor:
    tokens = torch.zeros((len(prompts), max_length), dtype=torch.long)
    for i, prompt in enumerate(prompts):
        ids = [vocab.get(t, 1) for t in tokenize(prompt)[:max_length]]
        if not ids:
            ids = [1]
        tokens[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return tokens


class RetrievalModel:
    """Runtime-only wrapper: holds network and image cache across calls."""

    def __init__(self, network: DualEncoder, vocab: dict[str, int], device: torch.device,
                 image_batch_size: int = 96, text_batch_size: int = 16):
        self.network = network.eval()
        self.vocab = vocab
        self.device = device
        self.image_batch_size = image_batch_size
        self.text_batch_size = text_batch_size
        self.workers = 4
        self.amp = device.type == "cuda"
        self.cache_builds = 0
        self.calls = 0
        self._candidate_signature: tuple[tuple[str, str], ...] | None = None
        self._image_features: torch.Tensor | None = None
        self._image_ids: list[str] | None = None


def load_model(model_dir: str, device: str = "cuda:0") -> RetrievalModel:
    """Required evaluator entry point. Never downloads weights or installs packages."""
    started = time.perf_counter()
    log("load_start", device=device, torch=torch.__version__)
    folder = Path(model_dir)
    with (folder / "config.json").open(encoding="utf-8") as fp:
        config = json.load(fp)
    with (folder / "vocab.json").open(encoding="utf-8") as fp:
        vocab = json.load(fp)
    if not isinstance(vocab, dict) or vocab.get("<pad>") != 0 or vocab.get("<unk>") != 1:
        raise ValueError("invalid vocabulary")
    network = DualEncoder(
        vocab_size=len(vocab),
        embedding_dim=int(config["embedding_dim"]),
        text_hidden=int(config["text_hidden"]),
        max_length=int(config["max_length"]),
    )
    network.load_state_dict(load_file(str(folder / "weights.safetensors"), device="cpu"), strict=True)
    torch_device = torch.device(device)
    network.to(torch_device).eval()
    result = RetrievalModel(
        network, vocab, torch_device,
        image_batch_size=int(config.get("image_batch_size", 96)),
        text_batch_size=int(config.get("text_batch_size", 32)),
    )

    result.workers = max(1, min(16, int(config.get("preprocess_workers", 4))))
    result.amp = result.device.type == "cuda" and bool(config.get("inference_amp", True))
    if min(result.image_batch_size, result.text_batch_size) < 1:
        raise ValueError("batch sizes must be positive")
    log("load_done", seconds=time.perf_counter()-started, amp=result.amp, workers=result.workers)
    return result


def log(event, **values):
    print(json.dumps(dict(event=event, unix_time=time.time(), **values)), flush=True)


def _read_image(item):
    with Image.open(item["image_path"]) as image:
        return image_to_tensor(image)


def _amp(model):
    return torch.autocast("cuda", dtype=torch.float16) if model.amp else nullcontext()


def _sync(model):
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)


@torch.inference_mode()
def _prepare_image_cache(model: RetrievalModel, candidates: list[dict[str, str]]) -> None:
    # Cache key includes paths, so accidental candidate-pool changes are detected.
    signature = tuple((c["image_id"], c["image_path"]) for c in candidates)
    if model._candidate_signature == signature and model._image_features is not None:
        return
    if len(signature) < TOP_K or len(set(c[0] for c in signature)) != len(signature):
        raise ValueError("candidate pool is invalid or has fewer than five images")
    started = time.perf_counter()
    log("cache_start", images=len(candidates), batch=model.image_batch_size)
    # Bounded lookahead: decode next batch while GPU encodes current batch.
    def decode(chunk):
        return torch.stack(list(pool.map(_read_image, chunk)))
    features = []
    with ThreadPoolExecutor(max_workers=model.workers) as pool, ThreadPoolExecutor(max_workers=1) as ahead:
        future = ahead.submit(decode, candidates[:model.image_batch_size])
        for start in range(0, len(candidates), model.image_batch_size):
            images = future.result()
            end = start + model.image_batch_size
            if end < len(candidates):
                future = ahead.submit(decode, candidates[end:end+model.image_batch_size])
            if model.device.type == "cuda":
                images = images.pin_memory()
            images = images.to(model.device, non_blocking=True)
            with _amp(model):
                output = model.network.encode_images(images)
            features.append(F.normalize(output.float(), dim=-1))
            if start == 0 or end >= len(candidates) or (start // model.image_batch_size) % 20 == 0:
                log("cache_progress", images=min(end,len(candidates)))
    model._image_features = torch.cat(features, dim=0)
    model._image_ids = [c["image_id"] for c in candidates]
    model._candidate_signature = signature
    model.cache_builds += 1
    _sync(model)
    log("cache_done", seconds=time.perf_counter()-started, images=len(candidates))


@torch.inference_mode()
def _predict_batch(model: RetrievalModel, prompts: list[str],
            candidates: list[dict[str, str]]) -> list[list[str]]:
    """Required batched evaluator entry point: one top-5 image ID list per prompt.

    Server normally supplies 32 prompts per call. Internal micro-batch sizes are
    independent and configurable. Never compare/rerank results between prompts.
    """
    if not isinstance(prompts, list) or any(not isinstance(x, str) for x in prompts):
        raise TypeError("prompts must be list[str]")
    if not prompts:
        return []
    started = time.perf_counter()
    _prepare_image_cache(model, candidates)
    assert model._image_features is not None and model._image_ids is not None
    results: list[list[str]] = []
    for start in range(0, len(prompts), model.text_batch_size):
        chunk = prompts[start:start + model.text_batch_size]
        tokens = encode_text_batch(chunk, model.vocab, model.network.max_length)
        if model.device.type == "cuda":
            tokens = tokens.pin_memory()
        tokens = tokens.to(model.device, non_blocking=True)
        text_features = model.network.encode_tokens(tokens)
        scores = text_features @ model._image_features.T
        indices = torch.topk(scores, k=TOP_K, dim=1).indices.cpu().tolist()
        results.extend([[model._image_ids[j] for j in row] for row in indices])
    model.calls += 1
    if model.calls == 1 or model.calls % 25 == 0:
        log("predict_done", call=model.calls, prompts=len(prompts), seconds=time.perf_counter()-started, cache_builds=model.cache_builds)
    return results


@torch.inference_mode()
def predict(model, prompts, candidates, top_k=5):
    """Support string-batch and legacy query-dictionary evaluator contracts."""
    if top_k != 5:
        raise ValueError("This submission supports top_k=5")
    if not isinstance(prompts, list):
        raise TypeError("prompts must be a list")
    if prompts and isinstance(prompts[0], dict):
        if any(not isinstance(q, dict) or not isinstance(q.get("prompt_id"), str)
               or not isinstance(q.get("prompt"), str) for q in prompts):
            raise TypeError("queries require string prompt_id and prompt")
        if len({q["prompt_id"] for q in prompts}) != len(prompts):
            raise ValueError("duplicate prompt_id")
        rows = _predict_batch(model, [q["prompt"] for q in prompts], candidates)
        return {q["prompt_id"]: row for q, row in zip(prompts, rows)}
    return _predict_batch(model, prompts, candidates)
