"""Stage-1 dual encoders. ``build(cfg, device)`` returns an object with
encode_images(paths, prof) / encode_texts(texts, prof) / similarity(t, i) / params_b / name."""
from .openclip import OpenCLIPRetriever, RandomRetriever, cache_key


def build(cfg: dict, device):
    kind = cfg.get('type', 'openclip')
    if kind == 'random':
        return RandomRetriever(**cfg)
    if kind == 'openclip':
        return OpenCLIPRetriever(device=device, **{k: v for k, v in cfg.items() if k != 'type'})
    raise ValueError(f'unknown retriever type {kind!r}')
