"""Stage-2 rerankers: rescore the stage-1 top-K candidates of each query independently.
``build(cfg, device)`` returns an object with score(prompt, image_paths, prof) -> np.ndarray."""


def build(cfg: dict, device):
    kind = cfg['type']
    if kind == 'vlm_yesno':
        from .vlm_yesno import VLMYesNoReranker
        return VLMYesNoReranker(device=device, **{k: v for k, v in cfg.items() if k != 'type'})
    raise ValueError(f'unknown reranker type {kind!r}')
