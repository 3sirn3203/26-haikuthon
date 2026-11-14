#!/usr/bin/env python
"""Download experiment models in priority order (HF Hub). Prints one OK/FAIL line per model.

    HF_XET_HIGH_PERFORMANCE=1 python scripts/download_models.py --tier 1 2

The per-connection speed to the HF CDN from this machine is ~0.3 MB/s, so xet's high-concurrency
mode (~8 MB/s) is needed. Tiers keep the disk (~60 GB free) and the schedule under control.
"""
import argparse
import time

TIERS = {
    1: [('openclip', 'ViT-SO400M-16-SigLIP2-384', 'webli'), ('openclip', 'ViT-H-14-378-quickgelu', 'dfn5b'),
        ('openclip', 'PE-Core-L-14-336', 'meta'), ('openclip', 'ViT-L-14-336-quickgelu', 'openai'),
        ('hf', 'Qwen/Qwen3-VL-Embedding-2B')],
    2: [('hf', 'Qwen/Qwen3-VL-Reranker-2B'), ('hf', 'Qwen/Qwen3.5-2B'), ('hf', 'Qwen/Qwen3.5-4B')],
    3: [('openclip', 'ViT-H-14', 'laion2b_s32b_b79k'), ('openclip', 'EVA02-L-14-336', 'merged2b_s6b_b61k'),
        ('openclip', 'ViT-H-14-worldwide-378', 'metaclip2_worldwide')],
    4: [('hf', 'Qwen/Qwen3.5-9B'), ('hf', 'Qwen/Qwen3-VL-Reranker-8B'),
        ('openclip', 'ViT-bigG-14', 'laion2b_s39b_b160k'), ('hf', 'Qwen/Qwen3-VL-Embedding-8B')],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', type=int, nargs='+', default=[1])
    a = ap.parse_args()
    for t in a.tier:
        for spec in TIERS[t]:
            t0 = time.time()
            try:
                if spec[0] == 'openclip':
                    import open_clip
                    m, _, _ = open_clip.create_model_and_transforms(spec[1], pretrained=spec[2], device='cpu')
                    del m
                else:
                    from huggingface_hub import snapshot_download
                    snapshot_download(spec[1], allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model',
                                                               '*.py', '*.jinja', 'tokenizer*', '*.md'])
                print('OK', t, *spec[1:], f'{time.time() - t0:.0f}s', flush=True)
            except Exception as e:  # keep going; report at the end
                print('FAIL', t, *spec[1:], repr(e)[:300], flush=True)


if __name__ == '__main__':
    main()
