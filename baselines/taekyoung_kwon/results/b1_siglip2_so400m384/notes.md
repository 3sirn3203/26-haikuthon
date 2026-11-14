# taekyoung_kwon/b1_siglip2_so400m384

## Observations

- Zero-shot SigLIP2 SO400M/16@384 (WebLI), cosine similarity over the full candidate pool; no fine-tuning or train hard negatives.
- Test-full evaluation used 8,000 queries and 15,000 candidates, including 7,000 hard negatives. Score: 0.42618 (R@1 0.31450, NDCG@5 0.53786, R@5 0.74088, R@10 0.90825).
- The 1,000-item public split is provisional and derived from hard-negative pairing; its score is 0.46770. The remaining 7,000-item private split scored 0.42025.
- Top-1 errors are concentrated among bundle siblings: 44.79% sibling, 13.60% sibling hard negative, and 7.09% own hard negative; ground truth is top-1 for 31.45%.
- Uncached run on an RTX 4060 Laptop GPU (8 GB): 584.4 s total, 5.72 GB peak VRAM. Timing is hardware-specific.
