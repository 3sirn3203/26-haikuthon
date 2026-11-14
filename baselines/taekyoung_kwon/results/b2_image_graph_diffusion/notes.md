# taekyoung_kwon/b2_image_graph_diffusion

- Query-wise image-neighbor graph diffusion with SigLIP2 SO400M/16@384 (WebLI); no fine-tuning or train hard negatives.
- Full test: 8,000 queries, 15,000 candidates (7,000 hard negatives). Score 0.41199 (R@1 0.30588, NDCG@5 0.51810).
- Graph settings: k=10 nearest candidate images, temperature 0.07, alpha 0.15, two propagation steps. Other queries and labels are not used.
- Provisional public/private scores are 0.47780 / 0.40259; the public split is inferred from hard-negative pairing, not official IDs.
- RTX 4060 Laptop GPU, uncached embeddings: 568.3 s total, 22.2 s graph postprocess, 5.72 GB peak VRAM. Overall score is 0.01419 below the measured B1 zero-shot baseline.
