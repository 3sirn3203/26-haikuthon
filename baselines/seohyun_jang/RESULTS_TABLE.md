| 실험 | 평가 | 문항 | score | R@1 | NDCG@5 | Public | Private | 총 시간(s, L4) | peak VRAM(GB) |
|---|---|---|---|---|---|---|---|---|---|
| b1_siglip2_so400m384_l4 | test_full | 8000 | 0.4263 | 0.3145 | 0.5381 | 0.4661 | 0.4206 | 366 | 5.6 |
| b2_siglip2_bundle_l4 | test_full | 8000 | 0.6771 | 0.5736 | 0.7805 | 0.7438 | 0.6675 | 345 | 5.6 |
| b2_siglip2_bundle_l4__val | val | 500 | 0.7348 | 0.6420 | 0.8276 |  |  | 41 | 5.6 |
| b3_siglip2_bundle_l4__qwen35_4b_full_k10 | test_subset | 1183 | 0.7991 | 0.7329 | 0.8653 | 0.8573 | 0.7906 | 12669 (추정) | 9.9 |
| b3_siglip2_bundle_l4__qwen35_4b_full_k3 | test_subset | 1183 | 0.7933 | 0.7337 | 0.8530 | 0.8497 | 0.7851 | 4077 (추정) | 9.9 |
| b3_siglip2_bundle_l4__qwen35_4b_full_k5 | test_subset | 1183 | 0.8092 | 0.7498 | 0.8686 | 0.8617 | 0.8015 | 6532 (추정) | 9.9 |
| b5_fusion_pairwise_l4 | test_subset | 1183 | 0.8292 |  |  | 0.8769 | 0.8222 |  | 8.8 |
