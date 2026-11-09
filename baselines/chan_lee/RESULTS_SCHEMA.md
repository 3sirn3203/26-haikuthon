# 결과 파일 표준 스키마

운영진 11명의 결과를 한 번에 모아 분석하려고 만든 형식입니다. 어떤 코드로 실험하든 **결과 파일만은 이 형식**으로 남겨 주세요. 모아 보는 명령은 `python baselines/chan_lee/scripts/aggregate.py --all-authors`이고, `baselines/leaderboard_all.csv`가 생성됩니다.

## 폴더

```
baselines/<본인폴더>/results/
├── leaderboard.csv          # (선택) 본인 실험 요약표
└── <실험이름>/              # exp_id에서 "<author>/"를 뺀 부분
    ├── metrics.json         # 필수: 아래 스키마
    ├── config.yaml          # 필수: 실행에 쓴 설정 사본 (재현용)
    └── notes.md             # 필수: 관찰, 실패 원인, 오류 예시 (10개 이내)
```

- 실험 이름은 `<b번호|l번호>_<모델>_<변형>` 형식입니다. 예: `b1_siglip2_so400m384`, `b3_siglip2__qwen3vl8b_k20`, `l1_hungarian_siglip2`.
  - `b`: honest 실험
  - `l`: loophole 실험
- 개발용으로 일부만 돌린 실험은 이름 뒤에 `__limit<N>`을 붙여 전체 실험과 구분합니다.
- **커밋하지 않는 것**: 제출 CSV, 문항별 예측·랭킹, 임베딩, 가중치. test 정답 매핑이 그대로 담기거나 용량이 크기 때문입니다.

## metrics.json

| 키 | 타입 | 필수 | 설명 |
|---|---|:-:|---|
| `exp_id` | str | ✓ | `<author>/<실험이름>` |
| `author` | str | ✓ | 폴더 이름. 예: `chan_lee` |
| `category` | `"honest"` \| `"loophole"` | ✓ | 규정 허점을 노린 실험은 반드시 `loophole` |
| `method` | str | ✓ | 한 줄 설명 |
| `stage1_model` | str | ✓ | 1단계 모델 id. 예: `ViT-SO400M-16-SigLIP2-384:webli`, 랜덤이면 `random` |
| `stage2_model` | str \| null | ✓ | 재정렬 모델 id. 없으면 null |
| `top_k` | int \| null | ✓ | 재정렬 후보 수 |
| `uses_train_finetune` | bool | ✓ | train으로 학습했는가 |
| `uses_train_hard_negatives` | bool | ✓ | train HN 이미지 500장을 썼는가 |
| `violates_cross_query_rule` | bool | ✓ | 한 문항의 예측이 다른 문항에 영향을 주는가 |
| `eval_split` | `"test_full"` \| `"test_public"` \| `"test_private"` \| `"val"` | ✓ | 평가한 문항 집합 |
| `scores` | `{recall_at_1, ndcg_at_5, score}` | ✓ | `eval_split` 전체 평균. `score = (R@1 + NDCG@5) / 2` |
| `scores_by_split` | `{public: {...}, private: {...}}` | ✓ (test) | `scores`와 같은 키 구성 |
| `timing` | object | ✓ | 아래 표 |
| `peak_vram_gb` | float | ✓ | CUDA는 `torch.cuda.max_memory_allocated`. MPS는 샘플링한 근사치 |
| `params_b` | float | ✓ | 추론에 쓰는 전체 파라미터 수(십억 단위) |
| `artifact_size_gb` | float | ✓ | 제출물(가중치) 크기. 측정하지 않았으면 fp16 기준 추정치(params × 2) |
| `git_commit` | str | ✓ | 실행 시점 커밋. 미커밋 변경이 있으면 `-dirty` |
| `date` | str | ✓ | ISO 8601 |
| `n_queries`, `pool_size`, `n_hn_in_pool` | int | 권장 | 평가 조건 |
| `scores_by_type` | `{easy, bundle}` | 권장 | stage1 기준 문항 / 8개 유사 묶음 문항 |
| `public_split_provisional` | bool | 권장 | 공식 Public id 목록 대신 잠정 규칙을 썼는가 |
| `diagnostics` | object | 권장 | 예: `top1_breakdown`(오답 top-1이 자기 HN / 묶음 형제 / 기타 중 무엇인가) |

`scores` 안에는 `n`, `recall_at_5`, `recall_at_10`, `median_rank`를 더 넣어도 됩니다. 키를 추가하는 것은 자유이지만, 위 필수 키의 **이름과 의미는 바꾸지 마세요**.

### timing

| 키 | 설명 |
|---|---|
| `device` | `cuda:RTX4090`, `mps:M4Max`처럼 `<백엔드>:<장치명>` |
| `image_embeddings_cached` | true이면 이미지 인코딩 시간이 측정되지 않은 것. **시간 비교에는 false인 실행만 쓴다** |
| `model_load_sec` | 모델 로딩(다운로드 제외) |
| `image_encode_sec` | 후보 이미지 전체 인코딩 wall time(I/O 포함). 캐시를 썼으면 null |
| `image_io_wait_sec` | 그중 디코딩·I/O를 기다린 시간 |
| `text_encode_sec` | 프롬프트 인코딩 |
| `stage1_search_sec` | 유사도 계산과 정렬 |
| `rerank_sec` | 2단계 재정렬 |
| `total_sec` | 전체 wall time |

**MPS나 CPU에서 잰 시간은 참고용입니다.** 규정(추론 시간 제한) 판단에는 24GB CUDA GPU에서 `--no-cache`로 잰 값만 씁니다.

## 지표 정의

- 정답 순위 r(1부터 시작)에 대해 `R@1 = [r = 1]`, `NDCG@5 = 1/log2(r+1)`(r ≤ 5일 때, 아니면 0), `score = (R@1 + NDCG@5)/2`입니다. 전체 점수는 문항 평균입니다.
- 동점은 비관적으로 처리합니다. 즉 정답과 점수가 같은 오답은 정답보다 위로 칩니다. 상수 점수를 내는 방법이 좋은 점수를 받지 못하게 하기 위해서입니다.
- 참조 구현은 `chan_lee/src/haiku_bl/metrics.py`이고, 테스트는 `chan_lee/tests/test_metrics.py`입니다.

## 평가 조건 (chan_lee 구현 기준, 잠정)

- **test_full**: 프롬프트 8,000개. 후보 풀은 원본 8,000장 + Private 문항의 HN 7,000장 = 15,000장입니다.
- **Public/Private**: 공식 id 목록이 아직 없습니다. 잠정 규칙은 stage1 '기준' 800문항 + 묶음 문항 중 seed로 뽑은 200문항 = Public 1,000이고, 이때는 `public_split_provisional: true`로 표시합니다. 공식 목록이 생기면 `HAIKU_PUBLIC_IDS`로 지정합니다.
- **val**: train 4,000문항에서 stage1 묶음 단위로 떼어 낸 500문항입니다. 후보 풀은 그 500장 + 해당 문항의 train HN입니다. train 묶음이 학습과 평가에 걸치지 않게 나눴습니다.
- 같은 조건으로 비교하려면 `chan_lee/src/haiku_bl/data.py`의 `build_eval_set`을 그대로 쓰는 것을 권장합니다.
