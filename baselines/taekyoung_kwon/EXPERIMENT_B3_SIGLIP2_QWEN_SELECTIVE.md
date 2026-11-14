# SigLIP2 번들 파인튜닝 + 저마진 쿼리 재순위화

## 결과 요약

SigLIP2를 train 번들로 파인튜닝한 뒤, 상위 후보의 점수가 비슷한 쿼리만 Qwen3.5-2B로 다시 정렬했습니다. 전체 test 점수는 **0.68005**로, 같은 평가 실행의 stage 1 점수 **0.66228**보다 **0.01778** 올랐습니다.

| 평가 | 쿼리 수 | R@1 | NDCG@5 | 점수 |
|---|---:|---:|---:|---:|
| SigLIP2 2블록 파인튜닝 | 8,000 | 0.55538 | 0.76918 | 0.66228 |
| 저마진 8%만 Qwen top-5 재순위화 | 8,000 | 0.57950 | 0.78061 | **0.68005** |

점수는 `(R@1 + NDCG@5) / 2`입니다. 검증 500개에서도 8% 게이트 선택 후 0.70351에서 0.71589로 올랐습니다. test 정답을 사용해 게이트를 조정하지 않았습니다.

## 방법

1. train의 4,000개 쿼리를 번들 단위로 나눠 500개를 검증용으로 보류하고, 3,500개로 학습했습니다. 번들 전체가 학습과 검증에 나뉘어 들어가지 않도록 했습니다.
2. SigLIP2 SO400M/16@384의 마지막 두 transformer 블록과 투영층을 5 epoch 파인튜닝했습니다. 한 배치에는 같은 번들의 쿼리를 모으고, train hard-negative 이미지도 정답이 아닌 후보로 넣었습니다. 학습은 2,190 step, batch 8, learning rate `1e-5`였습니다.
3. 각 쿼리에서 SigLIP2 상위 두 후보의 점수 차를 계산했습니다. 차가 작을수록 모델이 헷갈리는 쿼리로 봤습니다.
4. 홀드아웃 검증 쿼리 500개에서 재순위화 비율 0%, 2%, 4%, 6%, 8%를 비교했습니다. **가장 낮은 점수 차 8%**가 가장 높은 검증 점수를 내서 이 비율을 고정했습니다.
5. test에서도 점수 차가 가장 작은 쿼리 8%(640개)만 골라 상위 5개 이미지와 프롬프트를 Qwen3.5-2B에 넣었습니다. `logit(Yes) - logit(No)` 순으로 상위 5개를 다시 정렬하고, 나머지 후보 및 재순위화하지 않은 쿼리는 SigLIP2 순서를 유지했습니다.

각 쿼리는 독립적으로 처리했습니다. 다른 쿼리의 결과나 test 정답은 사용하지 않았습니다.

## 데이터와 모델 규모

| 항목 | 설정 |
|---|---|
| 학습 쿼리 | 3,500개 |
| 학습에 연결된 train hard negatives | 398장 |
| 검증 | 500개 쿼리, 후보 602장(원본 500 + HN 102) |
| 전체 test | 8,000개 쿼리, 후보 15,000장(원본 8,000 + HN 7,000) |
| SigLIP2 | SO400M/16@384, WebLI, 약 1.136B 파라미터 |
| 재순위 모델 | Qwen3.5-2B, BF16, 최대 이미지 크기 200,704 pixel, 배치 2 |
| 재순위 대상 | 쿼리 640개(8%), 쿼리당 top-5, 총 3,200개 이미지-문장 쌍 |
| 실제 로컬 weight 파일 | SigLIP2 4.544GB + Qwen 4.548GB + 학습 체크포인트 0.310GB = **9.403GB** |

검증 및 test의 Public/Private 구분은 공식 ID 목록이 아니라 hard-negative 짝짓기로 만든 잠정 분할입니다. 전체 점수와 비교에는 영향을 주지 않지만, 두 분할의 수치는 참고용입니다.

## 시간과 메모리

측정 장비는 **RTX 4060 Laptop GPU, 8GB VRAM**입니다. Qwen 모델은 로컬 캐시에 받은 뒤 측정했습니다.

| 작업 | 시간 |
|---|---:|
| SigLIP2 파인튜닝 | 816.1초 (13분 36초) |
| SigLIP2 전체 test 추론 | 561.4초 (9분 21초) |
| Qwen 로딩 + 선택적 재순위화 | 533.4초 (8분 53초) |
| 전체 test 추론 합계 | **1,094.8초 (18분 15초)** |
| 학습 peak VRAM | 5.92GB |
| 추론 peak VRAM | 5.72GB |

학습 시간은 추론 시간에 포함하지 않았습니다. 이 장비에서 전체 추론은 15분보다 길었습니다. 제출 서버의 L4에서는 속도가 달라질 수 있으므로, 15분 예산 충족 여부는 L4에서 다시 재야 합니다.

## 파일과 재현

- 파인튜닝 설정: [b3_siglip2_bundle_hn_2blk.yaml](configs/exp/b3_siglip2_bundle_hn_2blk.yaml)
- 재순위 설정: [b4_siglip2_bundle_qwen35_2b_margin.yaml](configs/exp/b4_siglip2_bundle_qwen35_2b_margin.yaml)
- 재순위 실행 코드: [run_selective_rerank.py](scripts/run_selective_rerank.py)
- 파인튜닝 실행 wrapper: [finetune_bundle.py](scripts/finetune_bundle.py)
- 전체 test 지표: [metrics.json](results/b4_siglip2_bundle_qwen35_2b_margin/metrics.json)
- stage 1 지표: [metrics.json](results/b3_siglip2_bundle_hn_2blk/metrics.json)

저장소 루트 PowerShell에서 실행합니다. 재현할 때는 Qwen3.5-2B와 SigLIP2 weight가 HF cache에 있어야 합니다.

```powershell
$env:HF_HOME = Join-Path (Get-Location) 'baselines/taekyoung_kwon/.cache/huggingface'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:HF_HUB_OFFLINE = '1'
$env:HAIKU_DATA_ROOT = 'data'
$env:HAIKU_CACHE_DIR = 'baselines/taekyoung_kwon/cache'

& 'baselines/taekyoung_kwon/.venv/Scripts/python.exe' -B `
  'baselines/taekyoung_kwon/scripts/pair_hn.py' --split train
& 'baselines/taekyoung_kwon/.venv/Scripts/python.exe' -B `
  'baselines/taekyoung_kwon/scripts/finetune_bundle.py' `
  --config 'baselines/taekyoung_kwon/configs/exp/b3_siglip2_bundle_hn_2blk.yaml'
& 'baselines/taekyoung_kwon/.venv/Scripts/python.exe' -B `
  'baselines/taekyoung_kwon/scripts/run_selective_rerank.py' `
  --config 'baselines/taekyoung_kwon/configs/exp/b4_siglip2_bundle_qwen35_2b_margin.yaml' `
  --paths 'baselines/taekyoung_kwon/configs/paths.yaml'
```
