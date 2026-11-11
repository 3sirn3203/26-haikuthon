# b4_rerank_qwen2b_lora

1단계 FG-CLIP(b3_ft_bundle_bval)이 뽑은 top-K를 Qwen3.5-2B VLM의 Yes/No 점수(logit Yes − logit No)로 다시 정렬한다. chan_lee와 seohyun_jang은 재정렬기를 zero-shot으로만 썼다. 이 실험에서는 train 데이터로 재정렬기를 **LoRA 파인튜닝**했다.

**현재는 val 결과만 있다. 실제 test 데이터셋(8,000문항, 후보 15,000장)으로는 추후에 검증할 예정이다.** 그래서 `metrics.json`은 아직 만들지 않았다.

## 설정
- 코드: `finetune/colab_rerank.ipynb`, `finetune/rerank/` (Colab L4)
- 1단계: b3_ft_bundle_bval (FG-CLIP base, 묶음 배치 FT). val 1단계 단독 0.7386, R@5 0.972, R@10 1.0
- 재정렬기: Qwen/Qwen3.5-2B, max_pixels 448², 질문 문구는 chan_lee `vlm_yesno.py`와 같음, enable_thinking=False
- LoRA: 언어 모델 Linear 층만(비전 인코더·lm_head 고정), r 16, α 32, dropout 0.05. 학습 파라미터 16.8M / 2.23B (0.75%)
- 학습: train 3,500문항(val 묶음 제외), 1 epoch, 438 step, 실효 배치 8문항, lr 1e-4 cosine. 학습에 약 40분(L4)
  - 문항마다 [정답 1 + 오답 3] 묶음에 listwise softmax CE. 오답은 같은 묶음의 형제 이미지와 1단계가 헷갈린 top-10 오답에서 뽑음
  - train HN 이미지는 쓰지 않음 (R2에서 쓰려 했으나 중단)
- 평가: chan_lee와 같은 묶음 val 500문항, 후보 풀은 val 원본 500장 + val HN 102장. 1단계 top-K만 재정렬하고 나머지는 1단계 순서를 유지. 동점은 비관적으로 처리.

## 결과 (val 500)

| 조건 | K=5 | K=10 | R@1 (K=5) |
|---|---|---|---|
| 1단계 단독 | 0.7386 | | 0.648 |
| R0 zero-shot 재정렬 | 0.8171 | 0.8072 | 0.752 |
| **R1 LoRA FT 재정렬** | **0.9219** | **0.9386** | 0.900 |

- zero-shot 재정렬로 +0.079, 파인튜닝으로 +0.105가 더 붙는다(K=5).
- R0는 K=10이 K=5보다 낮다. 후보가 늘면 오답에 속는 경우가 늘어서다(seohyun_jang의 관찰과 같다). R1은 반대로 K=10이 더 높다. 파인튜닝 뒤에는 늘어난 오답에 덜 속는다.
- 재정렬 속도: 쌍당 약 0.067초(L4, bf16, causal_conv1d 커널 없음).

## test 검증 계획 (추후)
- 제출 zip: `submission/rerank/model.py` + `scripts/modal_build_rerank.py`(Modal 안에서 빌드). FG-CLIP은 표준 CLIPModel로 로드해 transformers 5.x 하나로 두 단계를 돌린다. 로컬 검사에서 원본 FG-CLIP과 임베딩 코사인이 1.000000으로 같았다. LoRA는 Qwen 가중치에 병합하고 MTP 층은 뺀다(zip 약 3.8GB로 예상).
- 실행: `modal_run_fgclip.py --budget 0`으로 시간 예산 없이 전체를 재정렬해서 잰다.
- **서버 시간 제한 문제:** 서버 GPU 제한은 15분이다. K=5로 8,000문항을 재정렬하면 4만 쌍 × 0.067초 ≈ 45분이 걸려 제한을 넘는다. zip 기본값은 `budget_sec=660`으로, 시간이 다 되면 남은 문항은 1단계 순서로 낸다. 15분 안에 끝나는 설정(K 축소, 일부 문항만 재정렬, 이미지 해상도 축소 등)은 따로 정해야 한다.

## 한계
- val 500문항, 시드 1개다.
- val 후보 풀은 HN 비율이 17%로 test(47%)보다 낮다. 1단계 기준으로 val→test에서 0.066 떨어졌으므로, 재정렬 점수도 test에서 낮아질 수 있다.
- R2(train HN 이미지 추가)는 시간 관계상 140/438 step에서 중단해 결과가 없다.
