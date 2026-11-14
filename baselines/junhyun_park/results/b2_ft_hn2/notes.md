# b2_ft_hn2

FG-CLIP base를 공개 train으로 파인튜닝하고, 프롬프트 텍스트 규칙(방향·개수·색상)으로 만든 hard negative를 프롬프트당 최대 2개 추가했다.

## 결과
- test_full 0.5399 (R@1 0.421, NDCG@5 0.6589). Public 0.597 / Private 0.5318.
- 같은 모델 zero-shot(b1_fgclip_base_zs 0.4155) 대비 +0.124.
- chan_lee 파인튜닝 SigLIP2-so400m(test_full)과 비교하면 무작위 배치 (a) 0.619, 묶음 배치 (a') 0.680보다 낮다. 다만 모델 크기(0.15B vs 1.14B), epoch(3 vs 5), loss(InfoNCE vs sigmoid)가 달라 직접 비교는 어렵다.
- 대조군(b2_ft_ctrl, hard negative 없음)은 0.5546으로 hn2보다 0.0147 높다. 향상은 도메인 학습 효과이고, 이 규칙 기반 text hard negative는 도움이 되지 않았다.

## 시간 (Modal L4, 캐시 없음)
- 서버 측정 103.2s (이미지 95.0s / 텍스트 3.2s / 검색 0.4s). 피크 VRAM 1.0GB. 이 실행의 setup은 37.3s.
- 같은 구조인 b2_ft_ctrl의 이미지 인코딩은 72.4s였다. 95.0s와의 차이는 모델이 아니라 실행 간 환경 변동(Volume 읽기 등)으로 보인다.

## 한계
- val은 무작위 분할(번들 단위 아님)이고 시드 1개다. val 500 점수는 상대 비교용이며 test 점수와 다르다.
- 무작위 분할이라 같은 묶음의 형제 문항이 train과 val에 섞이고, val 후보 풀(500장)에 들어오는 형제는 문항당 평균 1개 미만이다(test는 7개). val(0.93)과 test(0.54)의 큰 차이와, val과 test에서 hn2·ctrl 순위가 뒤집힌 것은 이 때문일 가능성이 있다. 다음 실험은 `finetune/make_val_ids.py`(번들 단위 분할)로 val을 만든다.
- Public/Private은 GUIDE.md의 잠정 분할이다.
