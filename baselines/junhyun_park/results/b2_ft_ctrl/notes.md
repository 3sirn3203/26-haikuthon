# b2_ft_ctrl

b2_ft_hn2의 대조군. 같은 데이터·설정(시드 0, 3ep, bs 64, lr 1e-5, 마지막 4블록)에서 text hard negative만 뺐다.

## 결과
- test_full 0.5546 (R@1 0.438, NDCG@5 0.6711). Public 0.6084 / Private 0.5469.
- zero-shot(0.4155) 대비 +0.139. 점수 향상의 대부분은 train 도메인 학습 효과다.
- chan_lee 파인튜닝 SigLIP2-so400m(test_full)의 무작위 배치 (a) 0.619, 묶음 배치 (a') 0.680보다 낮다. 모델 크기(0.15B vs 1.14B), epoch(3 vs 5), loss(InfoNCE vs sigmoid)가 다르다.
- hn2(0.5399)보다 0.0147 높다. 이 실험의 규칙 기반 text hard negative는 도움이 되지 않았고 오히려 낮았다. Colab val 500에서는 반대(ctrl 0.9240 < hn2 0.9291)였으므로, 이 val은 test의 방향을 맞히지 못했다.

## 시간 (Modal L4, 캐시 없음)
- 서버 측정 80.2s (이미지 72.4s / 텍스트 3.2s / 검색 0.4s). 피크 VRAM 1.0GB. setup 33.6s.
- 제출물 0.31GB는 bf16 저장(`train_fgclip_hn.py`의 `model.to(torch.bfloat16).save_pretrained`) 기준이다. zero-shot zip(0.6GB)은 HF 배포 가중치(fp32)를 그대로 써서 크기가 2배이며, 모델 크기 차이가 아니다.

## 한계
- 시드 1개. 차이 0.0147이 시드 변동보다 큰지는 확인하지 않았다.
- val은 무작위 분할(번들 단위 아님)이라 같은 묶음의 형제 문항이 train과 val에 섞인다. val과 test의 방향이 어긋난 원인일 가능성이 있다.
- Public/Private은 GUIDE.md의 잠정 분할이다.
