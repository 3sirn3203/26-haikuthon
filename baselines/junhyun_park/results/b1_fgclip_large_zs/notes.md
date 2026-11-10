# b1_fgclip_large_zs

FG-CLIP large(약 0.43B, ViT-L/14-336) zero-shot. base와 같은 포장·같은 서버 평가 경로(Modal L4).

## 결과
- test_full 0.4234 (R@1 0.3121, NDCG@5 0.5348). Public 0.4530 / Private 0.4192.
- base(0.4155) 대비 +0.008로 작은 향상이다. 파라미터는 약 3배인데 점수 차는 작다. chan_lee zero-shot 비교: PE-Core-L 0.4419, DFN5B 0.4308, SigLIP2-so400m 0.4267, CLIP-L 0.2799.

## 시간 (Modal L4, 캐시 없음)
- 서버 측정 기준 304.9s, 함수 전체 wall 362.9s. 피크 VRAM 4.61GB. 검색 0.2s.
- 세부 구간(이미지/텍스트 인코딩)은 기록하지 못했다. 첫 실행이라 Volume 콜드 읽기 영향이 섞였을 가능성이 크다(base는 콜드 250s / 웜 81s). 웜 재측정은 하지 않았다.

## 한계
- 77토큰 제한으로 긴 프롬프트 뒷부분이 잘린다. `--long`은 large에서 시험하지 않았다.
- Public/Private은 GUIDE.md의 잠정 분할이다.
