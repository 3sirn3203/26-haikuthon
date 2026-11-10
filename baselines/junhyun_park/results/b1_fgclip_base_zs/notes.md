# b1_fgclip_base_zs

FG-CLIP base(0.15B) zero-shot. 서버 계약(`load_model`/`predict`)으로 포장한 zip을 서버 `evaluation.runner.evaluate`로 Modal L4에서 실행했다.

## 결과
- test_full 0.4155 (R@1 0.3078, NDCG@5 0.5232). Public 0.4534 / Private 0.4101.
- chan_lee zero-shot(M4 Max, test_full) 비교: PE-Core-L 0.4419, DFN5B 0.4308, SigLIP2-so400m 0.4267, CLIP-L 0.2799. 0.15B인 FG-CLIP base가 1.1B SigLIP2보다 0.011 낮고 CLIP-L보다 0.136 높다. 크기 대비 효율은 좋지만 상위권은 아니다.

## 시간 (Modal L4, 캐시 없음)
- 이미지 15,000장 인코딩 241.6s / 텍스트 8,000개 3.1s / 검색 0.4s / predict 전체 245.1s. 피크 VRAM 1.0GB.
- 전체의 98%가 이미지 처리였다. 다만 같은 이미지 처리 코드를 쓴 b1_fgclip_base_long_zs에서는 이미지 인코딩이 76.4s였으므로, 이 241.6s에는 첫 실행의 Volume 콜드 읽기 등 환경 영향이 섞였을 가능성이 크다같은 zip을 다시 돌리자 이미지 인코딩 73.4s, 서버 측정 기준 총 81.4s(setup 34.5s 별도)로 줄었다. 따라서 첫 실행의 241.6s는 Volume 콜드 읽기 등 첫 실행 효과로 보이며, 시간은 웜 상태(약 81s)와 콜드 상태(약 250s)를 구분해 기록해야 한다.
- 서버 측정 기준(load_model 직전 ~ CSV) 250.5s, 그 앞의 unzip·데이터셋 로드·의존성 설치(시간 제한 제외) 37.0s, 함수 전체 wall 288.6s. 모델 로드는 약 5.4s(CSV 쓰기 포함 근사).
- `image_io_wait_sec`는 측정하지 못했다.

## 한계
- v1은 최대 77토큰이라 긴 프롬프트(최대 71단어, 약 90토큰)의 뒷부분(시점·조명)이 잘린다. `--long`(248토큰, walk_short_pos=False)은 아직 시험하지 않았다.
- Public/Private은 GUIDE.md의 잠정 분할(test 7000-7999)이며 공식 목록이 아니다.
