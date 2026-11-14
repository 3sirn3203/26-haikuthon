# b1_fgclip_base_zs

FG-CLIP base(0.15B) zero-shot. 서버 계약(`load_model`/`predict`)으로 포장한 zip을 서버 `evaluation.runner.evaluate`로 Modal L4에서 실행했다.

## 결과
- test_full 0.4155 (R@1 0.3078, NDCG@5 0.5232). Public 0.4534 / Private 0.4101.
- chan_lee zero-shot(M4 Max, test_full) 비교: PE-Core-L 0.4419, DFN5B 0.4308, SigLIP2-so400m 0.4267, CLIP-L 0.2799. 0.15B인 FG-CLIP base가 1.1B SigLIP2보다 0.011 낮고 CLIP-L보다 0.136 높다. 크기 대비 효율은 좋지만 상위권은 아니다.

## 시간 (Modal L4, 캐시 없음)
- 첫 실행(콜드): 서버 측정 기준(load_model 직전 ~ CSV) 250.5s. 이미지 15,000장 인코딩 241.6s / 텍스트 8,000개 3.1s / 검색 0.4s / predict 전체 245.1s. 피크 VRAM 1.0GB. 모델 로드는 약 5.4s(CSV 쓰기 포함 근사).
- 같은 zip 재실행(웜): 서버 측정 기준 81.4s. 이미지 인코딩 73.4s / 텍스트 3.1s / 검색 0.4s.
- 이미지 처리 코드는 두 실행이 같으므로, 첫 실행의 241.6s는 Volume 콜드 읽기 등 첫 실행 효과로 보인다. 시간은 웜(약 81s)과 콜드(약 250s)를 구분해 기록한다. 어느 쪽이든 전체 시간의 대부분(웜 약 90%, 콜드 약 96%)은 이미지 인코딩이다.
- 서버 측정 구간 앞의 unzip·데이터셋 로드·의존성 설치(시간 제한 제외)는 37.0s(웜 34.5s), 첫 실행의 함수 전체 wall은 288.6s였다.
- `image_io_wait_sec`는 측정하지 못했다.

## 한계
- v1은 최대 77토큰이라 긴 프롬프트(최대 71단어, 약 90토큰)의 뒷부분(시점·조명)이 잘린다. 긴 캡션 모드(248토큰, walk_short_pos=False)는 b1_fgclip_base_long_zs에서 시험했고, 점수 차는 +0.0015로 무시할 수준이었다.
- Public/Private은 GUIDE.md의 잠정 분할(test 7000-7999)이며 공식 목록이 아니다.
