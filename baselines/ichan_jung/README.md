# 정이찬 (`ichan_jung`) 베이스라인

생성 이미지 선호 모델(PickScore) zero-shot과 실제 제출 서버(L4) 검증.

## 무엇을 했나
| 실험 | 내용 |
|---|---|
| `b1_pickscore_zs` | PickScore v1(CLIP-H/14, 프롬프트-생성이미지 선호로 미세조정)을 그대로 사용한 듀얼 인코더. 문항별 독립 Top-5 |

목적은 두 가지다.
1. **가설 검증**: test가 생성 이미지라면, 생성 이미지로 학습한 모델이 범용 모델(chan_lee b1: SigLIP2, DFN5B-H, PE-Core)보다 나은가?
2. **24GB CUDA 실측 시간**: 기존 b1 시간은 모두 Mac(M4 Max) 측정값이다. 실제 제출 사이트(L4 24GB, CPU 4코어)에서 잰 추론 시간을 규정 판단 참고값으로 남긴다.

## 결과
| exp | score (test_full) | R@1 | NDCG@5 | 추론 시간 (L4) |
|---|---|---|---|---|
| b1_pickscore_zs | 0.331 | 0.234 | 0.427 | 사이트 미보고 |

**가설 기각.** 생성 이미지 선호 모델인 PickScore는 범용 모델(chan_lee b1: PE-Core 0.442, SigLIP2 0.427)보다 약 0.1 낮다. 선호 학습이 검색에 도움이 되지 않았다. 자세한 내용은 [notes.md](results/b1_pickscore_zs/notes.md)에 있다.

## 재현
```sh
pip install torch transformers safetensors huggingface_hub   # 빌드에만 필요
python export_submission.py --model pickscore --out build/pickscore
# → build/pickscore/submission.zip (1.84 GiB)을 제출 사이트에 업로드
python record_site_result.py --exp b1_pickscore_zs --score <사이트 점수> --total-sec <사이트 시간>
```

## 파일
| 파일 | 역할 |
|---|---|
| `submission/model.py` | 제출용 `load_model` / `predict`. 순수 torch CLIP 구현과 BPE 토크나이저. 서버 기본 패키지만 사용 |
| `export_submission.py` | HF 가중치를 fp16 safetensors로 변환하고 ZIP 생성. transformers 원본과 임베딩이 같은지 자동 검증 |
| `record_site_result.py` | 사이트 결과를 `results/<exp>/metrics.json`(RESULTS_SCHEMA)으로 기록. 보고되지 않은 값은 null |

## 검증 내역 (사이트 제출 전, 로컬)
- 원본 transformers와 임베딩이 같음: text cos 1.00000, image cos 1.00000, 토크나이저 결과 같음
- 서버 코드 통과: 업로드 ZIP 검사(`validate_zip`), GPU 워커의 압축 해제·로딩·추론·출력 검증(`prepare_submission` → `load_model` → `run_inference`). 가짜 이미지로 확인

## 한계
- 사이트는 전체 시간만 보고한다. 구간별 시간과 피크 VRAM은 null로 남긴다.
- 사이트 채점의 Public/Private 분할이 chan_lee의 잠정 분할과 같은지 확인이 필요하다.

## 폴더 규칙
- 이 폴더 안에서만 작업하고 푸시해 주세요. 커밋할 때는 `git add baselines/ichan_jung`처럼 본인 폴더 경로만 지정합니다(`git add -A` / `git add .` 금지).
- 데이터, 모델 가중치, 임베딩 캐시는 커밋하지 않습니다.
- 결과 파일은 [RESULTS_SCHEMA.md](../RESULTS_SCHEMA.md) 형식을 따라 `results/<exp_id>/metrics.json`에 둡니다.
