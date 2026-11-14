# 권동휘 CNN + GRU 베이스라인

기존에 제출한 모델 ZIP과 결과 CSV를 정리한 기록입니다. 이번 정리 과정에서 모델을 다시 학습하거나 서버를 변경하지 않았습니다.

## 모델과 결과

| 항목 | 값 |
|---|---|
| 이미지 인코더 | ResNet18, 입력 160×160 RGB |
| 텍스트 인코더 | 128차원 단어 임베딩 + 양방향 GRU, 방향별 hidden 192 |
| 검색 | 256차원 정규화 임베딩, cosine similarity, 문항별 독립 Top5 |
| 파라미터 | 11,871,681개, 약 0.011872B |
| 가중치 | FP32, 47,538,012 bytes, 47.54 MB / 45.34 MiB |
| 어휘 | 737개, 최대 80토큰 |
| 전체 R@1 | 0.0035000000 |
| 전체 NDCG@5 | 0.0110003098 |
| 전체 score | 0.0072501549 |
| 전체 R@5 | 0.0181250000 |
| 평가 범위 | 8,000문항, 후보 15,000장, 그중 HN 7,000장 |
| GPU 평가 시간 | 사용자 관찰 약 240초; GPU 배정 대기 제외 |
| 관찰 장치 | NVIDIA L4 24 GB, 원본 실행 로그로 재검증하지 못함 |
| 피크 VRAM | 미측정 |

FG-CLIP 모델이 아닙니다. `GUIDE.md`는 사전학습 모델 사용을 필수로 요구하지 않습니다. 기존 Colab 학습 코드의 기본값은 ImageNet ResNet18로 이미지 인코더를 초기화하는 것이지만, 이번 ZIP에는 학습 로그가 없어 **실제로 어떤 초기화 옵션으로 학습했는지는 확정할 수 없습니다.**

`model.py`는 제공받은 제출 ZIP의 파일을 그대로 복사했습니다. 추론 중 온라인 다운로드가 없으며, 이미지 전처리 4개 스레드와 다음 배치 준비, 이미지 인코딩 CUDA AMP가 이미 들어 있습니다. 이미지 임베딩은 실행 안에서 한 번 생성해 재사용합니다. 문항 간 할당이나 다른 문항의 예측 결과를 이용하는 처리는 없습니다.

## 파일

- `model.py`: 기존 제출의 추론 코드 원본.
- `train.py`: 이전 Colab 노트북에서 추출한 학습 코드. 실제 실행 이력의 증명은 아닙니다.
- `score_submission.py`: 외부에 둔 ZIP·CSV를 읽고 집계 JSON만 생성합니다. Python 3.11 이상, 추가 라이브러리 불필요.
- `requirements.txt`: 모델 실행에 필요한 패키지 이름입니다. 원본 실행 환경을 고정한 lockfile은 아닙니다.
- `tests/test_score_submission.py`: 실제 데이터가 없는 합성 입력 검증.
- `results/b1_resnet18_bigru/`: 결과, 설정과 한계.

가중치·어휘·데이터·원본 CSV는 포함하지 않았습니다. 로컬에서는 `artifacts/`, `data/`, `private/`처럼 ignore된 경로에 보관합니다. 원본 ZIP, CSV의 SHA-256은 `metrics.json`에 있어 파일을 확인할 수 있습니다.

## 기존 CSV 다시 집계

저장소 루트에서 실행합니다. ZIP과 CSV는 별도로 보관한 실제 파일로 지정합니다.

```bash
python baselines/donghwi_kwon/score_submission.py \
  --csv baselines/donghwi_kwon/artifacts/original_submission.csv \
  --submission-zip baselines/donghwi_kwon/artifacts/original_submission.zip \
  --observed-eval-sec 240 \
  --observed-device 'cuda:NVIDIA L4' \
  --out baselines/donghwi_kwon/artifacts/recomputed_metrics.json

python -m unittest discover -s baselines/donghwi_kwon/tests -v
```

공식 Public 목록이 정해지면 로컬 JSON 문자열 배열 파일을 `--public-ids`에 지정할 수 있습니다. 해당 목록과 문항별 예측은 커밋하지 않습니다. 파일 지정만으로 그 목록의 공식 여부가 검증되지는 않습니다.

## 학습 코드 재실행

기존 CUDA 환경에서 실행합니다. 다음은 기존 코드의 기본 설정을 재실행하는 예시이며, 업로드된 가중치가 이 설정으로 만들어졌다는 증명은 아닙니다.

```bash
python baselines/donghwi_kwon/train.py \
  --data-root data/train_v1 \
  --output-dir baselines/donghwi_kwon/artifacts/retrain_01 \
  --epochs 3 --batch-size 64 --lr 0.0003 --weight-decay 0.0001 \
  --seed 42 --val-fraction 0.1 --negative-images-per-batch 0 --workers 2
```

기본 초기화에는 공식 ImageNet ResNet18 가중치를 내려받는 과정이 있습니다. 처음부터 학습하려면 `--no-imagenet`, 이전 제출에서 시작하려면 `--init-dir`를 사용합니다. 두 옵션에 따라 결과가 달라지므로 실행 기록을 남겨야 합니다.

이 코드의 검증 분할은 train 문항을 무작위로 90:10 분리합니다. 참조 구현의 묶음 단위 500문항 검증 분할과 다르며, 묶음이 학습·검증에 걸칠 수 있습니다. 기본값에서는 HN을 학습 손실에 넣지 않고, train 검증 후보에는 포함합니다. 학습 코드가 생성하는 `config.json`의 text batch 기본값은 16이며, 제공된 제출의 추론 설정은 32입니다. 제공된 추론 설정은 결과 폴더의 `config.yaml`에 따로 보존했습니다.

완전한 규정 판단에는 별도의 24 GB CUDA 실행에서 캐시 없이 시작해 단계별 시간과 `torch.cuda.max_memory_allocated()`를 측정하고 기록을 보완해야 합니다. 현재 자료만으로 그 측정값을 만들 수는 없습니다.
