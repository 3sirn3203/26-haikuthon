# 베이스라인

이 폴더는 권택영의 실험 코드와 결과를 보관합니다. 작업과 커밋은 이 폴더 안에서만 합니다.

## 첫 실험: `b1_siglip2_so400m384`

SigLIP2 SO400M/16@384 zero-shot 이중 인코더로 전체 후보 이미지를 임베딩하고, 프롬프트와의 코사인 유사도로 순위를 매깁니다. 학습, train hard negative, 문항 간 정보 공유는 사용하지 않는 정직한 기준선입니다. 설정은 `configs/exp/b1_siglip2_so400m384.yaml`에 있습니다.

이 레포의 `data/train_v1` 및 `data/test_v1` 이미지 데이터가 준비된 뒤 Linux/NVIDIA CUDA 환경에서 실행합니다. 의존성은 공용 기준선의 고정 버전을 사용합니다.

```bash
uv venv --python 3.12 baselines/taekyoung_kwon/.venv
uv pip install --python baselines/taekyoung_kwon/.venv/bin/python -r baselines/chan_lee/requirements.txt
baselines/taekyoung_kwon/.venv/bin/python baselines/taekyoung_kwon/scripts/run_exp.py \
  --config baselines/taekyoung_kwon/configs/exp/b1_siglip2_so400m384.yaml \
  --paths baselines/taekyoung_kwon/configs/paths.yaml --no-cache
```

결과는 실행을 마친 뒤 `results/b1_siglip2_so400m384/`에 저장합니다. 점수와 측정값은 직접 실행한 결과만 기록하며, 다른 작성자의 결과를 복사하지 않습니다. 산출물 형식은 [RESULTS_SCHEMA.md](../RESULTS_SCHEMA.md)를 따릅니다.

## 두 번째 실험: 이미지 후보 그래프 확산

차트에 없는 후보 이미지 그래프 확산 방식을 평가했습니다. 점수는 기존 zero-shot 기준선보다 낮아, 이 설정에서는 성능 향상이 확인되지 않았습니다. 방법, 데이터 규모, 시간, 결과는 [실험 문서](EXPERIMENT_B2_IMAGE_GRAPH_DIFFUSION.md)에 쉽게 정리했습니다. 설정과 원시 지표는 `configs/exp/b2_image_graph_diffusion.yaml` 및 `results/b2_image_graph_diffusion/`에 있습니다.

## 세 번째 실험: 번들 파인튜닝 + 저마진 재순위화

SigLIP2를 번들 단위로 파인튜닝하고, 검증에서 정한 저마진 쿼리 8%만 Qwen3.5-2B top-5로 재순위화했습니다. 전체 test 점수는 **0.68005**로 stage 1보다 **+0.01778** 올랐습니다. 방법, 데이터·모델 규모, 소요 시간은 [실험 문서](EXPERIMENT_B3_SIGLIP2_QWEN_SELECTIVE.md)에 정리했습니다. 설정과 상세 지표는 `configs/exp/b3_siglip2_bundle_hn_2blk.yaml`, `configs/exp/b4_siglip2_bundle_qwen35_2b_margin.yaml`, `results/`에 있습니다.
