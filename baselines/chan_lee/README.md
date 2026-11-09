# chan_lee: HAIKUTHON 베이스라인 (참가자 입장에서 풀어 보기)

목적: 대회 규정(추론 시간 제한, 모델 크기 제한, train HN 제공 여부, 난이도, 규정 허점)을 정할 근거를 측정합니다. 11/11 회의용 결론은 [REPORT.md](REPORT.md)에, 결과 형식은 [RESULTS_SCHEMA.md](RESULTS_SCHEMA.md)에 있습니다.

## 재현

```bash
cd baselines/chan_lee
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt
export HAIKU_DATA_ROOT=../../data          # 기본값도 레포 루트의 data/
.venv/bin/python -m pytest -q -p no:cacheprovider tests

# 실험 하나 (결과: results/<이름>/metrics.json)
.venv/bin/python scripts/run_exp.py --config configs/exp/b0_random_test.yaml
.venv/bin/python scripts/run_exp.py --config configs/exp/<exp>.yaml --no-cache   # 시간 측정용(캐시 미사용)
.venv/bin/python scripts/aggregate.py      # results/leaderboard.csv

# 채점 서버 형식의 추론 (정답 없이, 처음부터 끝까지 시간 측정)
.venv/bin/python scripts/make_blind_test.py                     # 익명 id로 test 패키지 생성 (cache/)
.venv/bin/python scripts/predict.py --config configs/exp/<exp>.yaml \
    --images cache/blind/test/images --prompts cache/blind/test/prompts.jsonl \
    --out cache/sub.csv --timing-out cache/timing.json
.venv/bin/python scripts/make_blind_test.py --score cache/sub.csv
```

장치(`cuda`/`mps`/`cpu`)는 자동으로 고릅니다. MPS에서 잰 시간은 참고용입니다.

## 구조

| 경로 | 내용 |
|---|---|
| `src/haiku_bl/data.py` | 프롬프트·이미지 로더, stage1 묶음 복원, val / Public / Private 분할 |
| `src/haiku_bl/metrics.py` | R@1, NDCG@5, score |
| `src/haiku_bl/submission.py` | 랭킹을 제출 CSV(`id,image_ids`)로 변환, 검증, 채점 |
| `src/haiku_bl/profiling.py` | 장치 선택, 단계별 시간, 피크 VRAM, `--vram-budget-gb` |
| `src/haiku_bl/pipeline.py` | 설정 하나로 1단계 검색 → (재정렬) → (후처리) 실행 |
| `src/haiku_bl/retrievers/` | open_clip 기반 dual encoder(CLIP, OpenCLIP, SigLIP2, EVA, DFN, PE) |
| `src/haiku_bl/rerankers/` | 2단계 재정렬기 |
| `src/haiku_bl/loopholes.py` | 규정 위반 실험(Hungarian 등). 측정 전용 |
| `configs/paths.yaml` | 데이터 경로(환경변수로 override) |
| `configs/exp/` | 실험별 설정 |

## 결과 요약

(실험이 진행되면 채웁니다. 원본 수치: `results/leaderboard.csv`)
