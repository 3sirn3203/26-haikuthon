# 박준현 (`junhyun_park`) 베이스라인

- 이 폴더 안에서만 작업하고 푸시해 주세요. 커밋할 때는 `git add baselines/junhyun_park`처럼 본인 폴더 경로만 지정합니다(`git add -A` / `git add .` 금지).
- 데이터, 모델 가중치, 임베딩 캐시는 커밋하지 않습니다.
- 결과 파일은 [RESULTS_SCHEMA.md](../RESULTS_SCHEMA.md) 형식을 따라 `results/<exp_id>/metrics.json`에 둡니다.

## 진행 중인 실험 (FG-CLIP, 서버 형식 end-to-end)

목표: (1) 제출 서버 계약(`model.py` + `*.safetensors` zip)이 실제로 돌아가는지, Modal L4에서의 시간·VRAM 측정, (2) 공개 train 프롬프트 기반 text hard negative 파인튜닝 효과.

| 폴더 | 내용 |
|---|---|
| `submission/fgclip/model.py` | 서버 계약 `load_model` / `predict` (FG-CLIP v1·v2 어댑터) |
| `scripts/build_fgclip_submission.py` | HF 스냅샷 또는 파인튜닝 결과 폴더 → `submission.zip` + 서버 규칙 검증 |
| `scripts/modal_run_fgclip.py` | 웹사이트 없이 서버 `evaluation` 코드를 Modal L4에서 직접 실행 |
| `scripts/score_csv.py` | 결과 CSV 채점 (scores / scores_by_split) |
| `finetune/hn_rules.py`, `finetune/train_fgclip_hn.py` | 규칙 기반 text HN, Colab 파인튜닝 (`--hn 0` 대조군) |
| `finetune/make_val_ids.py` | 번들 단위 val 분할 (chan_lee `val_split`) |

재현 순서는 각 스크립트 상단 docstring 참고. 결과는 `results/<실험이름>/`에 기록한다.
