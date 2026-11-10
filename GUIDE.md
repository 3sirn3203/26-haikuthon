# 운영진 베이스라인 실험 가이드

HAIKUTHON 출제 레포에서 운영진이 각자 베이스라인 실험을 하고 결과를 올리는 방법을 정리한 문서입니다. 11명이 같은 `main` 브랜치에 푸시하므로, **아래 규칙만 지키면 서로 충돌이 나지 않습니다.**

- 이미 끝난 실험 결과와 결론: [baselines/chan_lee/REPORT.md](baselines/chan_lee/REPORT.md)
- 결과 파일 형식: [baselines/RESULTS_SCHEMA.md](baselines/RESULTS_SCHEMA.md)

---

## 1. 레포 구조

```
haikuthon/
├── stage1_components/         # 장면 구성 요소 생성 (읽기 전용)
├── stage2_prompt_generation/  # 프롬프트 생성·검수 (읽기 전용)
├── stage3_image_generation/   # 이미지 생성 (읽기 전용)
├── outputs_stage1/            # 장면 구성 요소 jsonl (읽기 전용)
├── outputs_stage2/            # 프롬프트 jsonl. test_prompts_v2.jsonl이 실제 test 프롬프트 (읽기 전용)
├── data/                      # 이미지 데이터셋. 각자 받아서 둠, git 제외
├── baselines/
│   ├── README.md
│   ├── RESULTS_SCHEMA.md      # 결과 파일 표준 (필독)
│   ├── chan_lee/              # 이찬: 공용으로 쓸 수 있는 실험 코드 포함
│   ├── taekyoung_kwon/  donghwi_kwon/  doyoung_kim/  seongeun_kim/  woojin_kim/
│   └── seoyeon_park/  junhyun_park/  soohyuk_yang/  seohyun_jang/  ichan_jung/
└── GUIDE.md                   # 이 문서
```

- **본인 폴더(`baselines/<이름_성>/`) 밖의 파일은 수정하지 않습니다.** `stage*`, `outputs_stage*`, 루트 설정 파일은 읽기만 합니다.
- 다른 사람 폴더의 코드를 쓰고 싶으면 수정하지 말고 import하거나 본인 폴더로 복사합니다.

---

## 2. 데이터 받기

1. 운영진 드라이브 폴더에서 `train_v1.tar`(3.8GB), `test_v1.tar`(13GB)를 받습니다. 폴더 접근 권한이 필요합니다.
   https://drive.google.com/drive/folders/1tdLtjggz1gCMjzs4dPzHrfz7YBlGR4lO
2. 레포 루트의 `data/`에 풀어서 아래 구조를 만듭니다.

   ```
   data/
   ├── .gitignore            # 내용은 "*" 한 줄 (data/ 전체를 git에서 제외)
   ├── train_v1/  prompts.jsonl, images/0000.png … 4499.png
   └── test_v1/   prompts.jsonl, images/00000-07999/*.png, images/08000-14999/*.png
   ```

   `data/.gitignore`가 없으면 직접 만드세요(`printf '*\n' > data/.gitignore`). 루트 `.gitignore`는 건드리지 않습니다.

**데이터를 볼 때 알아 둘 점**
- `prompts.jsonl`의 키는 `{id, prompts}`입니다. HN 행은 `"prompts": "hard_negative"`로 표시되어 있습니다.
- 원본 이미지의 파일 id는 프롬프트 id와 같습니다(`00000.png` = test 프롬프트 `0000`의 정답).
- HN `N+k`는 원본 `k`의 변형입니다(test는 N=8000, train은 N=4000).
- **이 구조는 운영진 내부용 패키지라서 정답이 그대로 보입니다.** 참가자 조건을 흉내 내려면 id를 쓰지 말고 이미지와 텍스트만으로 예측해야 합니다. 정답은 평가할 때만 사용하세요.
- Public = test `7000–7999`(HN이 풀에 없는 1,000문항), Private = 나머지입니다. 공식 목록이 정해지면 이 문서를 갱신합니다.
- 자세한 데이터 분석과 누수 목록: [REPORT.md §6](baselines/chan_lee/REPORT.md)

---

## 3. 실험 방법

어떤 코드, 어떤 프레임워크를 써도 됩니다. **결과 파일만 표준 형식(§4)으로 남기면 됩니다.**

### 3.1 (선택) 이찬의 실험 코드 재사용

`baselines/chan_lee/`에는 데이터 로더, 지표, 평가 분할, 1단계 검색, 재정렬, 파인튜닝, 시간·메모리 측정 코드가 있습니다. 설정 파일 하나로 실험 하나가 돌아가고, 표준 형식의 결과가 자동으로 나옵니다.

```bash
cd baselines/chan_lee
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt
.venv/bin/python scripts/pair_hn.py --split train && .venv/bin/python scripts/pair_hn.py --split test   # 최초 1회
```

- 캐시(임베딩, HN 짝 정보)는 기본으로 `chan_lee/cache/`에 쌓입니다. 본인 폴더에 두려면 `export HAIKU_CACHE_DIR=baselines/<이름_성>/cache`를 설정하고, `cache/`를 본인 `.gitignore`에 넣습니다. 이 경우 `pair_hn.py`도 이 설정 상태에서 한 번 실행합니다.
- 본인 폴더에서 쓰려면 `sys.path`에 `baselines/chan_lee/src`를 넣고 `haiku_bl`을 import합니다.

  ```python
  import sys; sys.path.insert(0, '../chan_lee/src')
  from haiku_bl import data as D, metrics as M
  es = D.build_eval_set('test')          # 8,000 문항, 15,000장 후보, Public/Private 태그 포함
  ranks = M.ranks_from_scores(scores, es.gt_index)   # scores: (8000, 15000) 유사도 행렬
  print(M.summarize(ranks), M.summarize(ranks, es.masks()['private']))
  ```

- 같은 평가 조건(후보 풀, Public/Private 정의, 지표, 동점 처리)을 쓰면 11명의 결과를 바로 비교할 수 있습니다. **가능하면 `build_eval_set`과 `metrics.py`를 그대로 쓰는 것을 권장합니다.**
- 주요 스크립트

  | 스크립트 | 용도 |
  |---|---|
  | `run_exp.py --config <yaml>` | 1단계 검색 실험 |
  | `run_rerank.py` | 2단계 재정렬 |
  | `finetune.py` | dual encoder 파인튜닝 |
  | `predict.py` | 채점 서버 형식의 추론. 정답 없이, 시간 측정 포함 |
  | `aggregate.py` | 결과 집계 |

  설정 예시는 `chan_lee/configs/exp/`에 있습니다.
- 경로는 `chan_lee/configs/paths.yaml`과 환경변수 `HAIKU_DATA_ROOT`로 바꿀 수 있습니다.

### 3.2 공통 실험 규칙

- **평가는 test 전체(8,000문항, 15,000장)를 기본으로 합니다.** 부분집합으로 잰 경우 이름과 `metrics.json`에 명시합니다(`eval_split`, `n_queries`).
- **문항은 독립적으로 예측합니다.** Hungarian 같은 문항 간 할당이나, 다른 문항의 예측을 참조하는 방법은 규정 위반입니다. 이득을 재는 목적이라면 `category: "loophole"`, `violates_cross_query_rule: true`로 표시해 따로 기록합니다.
- **모델이나 하이퍼파라미터는 test가 아니라 val이나 train으로 고릅니다.** test로 고르면 결과가 낙관적으로 나옵니다.
- **시간을 잴 때는 캐시를 쓰지 않습니다.** 장치(`cuda:RTX4090`, `mps:M4Max` 등)도 함께 기록합니다. 규정 판단에는 24GB CUDA GPU 측정값을 씁니다.
- **피크 메모리도 기록합니다**(CUDA: `torch.cuda.max_memory_allocated`). 24GB 제한 판단의 근거가 됩니다.

---

## 4. 결과 남기기

[baselines/RESULTS_SCHEMA.md](baselines/RESULTS_SCHEMA.md)의 형식을 따릅니다. 요약하면 다음과 같습니다.

```
baselines/<이름_성>/
├── README.md               # 무엇을 했는지, 재현 방법, 핵심 결과 요약
├── .gitignore              # 본인 폴더용 ignore 규칙
└── results/
    └── <실험이름>/          # 예: b1_siglip2_so400m384, b3_ft_siglip2__qwen35_4b_k5
        ├── metrics.json    # 필수: exp_id, category, scores, scores_by_split, timing, peak_vram_gb, ...
        ├── config.yaml     # 필수: 재현용 설정 사본
        └── notes.md        # 필수: 관찰, 실패 원인, 오류 예시(10개 이내)
```

- `exp_id`는 `<이름_성>/<실험이름>` 형식입니다. 실험 이름은 `b`(honest) 또는 `l`(loophole) + 번호로 시작합니다.
- 전체 집계는 `python baselines/chan_lee/scripts/aggregate.py --all-authors`로 하고, 결과는 `baselines/leaderboard_all.csv`에 생성됩니다. 형식이 다르면 집계에서 빠집니다.

---

## 5. 커밋과 푸시 규칙

```bash
git pull --rebase origin main                 # 시작 전 / 푸시 전 항상
git add baselines/<이름_성>                    # 본인 폴더만
git status                                    # 본인 폴더 외 변경이 없는지 확인
git commit -m "<이름_성>: <무엇을 했는지>"
git push origin main
```

- **`git add -A`, `git add .`, `git add baselines/`는 쓰지 않습니다.** 다른 사람이나 다른 폴더의 변경이 섞입니다.
- 커밋 메시지는 `<이름_성>:` 접두어로 시작합니다.
- 푸시가 거절되면 `git pull --rebase origin main` 후 다시 푸시합니다. 본인 폴더만 건드렸다면 충돌이 나지 않습니다.

### 커밋하면 안 되는 것

| 종류 | 예시 | 처리 |
|---|---|---|
| 데이터 | 이미지, tar, 프롬프트 사본 | `data/`에만 둠 |
| 모델 가중치·체크포인트 | `*.pt`, `*.safetensors`, `*.bin` | 본인 폴더 `.gitignore` |
| 임베딩·캐시 | `*.npy`, `*.npz`, `cache/` | 본인 폴더 `.gitignore` |
| 문항별 예측·제출 CSV | 랭킹, `submission*.csv` | 정답 매핑이 담기므로 커밋 금지 |
| 가상환경 | `.venv/` | 본인 폴더 `.gitignore` |

- 루트 `.gitignore`의 `test*/` 규칙 때문에 `tests/` 폴더도 무시됩니다. 테스트 코드를 커밋하려면 본인 `.gitignore`에 `!tests/`를 추가합니다.
- 커밋 대상은 코드, 설정, 작은 결과 파일(json/csv/md/png, 대략 1MB 이하)입니다.
- 참고: [baselines/chan_lee/.gitignore](baselines/chan_lee/.gitignore)

---

## 6. 기밀 유지

- test 프롬프트 원문, 정답 매핑, HN 프롬프트는 **대회 기밀**입니다. 레포 밖(슬랙 공개 채널, 개인 노트, 외부 서비스)에 올리지 않습니다.
- 리포트나 `notes.md`에 쓰는 오류 예시는 **실험당 10개 이내**로 제한합니다.
- 상용 API(GPT, Claude 등)에 test 프롬프트나 이미지를 보내지 않습니다.

---

## 7. 체크리스트

- [ ] 본인 폴더 안에서만 작업했다
- [ ] `results/<실험이름>/`에 `metrics.json`, `config.yaml`, `notes.md`가 있다
- [ ] `metrics.json`에 `category`, `scores_by_split`(public/private), `timing.device`, `peak_vram_gb`가 있다
- [ ] 문항 간 정보를 쓴 실험은 `loophole`로 표시했다
- [ ] 데이터, 가중치, 캐시, 예측 파일이 커밋에 없다(`git status`, `git diff --cached --stat`으로 확인)
- [ ] `git add baselines/<이름_성>`으로만 스테이징했다

문의: 이찬
