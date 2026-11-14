"""colab_rerank.ipynb 를 이 폴더의 .py 파일로부터 만든다 (노트북 안의 코드가 레포 파일과 항상 같도록).

  python baselines/junhyun_park/finetune/rerank/make_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "colab_rerank.ipynb"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


def writefile(name):
    return code(f"%%writefile /content/work/{name}\n" + (HERE / name).read_text().rstrip("\n"))


cells = [
    md("""
# 재정렬기(Qwen3.5 VLM) LoRA 파인튜닝 (Colab)

1단계(FG-CLIP)가 뽑은 top-K 후보를 VLM에 "이 그림이 설명과 맞나? Yes/No"로 물어 다시 정렬한다. chan_lee는 이 재정렬기를 **zero-shot**으로만 썼다. 여기서는 train 데이터로 재정렬기를 **LoRA 파인튜닝**하고 같은 val로 비교한다.

| 조건 | 재정렬기 | 학습 때 오답 | 답하는 질문 |
|---|---|---|---|
| **R0** | zero-shot (chan_lee 방식 그대로) | — | 기준선 |
| **R1** | LoRA FT | 같은 묶음 형제 이미지 + 1단계가 헷갈린 오답 | 재정렬기 파인튜닝 효과 |
| **R2** | LoRA FT | R1 + 자기 train HN 이미지(398장, 있으면 1개 고정) | HN 이미지가 재정렬 단계에서는 도움이 되나 (Q3) |

- 학습: 프롬프트마다 [정답 1 + 오답 3] 묶음, 묶음 안 Yes−No 점수에 softmax CE. 비전 인코더 고정, 언어 모델 Linear 층에만 LoRA. 추론은 쌍마다 독립 채점(chan_lee와 동일).
- 평가: chan_lee와 같은 묶음 val 500문항, 후보 풀 = val 원본 500 + val HN 102. 1단계 top-5 / top-10을 재정렬.
- 학습 데이터: train 3,500문항(val 묶음 제외). test는 쓰지 않는다.

**순서**: 1) Drive → 2) 설정 → 3) 설치 → 4) 데이터 → 5) 코드 → 6) 1단계 후보 → 7) R0 → 8) R1 → 9) R2 → 10) 비교 → 11) test 재정렬

- **test만 돌릴 때**: 1~5 실행(SMOKE=False) → 6의 두 번째 셀(함수 정의)은 건너뛰어도 됨 → 11로 바로 간다.

- 런타임: **GPU (A100 권장, L4 가능)**. 결과(1단계 후보, LoRA 어댑터, 점수)는 Drive `RERANK_DIR`에 저장되고, 이미 있으면 건너뛴다. **런타임이 끊기면 1~5를 다시 실행한 뒤 이어서** 돌리면 된다.
- **처음에는 `SMOKE = True`로 끝까지 한 번 돌려** 오류가 없는지 확인한다(몇 분). 확인되면 `False`로 바꾸고 1~5 다음 6부터 다시 실행한다. SMOKE 결과는 별도 폴더에 저장된다.
- 필요한 Drive 파일: `train_v1.tar`, `val_ids.txt`, `train_bundles.json`(레포 `finetune/`에 있음). 1단계 FG-CLIP은 `OUT_DIR/bundle_bval` → `OUT_DIR/ctrl` 순서로 찾고, 없으면 zero-shot `qihoo360/fg-clip-base`를 쓴다.
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')
"""),
    md("## 2. 설정"),
    code("""
import os, glob, json, subprocess

def find_in_drive(name):
    hits = glob.glob(f"/content/drive/MyDrive/**/{name}", recursive=True)
    assert hits, f"MyDrive 에서 {name} 를 못 찾음 -> Drive 에 올리거나 경로를 직접 적으세요"
    return hits[0]

SMOKE = True            # True: 학습 64문항 / 평가 30문항으로 파이프라인만 점검. 확인 후 False

# --- 데이터 ---
TRAIN_TAR = ""          # 비우면 MyDrive 에서 train_v1.tar 자동 탐색
VAL_IDS_PATH = ""       # 비우면 val_ids.txt 자동 탐색
BUNDLES_PATH = ""       # 비우면 train_bundles.json 자동 탐색
OUT_DIR = "/content/drive/MyDrive/haikuthon_runs"
STAGE1 = ""             # 비우면 OUT_DIR/bundle_bval -> OUT_DIR/ctrl -> qihoo360/fg-clip-base 순서

# --- 재정렬기 ---
RERANKER = "Qwen/Qwen3.5-2B"
MAX_PIXELS = 448 * 448  # chan_lee 와 같음 (200,704)
KS = (5, 10)            # 재정렬 후보 수

# --- 학습 (R1/R2 공통) ---
N_NEG = 3               # 프롬프트당 오답 수 -> 묶음 크기 4
PROMPTS_PER_STEP = 2    # 한 번에 GPU 에 올리는 프롬프트 수 (시퀀스 2x4=8). OOM 이면 1
GRAD_ACCUM = 4          # 실효 배치 = 2x4 = 8 프롬프트
EPOCHS = 1
LR = 1e-4
LORA = {"r": 16, "alpha": 32, "dropout": 0.05}
SEED = 0

RERANK_DIR = f"{OUT_DIR}/rerank" + ("_smoke" if SMOKE else "")
MAX_PROMPTS = 64 if SMOKE else None
EVAL_LIMIT = 30 if SMOKE else None
os.makedirs(RERANK_DIR, exist_ok=True)

VAL_IDS_PATH = VAL_IDS_PATH or find_in_drive("val_ids.txt")
BUNDLES_PATH = BUNDLES_PATH or find_in_drive("train_bundles.json")
if not STAGE1:
    STAGE1 = next((d for d in (f"{OUT_DIR}/bundle_bval", f"{OUT_DIR}/ctrl")
                   if os.path.exists(f"{d}/model.safetensors")), "qihoo360/fg-clip-base")
print("STAGE1     =", STAGE1)
print("RERANK_DIR =", RERANK_DIR)
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음 (런타임 유형을 GPU 로 바꾸세요)")
"""),
    md("""
## 3. 설치

- 재정렬기(Qwen3.5)용 transformers 5.x를 시스템에 설치한다 (chan_lee `requirements.txt`와 같은 버전).
- 1단계 FG-CLIP 원격코드는 transformers 4.46.3이 필요해서 `/content/tf446`에 따로 설치하고, 6번에서 `PYTHONPATH`로 앞세워 별도 프로세스로 돌린다.
- `flash-linear-attention`: Qwen3.5의 linear attention CUDA 커널. 설치가 실패하면 느린 기본 구현으로 돌아간다(점수는 같음).
"""),
    code("""
!pip install -q "transformers==5.22.2" "peft==0.21.2" "accelerate==1.15.0"
!pip uninstall -y -q torchao   # Colab 기본 torchao 0.10 은 peft 와 충돌(>=0.16 요구). 재정렬기에는 필요 없음
!pip install -q --no-deps --target /content/tf446 "transformers==4.46.3" "tokenizers==0.20.3" "huggingface_hub==0.26.5" "einops==0.8.1"
!pip install -q flash-linear-attention || echo "flash-linear-attention 설치 실패 -> 기본 구현 사용"
import transformers, peft; print("transformers", transformers.__version__, "| peft", peft.__version__)
"""),
    md("## 4. 데이터: Drive → /content 로 복사 후 압축 해제"),
    code("""
if not TRAIN_TAR:
    TRAIN_TAR = find_in_drive("train_v1.tar")
os.makedirs("/content/data", exist_ok=True)
if not glob.glob("/content/data/**/prompts.jsonl", recursive=True):
    !cp "{TRAIN_TAR}" /content/train_v1.tar
    !tar -xf /content/train_v1.tar -C /content/data
    !rm /content/train_v1.tar
PROMPTS = glob.glob("/content/data/**/prompts.jsonl", recursive=True)[0]
DATA_DIR = os.path.dirname(PROMPTS)
print("DATA_DIR =", DATA_DIR, "| png:", sum(1 for _ in glob.iglob(DATA_DIR + "/images/**/*.png", recursive=True)), "(4500 이어야 함)")
"""),
    md("## 5. 코드 파일 생성 (레포 `finetune/rerank/`와 같은 내용)"),
    code("!mkdir -p /content/work"),
    writefile("stage1_topk.py"),
    writefile("rerank_lib.py"),
    md("""
## 6. 1단계 후보 만들기 (FG-CLIP, 한 번만)

train 3,500문항마다 1단계 top-20(학습용 오답 후보)과 val 500문항마다 top-20(재정렬 대상)을 저장한다. 마지막 줄에 1단계 단독 val 점수와 R@5/R@10(재정렬 상한)이 나온다.
"""),
    code("""
STAGE1_JSON = f"{RERANK_DIR}/stage1_topk.json"
if not os.path.exists(STAGE1_JSON):
    !PYTHONPATH=/content/tf446 python /content/work/stage1_topk.py --stage1 "{STAGE1}" --data "{DATA_DIR}" \\
        --val_ids "{VAL_IDS_PATH}" --out "{STAGE1_JSON}"
print(json.load(open(STAGE1_JSON))["stage1_val"])
"""),
    code("""
import sys, gc
sys.path.insert(0, "/content/work")
import importlib, rerank_lib as R; importlib.reload(R)
data = R.TrainData(DATA_DIR, VAL_IDS_PATH, BUNDLES_PATH, STAGE1_JSON)

def run_eval(rr, name):
    p = f"{RERANK_DIR}/results_{name}.json"
    res = R.evaluate(rr, data, ks=KS, limit=EVAL_LIMIT)
    res.update({"name": name, "reranker": RERANKER, "stage1": STAGE1, "max_pixels": MAX_PIXELS})
    json.dump(res, open(p, "w"), indent=1)
    print(name, json.dumps({k: v for k, v in res.items() if k.startswith("k") or k == "stage1"}, indent=1))
    print("쌍당", res["sec_per_pair"], "s")
    return res

def free(rr):
    del rr; gc.collect(); torch.cuda.empty_cache()

def run_ft(name, use_hn):
    ad = f"{RERANK_DIR}/adapter_{name}"
    if os.path.exists(f"{ad}/adapter_config.json"):
        print("어댑터 있음 -> 학습 건너뜀:", ad)
    else:
        rr = R.Reranker(RERANKER, MAX_PIXELS, lora=LORA)
        log = R.train(rr, data, use_hn=use_hn, n_neg=N_NEG, prompts_per_step=PROMPTS_PER_STEP,
                      grad_accum=GRAD_ACCUM, epochs=EPOCHS, lr=LR, max_prompts=MAX_PROMPTS, seed=SEED)
        rr.save(ad)
        json.dump(log | {"use_hn": use_hn, "n_neg": N_NEG, "lr": LR, "lora": LORA, "epochs": EPOCHS},
                  open(f"{ad}/train_log.json", "w"), indent=1)
        print("학습", log["train_min"], "분 ->", ad)
        free(rr)
    rr = R.Reranker(RERANKER, MAX_PIXELS, adapter=ad)
    res = run_eval(rr, name)
    free(rr)
    return res
"""),
    md("## 7. R0: zero-shot 재정렬 (chan_lee 방식)"),
    code("""
if not os.path.exists(f"{RERANK_DIR}/results_R0.json"):
    rr = R.Reranker(RERANKER, MAX_PIXELS)
    run_eval(rr, "R0"); free(rr)
else:
    print(json.load(open(f"{RERANK_DIR}/results_R0.json")))
"""),
    md("## 8. R1: LoRA 파인튜닝 (형제 + 1단계 오답)\n\n로그의 `acc`는 학습 묶음(정답 1 + 오답 3)에서 정답이 1등인 비율이다. 처음 0.25 근처에서 올라가야 정상이다."),
    code('run_ft("R1", use_hn=False)'),
    md("## 9. R2: LoRA 파인튜닝 + train HN 이미지"),
    code('run_ft("R2", use_hn=True)'),
    md("## 10. 비교 (val 500, 묶음 val + HN 102장)"),
    code("""
import pandas as pd
rows = []
for n in ["R0", "R1", "R2"]:
    p = f"{RERANK_DIR}/results_{n}.json"
    if not os.path.exists(p): continue
    r = json.load(open(p))
    if not rows:
        rows.append({"조건": "1단계 단독", "score": r["stage1"]["score"], "R@1": r["stage1"]["recall_at_1"]})
    for k in KS:
        rows.append({"조건": f"{n} K={k}", "score": r[f"k{k}"]["score"], "R@1": r[f"k{k}"]["recall_at_1"],
                     "쌍당 초": r["sec_per_pair"]})
display(pd.DataFrame(rows))
print("1단계 R@5 / R@10 (K=5 / K=10 의 상한):", {k: json.load(open(STAGE1_JSON))["stage1_val"][k] for k in ("recall_at_5", "recall_at_10")})
"""),
    md("""
## 11. test 재정렬 (test_v1 8,000문항, 후보 15,000장)

1단계(FG-CLIP)로 test 전체 풀에서 문항별 top-10을 뽑고, 그중 top-`TEST_K`를 LoRA 재정렬기로 다시 정렬해 서버 형식 CSV(`prompt_id,prediction` 상위 5개)를 만든다.

- 필요한 Drive 파일: `test_v1.tar` (공유 폴더에서 내 드라이브로 바로가기). **test 프롬프트·이미지·정답은 출력하지 않는다.**
- 시간(L4, 2B, 쌍당 약 0.067초): K=5면 4만 쌍으로 약 45분, K=10이면 약 90분.
- 문항별 점수를 Drive `test_scores_*.jsonl`에 이어 쓴다. **런타임이 끊기면 1~5와 이 셀들을 다시 실행하면 끝난 문항은 건너뛴다.**
- 마지막에 1단계 top-5 그대로의 점수(`stage1_top5`)도 같이 나온다. 이게 Modal 결과 0.6731과 거의 같으면 1단계가 제대로 돈 것이다.
"""),
    code("""
TEST_ADAPTER = "R1"     # 쓸 LoRA 어댑터 (RERANK_DIR/adapter_<이름>). zero-shot 이면 None
TEST_K = 5              # 재정렬할 1단계 후보 수
TEST_TAR = ""           # 비우면 MyDrive 에서 test_v1.tar 자동 탐색

TEST_ROOT = "/content/data_test"          # train 데이터(/content/data)와 섞이지 않게 따로 푼다
if not glob.glob(f"{TEST_ROOT}/**/prompts.jsonl", recursive=True):
    TEST_TAR = TEST_TAR or find_in_drive("test_v1.tar")
    os.makedirs(TEST_ROOT, exist_ok=True)
    !cp "{TEST_TAR}" /content/test_v1.tar
    !tar -xf /content/test_v1.tar -C "{TEST_ROOT}" --no-same-owner
    !rm /content/test_v1.tar
TEST_DIR = os.path.dirname(glob.glob(f"{TEST_ROOT}/**/prompts.jsonl", recursive=True)[0])
print("TEST_DIR =", TEST_DIR, "| png:", sum(1 for _ in glob.iglob(TEST_DIR + "/images/**/*.png", recursive=True)), "(15000 이어야 함)")

STAGE1_TEST_JSON = f"{RERANK_DIR}/stage1_test_topk.json"
if not os.path.exists(STAGE1_TEST_JSON):
    !PYTHONPATH=/content/tf446 python /content/work/stage1_topk.py --test --k_test 10 --stage1 "{STAGE1}" \\
        --data "{TEST_DIR}" --out "{STAGE1_TEST_JSON}"
print("1단계 단독 test:", json.load(open(STAGE1_TEST_JSON))["stage1_test"])
"""),
    code("""
import sys, gc
sys.path.insert(0, "/content/work")
import importlib, rerank_lib as R; importlib.reload(R)
tag = f"{TEST_ADAPTER or 'R0'}_k{TEST_K}"
ad = f"{RERANK_DIR}/adapter_{TEST_ADAPTER}" if TEST_ADAPTER else None
rr = R.Reranker(RERANKER, MAX_PIXELS, adapter=ad)
test_res = R.rerank_test(rr, TEST_DIR, STAGE1_TEST_JSON, f"{RERANK_DIR}/test_{tag}.csv", k=TEST_K,
                         ckpt=f"{RERANK_DIR}/test_scores_{tag}.jsonl")
test_res.update({"adapter": ad, "reranker": RERANKER, "stage1": STAGE1, "max_pixels": MAX_PIXELS,
                 "stage1_test_full": json.load(open(STAGE1_TEST_JSON))["stage1_test"],
                 "gpu": torch.cuda.get_device_name(0), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
json.dump(test_res, open(f"{RERANK_DIR}/test_results_{tag}.json", "w"), indent=1)
print("->", f"{RERANK_DIR}/test_results_{tag}.json")
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("->", OUT)
