"""VLM Yes/No 재정렬기: zero-shot 채점, LoRA 파인튜닝, val 평가.

채점은 chan_lee rerankers/vlm_yesno.py 와 같다: 같은 질문 문구, enable_thinking=False,
score = logsumexp(logit[Yes, yes, ' Yes']) - logsumexp(logit[No, no, ' No']) (마지막 위치).

파인튜닝 (LoRA, 비전 인코더 고정):
  프롬프트 하나마다 [정답 이미지 1 + 오답 N_NEG] 묶음을 만들고, 묶음 안의 Yes-No 점수로
  정답이 1등이 되도록 softmax cross-entropy(listwise)를 건다. 추론은 쌍마다 독립 채점 그대로다.
  오답 후보: 같은 stage1 묶음의 형제 원본 + 1단계(FG-CLIP) top-k 오답. use_hn=True 면 자기 train HN 이미지를
  (있을 때) 오답 1개로 반드시 넣는다. 묶음 크기는 HN 사용 여부와 무관하게 같다.
"""
from __future__ import annotations

import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

QUESTION = ('Does this image accurately match the following description, including every '
            'object count, color, material, pattern, pose and spatial relation?\n'
            'Description: "{prompt}"\nAnswer with Yes or No.')

_POOL = ThreadPoolExecutor(8)


def load_image(p):
    with Image.open(p) as im:
        return im.convert("RGB")


# ------------------------------------------------------------------ data
class TrainData:
    """train_v1 패키지 + val_ids + 번들 표 + 1단계 후보(stage1_topk.json)."""

    def __init__(self, data_dir, val_ids_path, bundles_path, stage1_json):
        data = Path(data_dir)
        rows = [json.loads(l) for l in open(data / "prompts.jsonl", encoding="utf-8") if l.strip()]
        text = {r["id"]: r.get("prompts", r.get("prompt")) for r in rows}
        self.orig = sorted(i for i, t in text.items() if t != "hard_negative")
        hn = sorted(i for i, t in text.items() if t == "hard_negative")
        w = len(self.orig[0])
        self.norm = lambda x: f"{int(x):0{w}d}"
        self.prompt = {i: text[i] for i in self.orig}
        self.path = {p.stem: p for p in (data / "images").rglob("*.png")}
        self.hn_of = {f"{int(h) - len(self.orig):0{w}d}": h for h in hn}           # 원본 id -> 자기 HN id
        val = {self.norm(l) for l in open(val_ids_path) if l.strip()}
        self.val = sorted(val)
        self.fit = [i for i in self.orig if i not in val]
        b = {self.norm(k): v for k, v in json.load(open(bundles_path)).items()}
        self.siblings = {}
        groups = {}
        for i in self.fit:
            if b.get(i):
                groups.setdefault(b[i], []).append(i)
        for g in groups.values():
            for i in g:
                self.siblings[i] = [j for j in g if j != i]
        s1 = json.load(open(stage1_json))
        self.stage1 = s1
        self.train_topk = s1["train_topk"]
        self.val_topk = s1["val_topk"]
        self.val_gt_rank = s1["val_gt_rank"]
        n_hn_fit = sum(i in self.hn_of for i in self.fit)
        print(f"[data] fit {len(self.fit)} (HN 있는 문항 {n_hn_fit}) / val {len(self.val)} | "
              f"형제 있는 fit 문항 {len(self.siblings)}")

    def negatives(self, pid, n_neg, use_hn, rng, s1_pool=10):
        negs = [self.hn_of[pid]] if (use_hn and pid in self.hn_of) else []
        cand = list(dict.fromkeys(self.siblings.get(pid, []) +
                                  [j for j in self.train_topk[pid][:s1_pool + 1] if j != pid]))
        rng.shuffle(cand)
        return negs + cand[:n_neg - len(negs)]


# ------------------------------------------------------------------ model
class Reranker:
    def __init__(self, model_id, max_pixels=448 * 448, adapter=None, lora=None, grad_ckpt=True):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.dev = torch.device("cuda")
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(self.dev)
        self.proc = AutoProcessor.from_pretrained(model_id, min_pixels=64 * 64, max_pixels=max_pixels)
        self.proc.tokenizer.padding_side = "left"
        tok = self.proc.tokenizer
        self.yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes")]
        self.no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No")]
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            print(f"[rerank] LoRA 어댑터 로드: {adapter}")
        elif lora:
            self._add_lora(**lora, grad_ckpt=grad_ckpt)
        self.model.eval()

    def _add_lora(self, r=16, alpha=32, dropout=0.05, grad_ckpt=True):
        """언어 모델의 Linear 층에만 LoRA (비전 인코더·lm_head 제외). 대상 층 이름은 실제 모델에서 찾는다."""
        from peft import LoraConfig, get_peft_model
        suffixes = set()
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Linear) and not any(k in name for k in ("visual", "vision", "lm_head")):
                suffixes.add(name.rsplit(".", 1)[-1])
        print(f"[lora] 대상 Linear 이름: {sorted(suffixes)}")
        pattern = r"^(?!.*(visual|vision)).*\.(" + "|".join(sorted(suffixes)) + r")$"
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=pattern, bias="none")
        if grad_ckpt:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        self.model = get_peft_model(self.model, cfg)
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()                 # LoRA 가중치는 fp32 로 학습
        self.model.print_trainable_parameters()

    def chat(self, prompt):
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": QUESTION.format(prompt=prompt)}]}]
        return self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)

    def margins(self, prompts, images):
        """(프롬프트, 이미지) 쌍마다 Yes-No 점수. 학습 중이면 그래디언트가 흐른다."""
        inputs = self.proc(text=[self.chat(p) for p in prompts], images=images, return_tensors="pt",
                           padding=True).to(self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(**inputs, logits_to_keep=1).logits[:, -1].float()
        return torch.logsumexp(logits[:, self.yes], -1) - torch.logsumexp(logits[:, self.no], -1)

    def save(self, out):
        self.model.save_pretrained(out)


# ------------------------------------------------------------------ eval
def score_metrics(ranks):
    n = len(ranks)
    r1 = sum(r == 1 for r in ranks) / n
    nd = sum(1 / math.log2(r + 1) for r in ranks if r <= 5) / n
    return {"n": n, "recall_at_1": round(r1, 4), "ndcg_at_5": round(nd, 4), "score": round((r1 + nd) / 2, 4)}


def rerank_ranks(cands, gt, gt_stage1_rank, scores):
    """후보 cands(1단계 순) 를 scores 로 재정렬했을 때 정답 순위. 정답이 후보 밖이면 1단계 순위 그대로.
    동점은 비관적(정답과 점수가 같은 다른 후보를 위로)."""
    if gt not in cands:
        return gt_stage1_rank
    g = scores[cands.index(gt)]
    return 1 + sum(1 for c, s in zip(cands, scores) if c != gt and s >= g)


@torch.no_grad()
def evaluate(rr: Reranker, data: TrainData, ks=(5, 10), batch=10, limit=None):
    rr.model.eval()
    kmax = max(ks)
    qs = data.val[:limit] if limit else data.val
    out = {k: [] for k in ks}
    t0, n_pairs = time.time(), 0
    for qi, q in enumerate(qs):
        cands = data.val_topk[q][:kmax]
        ims = list(_POOL.map(load_image, [data.path[c] for c in cands]))
        sc = []
        for s in range(0, len(cands), batch):
            sc += rr.margins([data.prompt[q]] * len(ims[s:s + batch]), ims[s:s + batch]).tolist()
        n_pairs += len(cands)
        for k in ks:
            out[k].append(rerank_ranks(cands[:k], q, data.val_gt_rank[q], sc[:k]))
        if qi % 100 == 0:
            print(f"  [eval] {qi}/{len(qs)}  {time.time() - t0:.0f}s", flush=True)
    sec = time.time() - t0
    res = {f"k{k}": score_metrics(out[k]) for k in ks}
    res["stage1"] = score_metrics([data.val_gt_rank[q] for q in qs])
    res["sec_per_pair"] = round(sec / n_pairs, 4)
    res["eval_sec"] = round(sec, 1)
    return res


# ------------------------------------------------------------------ train
def train(rr: Reranker, data: TrainData, *, use_hn, n_neg=3, prompts_per_step=2, grad_accum=4, epochs=1,
          lr=1e-4, warmup=0.05, max_prompts=None, seed=0, log_every=20):
    rng = random.Random(seed)
    params = [p for p in rr.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    n_prompts = min(len(data.fit), max_prompts or len(data.fit))
    steps = epochs * math.ceil(n_prompts / (prompts_per_step * grad_accum))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(warmup * steps))) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    log, t0, opt_step = [], time.time(), 0
    for ep in range(epochs):
        rr.model.train()
        order = data.fit[:]
        rng.shuffle(order)
        order = order[:n_prompts]
        run_loss, run_acc = [], []
        for i in range(0, len(order), prompts_per_step):
            group = order[i:i + prompts_per_step]
            prompts, paths, sizes = [], [], []
            for q in group:
                cands = [q] + data.negatives(q, n_neg, use_hn, rng)
                prompts += [data.prompt[q]] * len(cands)
                paths += [data.path[c] for c in cands]
                sizes.append(len(cands))
            ims = list(_POOL.map(load_image, paths))
            m = rr.margins(prompts, ims)
            # 묶음별 softmax CE (정답이 각 묶음의 0번)
            loss, acc, s = 0, 0, 0
            for sz in sizes:
                g = m[s:s + sz]
                loss = loss + F.cross_entropy(g[None], torch.zeros(1, dtype=torch.long, device=g.device))
                acc += int(len(g) == 1 or bool(g[0] > g[1:].max()))    # 동점은 오답 처리
                s += sz
            loss = loss / len(sizes)
            (loss / grad_accum).backward()
            run_loss.append(loss.item()); run_acc.append(acc / len(sizes))
            if (i // prompts_per_step + 1) % grad_accum == 0 or i + prompts_per_step >= len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % log_every == 0:
                    el = time.time() - t0
                    done = (ep * len(order) + i + len(group)) / (epochs * len(order))
                    print(f"[ep {ep}] step {opt_step}/{steps} loss {sum(run_loss[-log_every * grad_accum:]) / len(run_loss[-log_every * grad_accum:]):.4f} "
                          f"acc {sum(run_acc[-log_every * grad_accum:]) / len(run_acc[-log_every * grad_accum:]):.3f} "
                          f"lr {sched.get_last_lr()[0]:.2e} | {el / 60:.1f}분, 남은 시간 약 {el / done * (1 - done) / 60:.0f}분", flush=True)
        log.append({"epoch": ep, "loss": sum(run_loss) / len(run_loss), "acc": sum(run_acc) / len(run_acc)})
    rr.model.eval()
    return {"epochs": log, "train_min": round((time.time() - t0) / 60, 1), "opt_steps": opt_step}
