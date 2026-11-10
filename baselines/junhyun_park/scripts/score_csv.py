"""결과 CSV(prompt_id,prediction[공백 구분 5개])를 채점해 metrics.json 의 scores / scores_by_split 형식으로 출력.

정답: 기본은 정수로 본 prompt_id == image_id (운영진 패키지는 원본 이미지 id == 프롬프트 id; 5자리/4자리 표기 차이는 무시).
Public = test id 7000-7999 (GUIDE.md), Private = 나머지. 공식 목록이 정해지면 --public_ids 로 교체.
tie 처리: CSV 는 이미 정렬된 상위 5개라 별도 tie 처리 없음.
"""
import argparse, csv, json, math


def agg(rows):
    n = len(rows)
    r1 = sum(1 for r in rows if r == 1) / n
    nd = sum(1 / math.log2(r + 1) for r in rows if r <= 5) / n
    return {"n": n, "recall_at_1": round(r1, 4), "ndcg_at_5": round(nd, 4), "score": round((r1 + nd) / 2, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--answer", help="prompt_id,image_id CSV (생략 시 정수 prompt_id == 정수 image_id)")
    ap.add_argument("--public_ids", help="줄당 test prompt id (생략 시 7000-7999)")
    a = ap.parse_args()
    gt = None
    if a.answer:
        with open(a.answer, newline="") as f:
            r = csv.reader(f); next(r)
            gt = {int(x[0]): int(x[1]) for x in r}
    public = ({int(l) for l in open(a.public_ids) if l.strip()} if a.public_ids else set(range(7000, 8000)))
    ranks = {"all": [], "public": [], "private": []}
    with open(a.csv, newline="") as f:
        r = csv.reader(f); next(r)
        for pid, pred in r:
            p = int(pid)
            ans = gt[p] if gt else p
            ids = [int(x) for x in pred.split()[:5]]
            rank = ids.index(ans) + 1 if ans in ids else 99
            ranks["all"].append(rank)
            ranks["public" if p in public else "private"].append(rank)
    out = {"scores": agg(ranks["all"]),
           "scores_by_split": {"public": agg(ranks["public"]) if ranks["public"] else None,
                               "private": agg(ranks["private"]) if ranks["private"] else None}}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
