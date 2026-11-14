"""공개 train 프롬프트 텍스트만으로 만드는 규칙 기반 text hard negative (slot 하나만 바꾼 변형).

참가자도 쓸 수 있는 정보(프롬프트 문장)만 사용한다. 주최측 비공개 variants 파일은 쓰지 않는다.
종류: direction(left<->right, above<->below) / count(two<->three, three<->four; both<->all three) / attribute(색상).
"""
from __future__ import annotations

import random
import re

COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white",
          "gray", "grey", "beige", "silver", "gold", "teal", "cyan", "maroon", "navy", "turquoise"]
_DIR = {"left": "right", "right": "left", "above": "below", "below": "above"}
_CNT = {"two": "three", "three": "two", "four": "three"}


def _match_case(src: str, dst: str) -> str:
    return dst.capitalize() if src[:1].isupper() else dst


def swap_direction(text: str, rng: random.Random):
    hits = list(re.finditer(r"\b(left|right|above|below)\b", text, flags=re.I))
    if not hits:
        return None
    m = rng.choice(hits)
    w = m.group(0)
    return text[:m.start()] + _match_case(w, _DIR[w.lower()]) + text[m.end():]


def swap_count(text: str, rng: random.Random):
    # "(two|three|four) <복수명사>" 형태만 (단수<->복수는 문법이 깨져 제외)
    hits = list(re.finditer(r"\b(two|three|four)\s+(?!objects\b)([a-z-]+s)\b", text, flags=re.I))
    if not hits:
        return None
    m = rng.choice(hits)
    old = m.group(1)
    new = _CNT[old.lower()]
    out = text[:m.start(1)] + _match_case(old, new) + text[m.end(1):]
    if old.lower() == "two" and new == "three":
        out = re.sub(r"\bBoth\b", "All three", out)
        out = re.sub(r"\bboth\b", "all three", out)
    elif old.lower() == "three" and new == "two":
        out = re.sub(r"\bAll three\b", "Both", out)
        out = re.sub(r"\ball three\b", "both", out)
    return out


def swap_attribute(text: str, rng: random.Random):
    pat = r"\b(" + "|".join(COLORS) + r")\b"
    hits = list(re.finditer(pat, text, flags=re.I))
    if not hits:
        return None
    m = rng.choice(hits)
    old = m.group(0)
    new = rng.choice([c for c in COLORS if c != old.lower() and c not in ("grey",) ])
    return text[:m.start()] + _match_case(old, new) + text[m.end():]


RULES = {"direction": swap_direction, "count": swap_count, "attribute": swap_attribute}


def make_hard_negatives(text: str, k: int, rng: random.Random):
    """가능한 규칙 중 최대 k개(서로 다른 종류)를 적용해 [(kind, text)] 반환. 원문과 같은 결과는 버림."""
    kinds = list(RULES)
    rng.shuffle(kinds)
    out = []
    for kind in kinds:
        new = RULES[kind](text, rng)
        if new and new != text:
            out.append((kind, new))
        if len(out) >= k:
            break
    return out
