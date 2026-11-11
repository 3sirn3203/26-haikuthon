#!/usr/bin/env python3
"""Fetch the public Qwen3.5-2B weight shard into the Hugging Face cache (resumable)."""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parents[3]
HF_ROOT = Path(os.environ.get("HF_HUB_CACHE", REPO_ROOT / "baselines/taekyoung_kwon/.cache/huggingface/hub"))
MODEL = HF_ROOT / "models--Qwen--Qwen3.5-2B"
COMMIT = "15852e8c16360a2fea060d615a32b45270f8a8fc"
BLOB = MODEL / "blobs" / "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
PART = BLOB.with_name(BLOB.name + ".manual.incomplete")
URL = (
    "https://huggingface.co/Qwen/Qwen3.5-2B/resolve/"
    f"{COMMIT}/model.safetensors-00001-of-00001.safetensors"
)
EXPECTED = 4_548_221_488
N_PARTS = 4
_print_lock = threading.Lock()


def main() -> None:
    if BLOB.exists() and BLOB.stat().st_size == EXPECTED:
        print(f"already present: {BLOB}")
        return
    MODEL.joinpath("blobs").mkdir(parents=True, exist_ok=True)
    part_paths = [BLOB.with_name(BLOB.name + f".part{i}") for i in range(N_PARTS)]
    # Reuse bytes fetched by an interrupted single-stream download as the first range.
    if PART.exists() and not part_paths[0].exists():
        os.replace(PART, part_paths[0])
    bounds = [(EXPECTED * i // N_PARTS, EXPECTED * (i + 1) // N_PARTS - 1)
              for i in range(N_PARTS)]

    def fetch(i: int) -> None:
        start, end = bounds[i]
        dest = part_paths[i]
        expected_part = end - start + 1
        offset = dest.stat().st_size if dest.exists() else 0
        if offset > expected_part:
            dest.unlink()
            offset = 0
        if offset == expected_part:
            return
        first = start + offset
        headers = {"Accept-Encoding": "identity", "Range": f"bytes={first}-{end}"}
        with requests.get(URL, headers=headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(f"Range {i} returned HTTP {response.status_code}, expected 206")
            mode = "ab" if offset else "wb"
            last_print = time.time()
            with dest.open(mode) as out:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    out.write(chunk)
                    now = time.time()
                    if now - last_print >= 10:
                        current = dest.stat().st_size
                        with _print_lock:
                            print(f"part {i+1}/{N_PARTS}: {current:,}/{expected_part:,} bytes", flush=True)
                        last_print = now
        if dest.stat().st_size != expected_part:
            raise RuntimeError(f"Part {i} incomplete; rerun to resume")
        with _print_lock:
            print(f"part {i+1}/{N_PARTS} complete", flush=True)

    with ThreadPoolExecutor(max_workers=N_PARTS) as pool:
        list(pool.map(fetch, range(N_PARTS)))

    assembled = BLOB.with_name(BLOB.name + ".manual.assembled")
    with assembled.open("wb") as out:
        for part in part_paths:
            with part.open("rb") as src:
                while chunk := src.read(16 * 1024 * 1024):
                    out.write(chunk)
    size = assembled.stat().st_size
    if size != EXPECTED:
        raise RuntimeError(f"Incomplete weight shard: got {size:,}, expected {EXPECTED:,}; rerun to resume")
    os.replace(assembled, BLOB)
    for part in part_paths:
        part.unlink(missing_ok=True)
    print(f"saved {BLOB} ({size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
