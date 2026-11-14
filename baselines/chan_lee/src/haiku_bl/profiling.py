"""Device selection, stage timers and peak-memory tracking.

MPS timings are reference only; rule decisions use CUDA measurements.
On CUDA peak memory is ``torch.cuda.max_memory_allocated``. MPS has no peak counter,
so we sample ``torch.mps.driver_allocated_memory`` at every ``sample()`` call (once per
batch in the encoders) and keep the maximum. That is an upper bound on tensor memory
(it includes the allocator cache) and can miss short spikes between samples.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
import warnings
from contextlib import contextmanager

import torch


def pick_device(pref: str = 'auto') -> torch.device:
    if pref != 'auto':
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def device_label(device: torch.device) -> str:
    if device.type == 'cuda':
        return 'cuda:' + torch.cuda.get_device_name(device).replace('NVIDIA ', '').replace(' ', '')
    if device.type == 'mps':
        try:
            chip = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True).stdout.strip()
        except OSError:
            chip = 'Apple'
        return 'mps:' + chip.replace('Apple ', '').replace(' ', '')
    return 'cpu:' + (platform.processor() or platform.machine())


def sync(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


class Profiler:
    def __init__(self, device: torch.device, vram_budget_gb: float | None = None):
        self.device = device
        self.budget = vram_budget_gb
        self.times: dict[str, float] = {}
        self._peak = 0
        self.reset_peak()
        self._t0 = time.perf_counter()

    def reset_peak(self):
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        self._peak = 0

    def sample(self):
        if self.device.type == 'mps':
            self._peak = max(self._peak, torch.mps.driver_allocated_memory())

    @property
    def peak_gb(self) -> float:
        if self.device.type == 'cuda':
            return torch.cuda.max_memory_allocated(self.device) / 1024**3
        self.sample()
        return self._peak / 1024**3

    @contextmanager
    def stage(self, name: str):
        """Accumulate synchronized wall time under ``name`` (repeated stages add up)."""
        sync(self.device)
        t = time.perf_counter()
        try:
            yield
        finally:
            sync(self.device)
            self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - t
            self.sample()

    def add(self, name: str, seconds: float):
        self.times[name] = self.times.get(name, 0.0) + seconds

    def total(self) -> float:
        return time.perf_counter() - self._t0

    def check_budget(self) -> bool:
        ok = self.budget is None or self.peak_gb <= self.budget
        if not ok:
            warnings.warn(f'peak memory {self.peak_gb:.2f} GB exceeds the '
                          f'{self.budget} GB budget', stacklevel=2)
        return ok

    def report(self) -> dict:
        return {'device': device_label(self.device), 'stages_sec': dict(self.times),
                'total_sec': self.total(), 'peak_vram_gb': self.peak_gb,
                'vram_budget_gb': self.budget, 'within_budget': self.check_budget()}

    def dump(self, path):
        with open(path, 'w') as f:
            json.dump(self.report(), f, indent=2)


def param_count_b(*modules) -> float:
    return sum(p.numel() for m in modules if m is not None for p in m.parameters()) / 1e9


_progress_t0 = {}


def log_progress(done: int, total: int, every: float = 0.05, key: str = 'images'):
    """Print 'images 3000/15000 (20%) 12.3/s eta 16.2 min' to stderr about every 5%."""
    import sys
    now = time.perf_counter()
    t0 = _progress_t0.setdefault(key, [now, done, -1])   # start time, start count, last bucket
    step = max(1, int(total * every))
    if done // step == t0[2] and done != total:
        return
    t0[2] = done // step
    rate = (done - t0[1]) / max(now - t0[0], 1e-9)
    eta = (total - done) / rate / 60 if rate > 0 else float('nan')
    print(f'{key} {done}/{total} ({100 * done / total:.0f}%) {rate:.1f}/s eta {eta:.1f} min', file=sys.stderr, flush=True)
    if done == total:
        _progress_t0.pop(key, None)
