"""Cross-process global rate limiter for the shared HY4 quota pool.

Measured 2026-11-06: the TokenLink pool is shared and, under load from other
users, sustained throughput collapses to roughly 1 request/second (2 QPS gave
>90% 429s while 1 QPS gave ~90% success).  Every process therefore reserves a
slot in a shared file before issuing a request, so N worker processes together
never exceed the global rate instead of each hammering the endpoint and burning
retries.
"""
from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _qps() -> float:
    try:
        return max(0.1, float(os.environ.get("HY4_GLOBAL_QPS", "1.0")))
    except ValueError:
        return 1.0


class FileRateLimiter:
    def __init__(self, path: str | Path | None = None, qps: float | None = None):
        if path is None:
            # Separate quota buckets: keys known to share a pool must pass the same
            # HY4_RATE_FILE; otherwise fall back to a per-key bucket.
            import hashlib
            group = os.environ.get("HY4_RATE_FILE")
            if not group:
                key = os.environ.get("HY4_API_KEY", "")
                group = "k" + hashlib.sha1(key.encode()).hexdigest()[:8]
            path = ROOT / "cache" / f"rate_{group}.lock"
        self.group = group
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.qps = qps or _qps()
        self.qps_file = self.path.parent / f"rate_qps_{group}.txt"
        self.state_path = self.path.with_suffix(".state")
        if not self.state_path.exists():
            self.state_path.write_text("0")

    def _live_qps(self) -> float:
        """Allow live tuning without restarting workers: write a float to rate_qps_<group>.txt."""
        try:
            if self.qps_file.exists():
                v = float(self.qps_file.read_text().strip())
                if 0.1 <= v <= 50:
                    return v
        except (OSError, ValueError):
            pass
        return self.qps

    def acquire(self) -> None:
        interval = 1.0 / self._live_qps()
        with open(self.path, "a+") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            try:
                try:
                    last = float(self.state_path.read_text().strip() or 0)
                except (OSError, ValueError):
                    last = 0.0
                now = time.time()
                slot = max(now, last + interval)
                self.state_path.write_text(f"{slot:.6f}")
            finally:
                fcntl.flock(lockf, fcntl.LOCK_UN)
        wait = slot - time.time()
        if wait > 0:
            time.sleep(wait)


_LIMITER = None


def acquire() -> None:
    global _LIMITER
    if _LIMITER is None:
        _LIMITER = FileRateLimiter()
    _LIMITER.acquire()
