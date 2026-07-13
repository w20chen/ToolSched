"""Finite-duration co-runner used by the Linux placement collector."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time


_stop = False


def _request_stop(_signum, _frame) -> None:
    global _stop
    _stop = True


def cpu_work(max_seconds: float) -> tuple[int, float]:
    payload = b"ToolSched placement co-runner"
    operations = 0
    start = time.perf_counter()
    deadline = start + max_seconds
    while not _stop and time.perf_counter() < deadline:
        for _ in range(512):
            payload = hashlib.sha256(payload).digest()
        operations += 512
    return operations, max(1e-9, time.perf_counter() - start)


def cache_work(max_seconds: float, size_mb: int) -> tuple[int, float]:
    # A private working set larger than typical private caches. The stride is
    # cache-line sized and the index permutation prevents a trivial streaming
    # prefetch-only workload.
    data = bytearray(max(1, size_mb) * 1024 * 1024)
    mask = len(data) - 1
    use_mask = len(data) & (len(data) - 1) == 0
    index = 0
    operations = 0
    start = time.perf_counter()
    deadline = start + max_seconds
    while not _stop and time.perf_counter() < deadline:
        for _ in range(8192):
            index = (index * 1103515245 + 12345) & 0x7FFFFFFF
            pos = (index & mask) if use_mask else index % len(data)
            data[pos] = (data[pos] + 1) & 0xFF
        operations += 8192
    return operations, max(1e-9, time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("cpu", "cache"), required=True)
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    parser.add_argument("--working-set-mb", type=int, default=64)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    max_seconds = max(0.01, args.max_seconds)
    if args.kind == "cpu":
        operations, elapsed = cpu_work(max_seconds)
    else:
        operations, elapsed = cache_work(max_seconds, args.working_set_mb)
    print(
        json.dumps(
            {
                "kind": args.kind,
                "operations": operations,
                "elapsed_seconds": elapsed,
                "operations_per_second": operations / elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
