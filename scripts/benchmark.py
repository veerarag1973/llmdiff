#!/usr/bin/env python3
"""CLI cold-start benchmark for llm-diff.

Measures how long the ``llm-diff --help`` subprocess takes to start and print
help text.  Useful for catching import-time regressions (e.g. accidentally
importing heavy ML models at the top level).

Usage::

    python scripts/benchmark.py            # 20 runs, warn threshold 200 ms
    python scripts/benchmark.py -n 50      # 50 runs
    python scripts/benchmark.py -t 150     # custom warn threshold (ms)
    python scripts/benchmark.py --verbose  # print every individual latency

Exit codes:
    0  All runs completed; average latency is within threshold.
    1  Average latency *exceeded* the threshold (print a warning).
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RUNS: int = 20
DEFAULT_WARN_THRESHOLD_MS: float = 200.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _benchmark_once(cmd: list[str]) -> float:
    """Return wall-clock time in milliseconds for a single subprocess run."""
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True)  # noqa: S603
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if result.returncode not in (0, 1):
        print(
            f"  [WARN] Unexpected exit code {result.returncode}",
            file=sys.stderr,
        )
    return elapsed_ms


def _run_benchmark(
    *,
    runs: int,
    threshold_ms: float,
    verbose: bool,
) -> bool:
    """Run the benchmark and return *True* when average is within threshold."""
    cmd = [sys.executable, "-m", "llm_diff", "--help"]

    print(f"llm-diff cold-start benchmark  ({runs} runs)")
    print(f"Command: {' '.join(cmd)}")
    print(f"Threshold: {threshold_ms:.0f} ms")
    print()

    latencies: list[float] = []

    for i in range(1, runs + 1):
        ms = _benchmark_once(cmd)
        latencies.append(ms)
        if verbose:
            print(f"  run {i:>{len(str(runs))}}: {ms:7.1f} ms")

    avg = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p90 = sorted(latencies)[int(len(latencies) * 0.9)]
    low = min(latencies)
    high = max(latencies)

    print()
    print("Results")
    print("-------")
    print(f"  avg      : {avg:7.1f} ms")
    print(f"  median   : {p50:7.1f} ms")
    print(f"  p90      : {p90:7.1f} ms")
    print(f"  min      : {low:7.1f} ms")
    print(f"  max      : {high:7.1f} ms")
    print()

    passed = avg <= threshold_ms
    if passed:
        print(f"  OK  Average {avg:.1f} ms ≤ {threshold_ms:.0f} ms threshold")
    else:
        print(
            f"  WARN  Average {avg:.1f} ms > {threshold_ms:.0f} ms threshold  "
            f"— possible import-time regression!",
            file=sys.stderr,
        )
    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark llm-diff CLI cold-start latency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--runs",
        type=int, default=DEFAULT_RUNS,
        metavar="INT",
        help="Number of subprocess invocations to run.",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float, default=DEFAULT_WARN_THRESHOLD_MS,
        metavar="MS",
        help="Warn (exit 1) when average latency exceeds this value (ms).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print every individual latency.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    ok = _run_benchmark(
        runs=args.runs,
        threshold_ms=args.threshold,
        verbose=args.verbose,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
