from __future__ import annotations

import json
import time
from typing import Any

import torch


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def log_run_summary(
    stage: str,
    started_at: float,
    samples_processed: int,
    unit: str,
    device: torch.device | None = None,
    **metrics: Any,
) -> None:
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    summary: dict[str, Any] = {
        "stage": stage,
        "elapsed_seconds": round(elapsed, 3),
        "samples_processed": samples_processed,
        "unit": unit,
        "throughput_per_second": round(samples_processed / elapsed, 3),
        **metrics,
    }
    if device is not None and device.type == "cuda":
        summary["gpu_peak_memory_mb"] = round(
            torch.cuda.max_memory_allocated(device) / (1024**2), 2
        )
    print(json.dumps(summary, ensure_ascii=False))