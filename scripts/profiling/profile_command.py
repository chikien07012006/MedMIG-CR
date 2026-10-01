"""Run a command and record runtime, CPU/RAM of its process tree, GPU usage, and output size.

Example:
    .venv/bin/python scripts/profiling/profile_command.py \
        --label datagen_k3_w6 --output_path out/paths.jsonl --report_json out/profile.json -- \
        .venv/bin/python scripts/reranking/build_gru_path_reranker_dataset.py ... --num_workers 6
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, action="append", default=[], help="File or folder whose size is reported; repeatable.")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command after --")
    return args


def path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return 0


def query_gpus() -> List[Dict[str, float]]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        util, used, total = (float(v) for v in line.split(","))
        gpus.append({"util_pct": util, "mem_used_mb": used, "mem_total_mb": total})
    return gpus


class Sampler(threading.Thread):
    def __init__(self, root_pid: int, interval: float) -> None:
        super().__init__(daemon=True)
        self.root = psutil.Process(root_pid)
        self.interval = interval
        self.samples: List[Dict[str, float]] = []
        self.stop_event = threading.Event()
        self.known: Dict[int, psutil.Process] = {}

    def tree(self) -> List[psutil.Process]:
        # Reuse Process objects: cpu_percent() measures since the previous call on the same object.
        try:
            found = [self.root, *self.root.children(recursive=True)]
        except psutil.NoSuchProcess:
            return []
        procs = []
        for proc in found:
            procs.append(self.known.setdefault(proc.pid, proc))
        return procs

    def run(self) -> None:
        psutil.cpu_percent(interval=None)
        primed: set[int] = set()
        while not self.stop_event.wait(self.interval):
            rss = 0
            proc_cpu = 0.0
            procs = self.tree()
            for proc in procs:
                try:
                    rss += proc.memory_info().rss
                    if proc.pid in primed:
                        proc_cpu += proc.cpu_percent(interval=None)
                    else:
                        proc.cpu_percent(interval=None)
                        primed.add(proc.pid)
                except psutil.NoSuchProcess:
                    continue
            vm = psutil.virtual_memory()
            sample = {
                "t": time.time(),
                "processes": len(procs),
                "tree_rss_gb": rss / 1024**3,
                "tree_cpu_pct": proc_cpu,
                "system_cpu_pct": psutil.cpu_percent(interval=None),
                "system_ram_used_pct": vm.percent,
                "system_ram_available_gb": vm.available / 1024**3,
            }
            gpus = query_gpus()
            if gpus:
                sample["gpu_util_pct"] = max(g["util_pct"] for g in gpus)
                sample["gpu_mem_used_mb"] = sum(g["mem_used_mb"] for g in gpus)
            self.samples.append(sample)


def summarize(values: List[float]) -> Dict[str, float] | None:
    if not values:
        return None
    return {"mean": round(sum(values) / len(values), 2), "peak": round(max(values), 2)}


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    proc = subprocess.Popen(args.command)
    sampler = Sampler(proc.pid, args.interval)
    sampler.start()
    return_code = proc.wait()
    sampler.stop_event.set()
    sampler.join()
    elapsed = time.perf_counter() - started

    samples = sampler.samples
    swap = psutil.swap_memory()
    gpus = query_gpus()
    report = {
        "label": args.label,
        "command": args.command,
        "return_code": return_code,
        "elapsed_seconds": round(elapsed, 2),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
            "gpus": gpus,
        },
        "max_processes": max((s["processes"] for s in samples), default=0),
        "tree_rss_gb": summarize([s["tree_rss_gb"] for s in samples]),
        "tree_cpu_pct": summarize([s["tree_cpu_pct"] for s in samples]),
        "system_cpu_pct": summarize([s["system_cpu_pct"] for s in samples]),
        "system_ram_used_pct": summarize([s["system_ram_used_pct"] for s in samples]),
        "min_system_ram_available_gb": round(min((s["system_ram_available_gb"] for s in samples), default=0.0), 2),
        "swap_used_gb_at_end": round(swap.used / 1024**3, 2),
        "gpu_util_pct": summarize([s["gpu_util_pct"] for s in samples if "gpu_util_pct" in s]),
        "gpu_mem_used_mb": summarize([s["gpu_mem_used_mb"] for s in samples if "gpu_mem_used_mb" in s]),
        "outputs": {str(p): path_size_bytes(p) for p in args.output_path},
        "num_samples": len(samples),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "command"}, indent=2))
    sys.exit(return_code)


if __name__ == "__main__":
    main()
