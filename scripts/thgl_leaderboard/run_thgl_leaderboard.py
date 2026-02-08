#!/usr/bin/env python3
"""Resumable launcher for reproducing THGL leaderboard baselines.

This script runs dataset/method combinations using the existing TGB example scripts,
tracks job state on disk, streams logs to files, and snapshots metrics after each job.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]

THGL_DATASETS = ["thgl-software", "thgl-forum", "thgl-github", "thgl-myket"]
STOCHASTIC_MODELS = {"tgn", "tgn_edge_type", "sthn"}


@dataclass
class Job:
    job_id: str
    dataset: str
    model: str
    seed: int
    command: List[str]
    log_path: Path
    script_dir: Path


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    out = value.strip().lower()
    if out in {"1", "true", "yes", "y"}:
        return True
    if out in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("THGL leaderboard runner")
    parser.add_argument("--run-dir", default="runs/default", help="Relative path under scripts/thgl_leaderboard")
    parser.add_argument("--datasets", nargs="+", default=THGL_DATASETS, choices=THGL_DATASETS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", "edgebank", "tgn", "tgn_edge_type", "sthn", "recurrencybaseline"],
        help="Model families to run",
    )
    parser.add_argument("--num-seeds", type=int, default=5, help="Seeds per stochastic method")
    parser.add_argument("--seed-start", type=int, default=1, help="First seed value")

    parser.add_argument("--tgn-num-epoch", type=int, default=30)
    parser.add_argument("--tgn-lr", type=float, default=1e-4)
    parser.add_argument("--tgn-bs", type=int, default=200)
    parser.add_argument("--tgn-mem-dim", type=int, default=100)
    parser.add_argument("--tgn-time-dim", type=int, default=100)
    parser.add_argument("--tgn-emb-dim", type=int, default=100)
    parser.add_argument("--tgn-patience", type=int, default=5)
    parser.add_argument("--tgn-tolerance", type=float, default=1e-6)

    parser.add_argument("--sthn-epochs", type=int, default=2)
    parser.add_argument("--sthn-batch-size", type=int, default=600)
    parser.add_argument("--sthn-lr", type=float, default=5e-4)
    parser.add_argument("--sthn-max-edges", type=int, default=50)
    parser.add_argument("--sthn-num-neighbors", type=int, default=50)

    parser.add_argument("--edgebank-bs", type=int, default=200)
    parser.add_argument("--edgebank-time-window-ratio", type=float, default=0.15)

    parser.add_argument("--rb-num-processes", type=int, default=1)
    parser.add_argument("--rb-window", type=int, default=0)
    parser.add_argument("--rb-lmbda", type=float, default=0.1)
    parser.add_argument("--rb-alpha", type=float, default=0.99)

    parser.add_argument("--compile-sthn-sampler", type=str2bool, default=True)
    parser.add_argument("--continue-on-error", type=str2bool, default=True)
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def should_run_model(model: str, requested: List[str]) -> bool:
    if "all" in requested:
        return True
    return model in requested


def leaderboard_models_for_dataset(dataset: str, requested: List[str]) -> List[str]:
    if dataset in {"thgl-github", "thgl-myket"}:
        candidates = ["edgebank"]
    else:
        candidates = ["sthn", "tgn_edge_type", "tgn", "edgebank", "recurrencybaseline"]
    return [m for m in candidates if should_run_model(m, requested)]


def get_script_dir(dataset: str) -> Path:
    return REPO_ROOT / "examples" / "linkproppred" / dataset


def get_results_file(dataset: str, model: str, variant: str = "") -> Path:
    base = get_script_dir(dataset) / "saved_results"
    if model == "edgebank":
        return base / f"EdgeBank_{variant}_{dataset}_results.json"
    if model == "tgn" or model == "tgn_edge_type":
        return base / f"TGN_{dataset}_results.json"
    if model == "sthn":
        return base / f"STHN_{dataset}_results.json"
    if model == "recurrencybaseline":
        return base / f"RecurrencyBaseline_NONE_{dataset}_results.json"
    raise ValueError(f"Unknown model: {model}")


def build_jobs(args: argparse.Namespace, logs_dir: Path) -> List[Job]:
    jobs: List[Job] = []
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    for dataset in args.datasets:
        model_list = leaderboard_models_for_dataset(dataset, args.models)
        dataset_script_dir = get_script_dir(dataset)

        for model in model_list:
            if model == "edgebank":
                for mem_mode in ["unlimited", "fixed_time_window"]:
                    job_id = f"{dataset}__edgebank__{mem_mode}"
                    log_path = logs_dir / f"{job_id}.log"
                    cmd = [
                        sys.executable,
                        "edgebank.py",
                        "--data",
                        dataset,
                        "--seed",
                        str(args.seed_start),
                        "--bs",
                        str(args.edgebank_bs),
                        "--mem_mode",
                        mem_mode,
                        "--time_window_ratio",
                        str(args.edgebank_time_window_ratio),
                    ]
                    jobs.append(Job(job_id, dataset, "edgebank", args.seed_start, cmd, log_path, dataset_script_dir))

            elif model == "recurrencybaseline":
                job_id = f"{dataset}__recurrencybaseline"
                log_path = logs_dir / f"{job_id}.log"
                cmd = [
                    sys.executable,
                    "recurrencybaseline.py",
                    "--dataset",
                    dataset,
                    "--seed",
                    str(args.seed_start),
                    "--num_processes",
                    str(args.rb_num_processes),
                    "--window",
                    str(args.rb_window),
                    "--lmbda",
                    str(args.rb_lmbda),
                    "--alpha",
                    str(args.rb_alpha),
                    "--train_flag",
                    "False",
                ]
                jobs.append(Job(job_id, dataset, "recurrencybaseline", args.seed_start, cmd, log_path, dataset_script_dir))

            elif model in {"tgn", "tgn_edge_type"}:
                use_edge_type = model == "tgn_edge_type"
                for seed in seeds:
                    job_id = f"{dataset}__{model}__seed{seed}"
                    log_path = logs_dir / f"{job_id}.log"
                    cmd = [
                        sys.executable,
                        "tgn.py",
                        "--seed",
                        str(seed),
                        "--num_run",
                        "1",
                        "--num_epoch",
                        str(args.tgn_num_epoch),
                        "--lr",
                        str(args.tgn_lr),
                        "--bs",
                        str(args.tgn_bs),
                        "--mem_dim",
                        str(args.tgn_mem_dim),
                        "--time_dim",
                        str(args.tgn_time_dim),
                        "--emb_dim",
                        str(args.tgn_emb_dim),
                        "--patience",
                        str(args.tgn_patience),
                        "--tolerance",
                        str(args.tgn_tolerance),
                    ]
                    cmd += ["--use_edge_type", "1" if use_edge_type else "0", "--use_node_type", "1"]
                    jobs.append(Job(job_id, dataset, model, seed, cmd, log_path, dataset_script_dir))

            elif model == "sthn":
                for seed in seeds:
                    job_id = f"{dataset}__sthn__seed{seed}"
                    log_path = logs_dir / f"{job_id}.log"
                    cmd = [
                        sys.executable,
                        "sthn.py",
                        "--seed",
                        str(seed),
                        "--num_run",
                        "1",
                        "--epochs",
                        str(args.sthn_epochs),
                        "--batch_size",
                        str(args.sthn_batch_size),
                        "--lr",
                        str(args.sthn_lr),
                        "--max_edges",
                        str(args.sthn_max_edges),
                        "--num_neighbors",
                        str(args.sthn_num_neighbors),
                    ]
                    jobs.append(Job(job_id, dataset, "sthn", seed, cmd, log_path, dataset_script_dir))

    return jobs


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def read_results_payload(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = load_json(path, [])
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    return []


def extract_result_for_job(job: Job) -> Dict[str, Any]:
    if job.model == "edgebank":
        variant = "unlimited" if "unlimited" in job.job_id else "fixed_time_window"
        result_file = get_results_file(job.dataset, job.model, variant=variant)
        records = read_results_payload(result_file)
        return records[-1] if records else {}

    if job.model in {"tgn", "tgn_edge_type", "sthn"}:
        result_file = get_results_file(job.dataset, job.model)
        records = read_results_payload(result_file)
        if not records:
            return {}
        matched = [r for r in records if int(r.get("seed", -999999)) == int(job.seed)]
        return matched[-1] if matched else records[-1]

    if job.model == "recurrencybaseline":
        result_file = get_results_file(job.dataset, job.model)
        records = read_results_payload(result_file)
        return records[-1] if records else {}

    return {}


def _is_numeric_series(values: Any) -> bool:
    if not isinstance(values, list):
        return False
    if not values:
        return True
    for item in values:
        if not isinstance(item, (int, float)):
            return False
    return True


def _compute_series_envelope(series_list: List[List[float]]) -> Dict[str, Any]:
    if not series_list:
        return {"num_runs": 0, "epochs": []}
    max_len = max(len(s) for s in series_list)
    epochs = []
    for epoch_idx in range(max_len):
        vals = [float(s[epoch_idx]) for s in series_list if epoch_idx < len(s)]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        epochs.append(
            {
                "epoch": epoch_idx + 1,
                "count": len(vals),
                "mean": mean,
                "std": std,
                "lower": mean - std,
                "upper": mean + std,
                "min": min(vals),
                "max": max(vals),
            }
        )
    return {"num_runs": len(series_list), "epochs": epochs}


def update_loss_envelope(run_dir: Path, dataset: str, model: str) -> None:
    script_dir = get_script_dir(dataset)
    curves_dir = script_dir / "saved_results" / "loss_curves"
    if not curves_dir.exists():
        return

    if model == "tgn_edge_type":
        pattern = f"TGN_edge_type_{dataset}_seed*_run*.json"
    elif model == "tgn":
        pattern = f"TGN_plain_{dataset}_seed*_run*.json"
    elif model == "sthn":
        pattern = f"STHN_{dataset}_seed*_run*.json"
    else:
        return

    curve_files = sorted(curves_dir.glob(pattern))
    if not curve_files:
        return

    records = []
    for path in curve_files:
        rec = load_json(path, default={})
        if isinstance(rec, dict):
            rec["__file"] = str(path)
            records.append(rec)

    if not records:
        return

    series_keys = set()
    for rec in records:
        for k, v in rec.items():
            if _is_numeric_series(v):
                series_keys.add(k)

    envelope = {
        "dataset": dataset,
        "model": model,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_curves": len(records),
        "curve_files": [r["__file"] for r in records],
        "series": {},
    }

    for key in sorted(series_keys):
        curves = [rec.get(key, []) for rec in records if _is_numeric_series(rec.get(key))]
        envelope["series"][key] = _compute_series_envelope(curves)

    out_dir = run_dir / "envelopes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{dataset}__{model}_loss_envelope.json"
    write_json(out_file, envelope)


def run_job(job: Job, state: Dict[str, Any], state_path: Path) -> int:
    state["jobs"][job.job_id]["status"] = "running"
    state["jobs"][job.job_id]["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(state_path, state)

    print(f"\\n=== Running: {job.job_id} ===")
    print(f"cwd: {job.script_dir}")
    print(f"log: {job.log_path}")
    print(f"cmd: {' '.join(job.command)}")

    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    with job.log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\\n# START {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
        logf.write("# CMD " + " ".join(job.command) + "\\n")
        logf.flush()

        proc = subprocess.Popen(
            job.command,
            cwd=str(job.script_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()

        logf.write(f"# END rc={proc.returncode} at {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")

    return int(proc.returncode)


def compile_sthn_sampler_if_needed(args: argparse.Namespace, selected_datasets: List[str], selected_models: List[str], run_dir: Path) -> None:
    needs_sthn = False
    for ds in selected_datasets:
        ds_models = leaderboard_models_for_dataset(ds, selected_models)
        if "sthn" in ds_models:
            needs_sthn = True
            break

    if not needs_sthn or not args.compile_sthn_sampler:
        return

    log_path = run_dir / "logs" / "sthn_sampler_build.log"
    cmd = [sys.executable, "sthn_sampler_setup.py", "build_ext", "--inplace"]
    print("Compiling STHN sampler extension...")

    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\\n# START {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
        logf.write("# CMD " + " ".join(cmd) + "\\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT / "modules"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
        logf.write(f"# END rc={proc.returncode} at {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")

    if proc.returncode != 0:
        raise RuntimeError("Failed to compile STHN sampler. Check logs/sthn_sampler_build.log")


def initialize_state(jobs: List[Job], state_path: Path) -> Dict[str, Any]:
    state = load_json(state_path, default={"jobs": {}, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    jobs_obj = state.setdefault("jobs", {})

    for job in jobs:
        if job.job_id not in jobs_obj:
            jobs_obj[job.job_id] = {
                "dataset": job.dataset,
                "model": job.model,
                "seed": job.seed,
                "status": "pending",
                "command": job.command,
                "log_path": str(job.log_path),
                "started_at": None,
                "finished_at": None,
                "return_code": None,
                "result": None,
            }
        elif jobs_obj[job.job_id].get("status") == "running":
            jobs_obj[job.job_id]["status"] = "pending"

    write_json(state_path, state)
    return state


def main() -> int:
    args = parse_args()
    run_dir = (REPO_ROOT / "scripts" / "thgl_leaderboard" / args.run_dir).resolve()
    logs_dir = run_dir / "logs"
    state_path = run_dir / "state.json"
    snapshot_path = run_dir / "results_snapshot.jsonl"

    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args, logs_dir)
    if not jobs:
        print("No jobs selected. Nothing to run.")
        return 0

    compile_sthn_sampler_if_needed(args, args.datasets, args.models, run_dir)

    state = initialize_state(jobs, state_path)

    print(f"Run directory: {run_dir}")
    print(f"Total jobs in queue: {len(jobs)}")

    if args.dry_run:
        for job in jobs:
            print(f"[DRY RUN] {job.job_id}: {' '.join(job.command)}")
        return 0

    failures = 0
    for job in jobs:
        job_state = state["jobs"][job.job_id]
        if job_state.get("status") == "completed":
            print(f"Skipping completed job: {job.job_id}")
            continue

        rc = run_job(job, state, state_path)
        job_state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        job_state["return_code"] = rc

        if rc == 0:
            job_state["status"] = "completed"
            result = extract_result_for_job(job)
            job_state["result"] = result
            if job.model in STOCHASTIC_MODELS:
                update_loss_envelope(run_dir, job.dataset, job.model)
            append_jsonl(
                snapshot_path,
                {
                    "job_id": job.job_id,
                    "dataset": job.dataset,
                    "model": job.model,
                    "seed": job.seed,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "result": result,
                },
            )
        else:
            job_state["status"] = "failed"
            failures += 1
            if not args.continue_on_error:
                write_json(state_path, state)
                print(f"Stopping after failure in {job.job_id}")
                return rc

        write_json(state_path, state)

    print("\\nRun complete.")
    print(f"Failures: {failures}")
    print(f"State: {state_path}")
    print(f"Snapshots: {snapshot_path}")

    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
