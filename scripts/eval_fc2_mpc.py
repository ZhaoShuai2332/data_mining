#!/usr/bin/env python3
"""Batch evaluation helper for FC2 MPC inference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

def parse_indices(indices_text: str | None, first_n: int | None) -> list[int]:
    if indices_text and first_n is not None:
        raise ValueError("--indices and --first_n are mutually exclusive")
    if not indices_text and first_n is None:
        raise ValueError("Either --indices or --first_n is required")
    if first_n is not None:
        if first_n <= 0:
            raise ValueError("--first_n must be positive")
        return list(range(first_n))
    assert indices_text is not None
    out: list[int] = []
    for item in indices_text.split(","):
        token = item.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("--indices produced an empty list")
    return out


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_python = repo_root / ".venv" / "bin" / "python"
    if not default_python.exists():
        default_python = Path(sys.executable)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Directory containing fixed_params.txt and meta.json")
    parser.add_argument("--mp-spdz-dir", default=None, help="MP-SPDZ directory passed to run_fc2.sh")
    parser.add_argument("--first_n", type=int, default=None, help="Evaluate indices [0, first_n)")
    parser.add_argument("--indices", default=None, help="Comma-separated sample indices, e.g. 0,1,2")
    parser.add_argument("--output-dir", default=None, help="Output directory, default eval_results/<timestamp>")
    parser.add_argument("--python-bin", default=str(default_python), help="Python used to run prepare_input.py")
    args = parser.parse_args()

    python_bin = Path(args.python_bin).expanduser()
    if not python_bin.is_absolute():
        python_bin = (repo_root / python_bin).resolve()
    prepare_script = repo_root / "prepare_input.py"
    run_script = repo_root / "run_fc2.sh"

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not (model_dir / "fixed_params.txt").exists():
        raise FileNotFoundError(f"Missing fixed_params.txt in {model_dir}")
    if not (model_dir / "meta.json").exists():
        raise FileNotFoundError(f"Missing meta.json in {model_dir}")
    if not prepare_script.exists():
        raise FileNotFoundError(f"prepare_input.py not found: {prepare_script}")
    if not run_script.exists():
        raise FileNotFoundError(f"run_fc2.sh not found: {run_script}")
    if not python_bin.exists():
        raise FileNotFoundError(f"python-bin not found: {python_bin}")

    indices = parse_indices(args.indices, args.first_n)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (repo_root / "eval_results" / timestamp).resolve()

    inputs_dir = output_dir / "inputs"
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for idx in indices:
        input_file = inputs_dir / f"input_{idx}.txt"
        run_dir = runs_dir / f"sample_{idx:05d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        prepare_cmd = [
            str(python_bin),
            str(prepare_script),
            "--index",
            str(idx),
            "--outfile",
            str(input_file),
        ]
        try:
            prepare_proc = subprocess.run(
                prepare_cmd,
                check=True,
                cwd=str(repo_root),
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            stdout = e.stdout or ""
            raise RuntimeError(
                f"prepare_input failed for index {idx}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            ) from e
        if prepare_proc.stdout:
            print(prepare_proc.stdout.strip())
        if prepare_proc.stderr:
            print(prepare_proc.stderr.strip())

        label_match = re.search(r"label=(\d+)", prepare_proc.stdout or "")
        if not label_match:
            raise RuntimeError(
                f"Failed to parse true label from prepare_input output for index {idx}: {prepare_proc.stdout!r}"
            )
        true_label = int(label_match.group(1))

        env = os.environ.copy()
        env["MODEL_DIR"] = str(model_dir)
        env["INPUT_FILE"] = str(input_file)
        env["RUN_DIR"] = str(run_dir)
        env["SAMPLE_INDEX"] = str(idx)
        env["TRUE_LABEL"] = str(true_label)
        if args.mp_spdz_dir:
            env["MP_SPDZ_DIR"] = str(Path(args.mp_spdz_dir).expanduser().resolve())

        subprocess.run([str(run_script)], check=True, cwd=str(repo_root), env=env)

        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary.json after run: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        predicted_label = summary.get("predicted_label")
        is_correct = summary.get("is_correct")
        elapsed = summary.get("elapsed_time_seconds")
        total_sent_mb = summary.get("total_sent_mb")

        row = {
            "index": idx,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "is_correct": is_correct,
            "elapsed_time_seconds": elapsed,
            "total_sent_mb": total_sent_mb,
            "run_dir": str(run_dir),
        }
        rows.append(row)

        print(
            f"[sample {idx}] true={true_label} pred={predicted_label} "
            f"correct={is_correct} time={elapsed} total_sent_mb={total_sent_mb}"
        )

    correct_values = [bool(r["is_correct"]) for r in rows if isinstance(r["is_correct"], bool)]
    time_values = [v for v in (as_float(r["elapsed_time_seconds"]) for r in rows) if v is not None]
    sent_values = [v for v in (as_float(r["total_sent_mb"]) for r in rows) if v is not None]

    accuracy = (sum(1 for v in correct_values if v) / len(correct_values)) if correct_values else None
    avg_time_seconds = mean(time_values) if time_values else None
    avg_total_sent_mb = mean(sent_values) if sent_values else None

    results_csv = output_dir / "results.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "true_label",
                "predicted_label",
                "is_correct",
                "elapsed_time_seconds",
                "total_sent_mb",
                "run_dir",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_json = output_dir / "summary.json"
    summary_payload = {
        "sample_count": len(rows),
        "indices": indices,
        "accuracy": accuracy,
        "avg_time_seconds": avg_time_seconds,
        "avg_total_sent_mb": avg_total_sent_mb,
        "results_csv": str(results_csv),
        "rows": rows,
    }
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Batch Evaluation Summary ===")
    print(f"output_dir: {output_dir}")
    print(f"sample_count: {len(rows)}")
    print(f"accuracy: {accuracy}")
    print(f"avg_time_seconds: {avg_time_seconds}")
    print(f"avg_total_sent_mb: {avg_total_sent_mb}")
    print(f"results_csv: {results_csv}")
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    main()
