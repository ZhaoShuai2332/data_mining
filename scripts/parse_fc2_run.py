#!/usr/bin/env python3
"""Parse MP-SPDZ fc2 run logs into machine-readable summary files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_numeric(text: str, pattern: str, cast: type) -> Any:
    regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for match in regex.finditer(text):
        raw = match.group(1).strip()
        try:
            if cast is int:
                return int(float(raw))
            return float(raw)
        except ValueError:
            continue
    return None


def parse_predicted(*texts: str) -> int | None:
    pattern = re.compile(r"Predicted label(?:\s*\([^)]*\))?:\s*(-?\d+)", re.IGNORECASE)
    for text in texts:
        match = pattern.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def maybe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def to_csv_value(value: Any) -> str:
    if value is None:
        return "N/A"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-log", required=True)
    parser.add_argument("--party0-log", required=True)
    parser.add_argument("--party1-log", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--program", default="fc2_mnist_infer")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--sample-index", default=None)
    parser.add_argument("--true-label", default=None)
    parser.add_argument("--fractional-bits", default=None)
    args = parser.parse_args()

    compile_log = Path(args.compile_log)
    party0_log = Path(args.party0_log)
    party1_log = Path(args.party1_log)
    summary_json = Path(args.summary_json)
    summary_csv = Path(args.summary_csv)

    compile_text = read_text(compile_log)
    party0_text = read_text(party0_log)
    party1_text = read_text(party1_log)
    combined_party = "\n".join([party0_text, party1_text])

    predicted_label = parse_predicted(party0_text, party1_text, compile_text)

    elapsed_time_seconds = parse_numeric(
        party0_text,
        r"Time\s*=\s*([0-9.+\-eE]+)\s*seconds",
        float,
    )
    if elapsed_time_seconds is None:
        elapsed_time_seconds = parse_numeric(
            party1_text,
            r"Time\s*=\s*([0-9.+\-eE]+)\s*seconds",
            float,
        )

    party0_sent_mb = parse_numeric(
        party0_text,
        r"Data sent\s*=\s*([0-9.+\-eE]+)\s*MB",
        float,
    )
    party1_sent_mb = parse_numeric(
        party1_text,
        r"Data sent\s*=\s*([0-9.+\-eE]+)\s*MB",
        float,
    )
    total_sent_mb = parse_numeric(
        combined_party,
        r"Global data sent\s*=\s*([0-9.+\-eE]+)\s*MB",
        float,
    )
    if total_sent_mb is None and party0_sent_mb is not None and party1_sent_mb is not None:
        total_sent_mb = party0_sent_mb + party1_sent_mb

    rounds = parse_numeric(
        compile_text,
        r"([0-9]+)\s+virtual machine rounds",
        int,
    )
    if rounds is None:
        rounds = parse_numeric(
            combined_party,
            r"in\s*~?\s*([0-9]+)\s+rounds",
            int,
        )

    triples = parse_numeric(
        compile_text,
        r"([0-9]+)\s+integer triples",
        int,
    )

    summary = {
        "program": args.program,
        "sample_index": maybe_int(args.sample_index),
        "true_label": maybe_int(args.true_label),
        "predicted_label": predicted_label,
        "is_correct": (
            None
            if predicted_label is None or maybe_int(args.true_label) is None
            else predicted_label == maybe_int(args.true_label)
        ),
        "fractional_bits": maybe_int(args.fractional_bits),
        "elapsed_time_seconds": elapsed_time_seconds,
        "party0_sent_mb": party0_sent_mb,
        "party1_sent_mb": party1_sent_mb,
        "total_sent_mb": total_sent_mb,
        "rounds": rounds,
        "triples": triples,
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "compile_log": str(compile_log),
        "party0_log": str(party0_log),
        "party1_log": str(party1_log),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "program",
        "sample_index",
        "true_label",
        "predicted_label",
        "is_correct",
        "fractional_bits",
        "elapsed_time_seconds",
        "party0_sent_mb",
        "party1_sent_mb",
        "total_sent_mb",
        "rounds",
        "triples",
        "run_dir",
        "compile_log",
        "party0_log",
        "party1_log",
        "generated_at_utc",
    ]

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: to_csv_value(summary.get(k)) for k in fieldnames})


if __name__ == "__main__":
    main()
