#!/usr/bin/env python3
"""Create assignment summary metrics and failure-inspection candidates."""

import argparse
import csv
import json
from pathlib import Path

from evaloop.analysis.metrics import asl_semantic
from evaloop.analysis.similarity import (
    default_similarity_path,
    extract_boundary_pair,
    group_scores_by_bucket,
    load_similarity_scores,
)


def summarize(path: Path, judge_name: str) -> tuple[dict, list[dict]]:
    """Return one summary row and compact records for all failed loops."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"Expected one model result in {path}")

    result = payload[0]
    tasks = result["prompt_results"]
    max_cycles = int(result.get("max_cycles", 10))
    n_tasks = len(tasks)
    if n_tasks == 0:
        raise ValueError(f"No task results in {path}")

    sustained = [int(task["successful_cycles"]) for task in tasks]
    pairs = [
        pair
        for task in tasks
        if (pair := extract_boundary_pair(task, max_cycles)) is not None
    ]
    sidecar_path = default_similarity_path(path, judge_name)
    cached = load_similarity_scores(sidecar_path)
    scores = {key: value.score for key, value in cached.items()}
    missing = [pair.key for pair in pairs if scores.get(pair.key) is None]
    if missing:
        raise ValueError(
            f"{sidecar_path} lacks usable scores for {len(missing)} boundaries; "
            "run scripts/judge_assignment4.py first"
        )
    scores_by_bucket = group_scores_by_bucket(pairs, scores)

    row = {
        "model": result["model"],
        "configuration": result.get("configuration", "standard"),
        "n_tasks": n_tasks,
        "success_rate": sum(value >= 1 for value in sustained) / n_tasks,
        "asl_score": asl_semantic(sustained, max_cycles, scores_by_bucket),
    }

    failures = []
    for task in tasks:
        if task["successful_cycles"] >= max_cycles:
            continue
        cycles = task.get("cycles", [])
        failing_cycle = next(
            (cycle for cycle in cycles if not cycle.get("success", False)), None
        )
        previous = None
        if failing_cycle is not None:
            failing_number = failing_cycle["cycle"]
            previous = next(
                (
                    cycle
                    for cycle in cycles
                    if cycle["cycle"] == failing_number - 1
                ),
                None,
            )
        failures.append(
            {
                "source_file": path.name,
                "model": result["model"],
                "configuration": result.get("configuration", "standard"),
                "task_id": task["task_id"],
                "successful_cycles": task["successful_cycles"],
                "initial_prompt": task.get("initial_prompt"),
                "last_successful_summary": (
                    previous.get("task_2_output") if previous else None
                ),
                "failing_code": (
                    failing_cycle.get("task_1_output") if failing_cycle else None
                ),
                "test_results": (
                    failing_cycle.get("test_results") if failing_cycle else None
                ),
            }
        )
    return row, failures


def main() -> None:
    """Summarize every formal experiment result in the provided directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--judge-name", default="qwen35-27b-local")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    paths = sorted(results_dir.glob("*_results.json"))
    if not paths:
        raise SystemExit(f"no result JSON files found under {results_dir}")

    rows = []
    failures = []
    for path in paths:
        row, file_failures = summarize(path, args.judge_name)
        rows.append(row)
        failures.extend(file_failures)

    summary_path = results_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "configuration",
                "n_tasks",
                "success_rate",
                "asl_score",
            ],
        )
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["success_rate"] = f"{row['success_rate']:.6f}"
            row["asl_score"] = f"{row['asl_score']:.6f}"
            writer.writerow(row)

    failures_path = results_dir / "failure_candidates.json"
    failures_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {summary_path}")
    print(f"wrote {failures_path} ({len(failures)} failures)")


if __name__ == "__main__":
    main()
