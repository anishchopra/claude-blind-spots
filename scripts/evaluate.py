#!/usr/bin/env python3
"""Evaluate predictions against ground truth.

Usage: python -m scripts.evaluate --run exp_01
"""

import argparse
import json
from pathlib import Path

from src.generators import GENERATORS
from src.utils import load_run_metadata


def evaluate_task(run_name: str, data_dir: Path) -> dict:
    """Evaluate predictions for a single task run.

    Args:
        run_name: Run identifier
        data_dir: Root data directory

    Returns:
        Report dict with accuracy stats
    """
    # Path: data/{run_name}/
    task_dir = data_dir / run_name

    if not task_dir.exists():
        return {"error": f"Task directory not found: {task_dir}"}

    # Load task name from metadata
    task_name = load_run_metadata(data_dir, run_name)["task"]

    results = []

    for sample_dir in task_dir.iterdir():
        if not sample_dir.is_dir():
            continue

        metadata_path = sample_dir / "metadata.json"
        prediction_path = sample_dir / "prediction.json"

        if not metadata_path.exists():
            continue

        if not prediction_path.exists():
            # No prediction yet
            continue

        with open(metadata_path) as f:
            metadata = json.load(f)

        with open(prediction_path) as f:
            prediction_data = json.load(f)

        ground_truth = metadata["ground_truth"]
        prediction = prediction_data["prediction"]

        results.append({
            "sample_id": sample_dir.name,
            "ground_truth": ground_truth,
            "prediction": prediction,
        })

    # Compute metrics using generator's method
    generator_cls = GENERATORS[task_name]
    metrics = generator_cls.compute_metrics(results)

    # Collect incorrect sample IDs
    incorrect_samples = [
        r["sample_id"] for r in results
        if r["prediction"] != r["ground_truth"]
    ]

    report = {
        "task": task_name,
        "run_name": run_name,
        **metrics,
        "incorrect_samples": incorrect_samples,
    }

    # Save report
    report_path = task_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def print_report(report: dict):
    """Print a formatted report."""
    if "error" in report:
        print(f"  Error: {report['error']}")
        return

    # Print all metrics (skip metadata fields and long lists)
    skip_keys = {"task", "run_name", "incorrect_samples"}
    for key, value in report.items():
        if key in skip_keys:
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth")
    parser.add_argument(
        "--run",
        required=True,
        help="Run identifier to evaluate",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data)",
    )

    args = parser.parse_args()

    # Load task name from metadata for display
    metadata = load_run_metadata(args.data_dir, args.run)
    task_name = metadata["task"]

    print(f"\n=== {task_name} (run: {args.run}) ===")
    report = evaluate_task(args.run, args.data_dir)
    print_report(report)


if __name__ == "__main__":
    main()
