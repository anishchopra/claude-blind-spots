#!/usr/bin/env python3
"""Run Claude inference on generated samples.

Usage: python -m scripts.run_inference --run exp_01
"""

import argparse
import asyncio
from pathlib import Path

from src.inference.run import run_inference
from src.utils import load_run_metadata


def main():
    parser = argparse.ArgumentParser(description="Run Claude inference on generated samples")
    parser.add_argument(
        "--run",
        required=True,
        help="Run identifier to process",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-haiku-4-5-20251001",
        help="Model to use (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=100,
        help="Max concurrent API calls (default: 100)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run inference even if prediction.json exists (default: skip)",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Enable extended thinking with this token budget (default: disabled)",
    )

    args = parser.parse_args()

    # Load task from metadata
    metadata = load_run_metadata(args.data_dir, args.run)
    task_name = metadata["task"]

    print(f"\n=== {task_name} (run: {args.run}) ===")
    results = asyncio.run(
        run_inference(
            task_name=task_name,
            run_name=args.run,
            data_dir=args.data_dir,
            model=args.model,
            max_concurrent=args.max_concurrent,
            rerun=args.rerun,
            thinking_budget=args.thinking_budget,
        )
    )

    # Print total token usage
    total_input = sum(r.get("usage", {}).get("input_tokens", 0) for r in results)
    total_output = sum(r.get("usage", {}).get("output_tokens", 0) for r in results)
    print(f"\nTotal tokens: {total_input} input, {total_output} output")


if __name__ == "__main__":
    main()
