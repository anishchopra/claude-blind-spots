#!/usr/bin/env python3
"""Generate test data for visual perception tasks.

Usage:
    python -m scripts.generate_data bar_height --run exp_01 -n 10
    python -m scripts.generate_data bar_height --run exp_01 -n 10 --param min_height_diff=0.05
    python -m scripts.generate_data bar_height --run x --list-params
"""

import argparse
from pathlib import Path

from src.generators import GENERATORS
from src.generators.base import ParamSpec


def parse_param(param_str: str, specs: list[ParamSpec]) -> tuple[str, any]:
    """Parse a key=value parameter string."""
    if "=" not in param_str:
        raise ValueError(f"Parameter must be in key=value format: {param_str}")

    key, value = param_str.split("=", 1)

    # Find the spec for this key
    spec = next((s for s in specs if s.name == key), None)
    if spec is None:
        valid_keys = [s.name for s in specs]
        raise ValueError(f"Unknown parameter '{key}'. Valid: {valid_keys}")

    # Convert to appropriate type
    if spec.param_type == int:
        typed_value = int(value)
    elif spec.param_type == float:
        typed_value = float(value)
    elif spec.param_type == bool:
        typed_value = value.lower() in ("true", "1", "yes")
    else:
        typed_value = value

    # Validate bounds
    if spec.min_value is not None and typed_value < spec.min_value:
        raise ValueError(f"{key} must be >= {spec.min_value}")
    if spec.max_value is not None and typed_value > spec.max_value:
        raise ValueError(f"{key} must be <= {spec.max_value}")

    return key, typed_value


def main():
    parser = argparse.ArgumentParser(
        description="Generate test images for visual perception tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "task",
        choices=list(GENERATORS.keys()),
        help="Task type to generate",
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Run identifier (required). Used in output path: data/{run}/",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of samples to generate (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Base output directory (default: data)",
    )
    parser.add_argument(
        "--param",
        action="append",
        dest="params",
        default=[],
        metavar="KEY=VALUE",
        help="Generator-specific parameter (can be repeated)",
    )
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="List available parameters for the specified task and exit",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing samples (default: skip)",
    )

    args = parser.parse_args()

    generator_cls = GENERATORS[args.task]
    param_specs = generator_cls.get_param_specs()

    # Handle --list-params
    if args.list_params:
        print(f"Parameters for '{args.task}':")
        if not param_specs:
            print("  (no custom parameters)")
        for spec in param_specs:
            bounds = ""
            if spec.min_value is not None or spec.max_value is not None:
                bounds = f" [{spec.min_value}-{spec.max_value}]"
            print(f"  {spec.name}: {spec.param_type.__name__}{bounds}")
            print(f"      Default: {spec.default}")
            print(f"      {spec.help}")
        return

    # Parse custom params
    custom_params = {}
    for param_str in args.params:
        key, value = parse_param(param_str, param_specs)
        custom_params[key] = value

    # Fill in defaults for unspecified params
    final_params = {}
    for spec in param_specs:
        final_params[spec.name] = custom_params.get(spec.name, spec.default)

    # Create generator
    generator = generator_cls(
        output_dir=args.output_dir,
        run_name=args.run,
        seed=args.seed,
    )

    print(f"=== {args.task} (run: {args.run}) ===")
    print(f"Parameters: {final_params}")
    print(f"Generating {args.n} samples...")

    generated = 0
    skipped = 0

    for i in range(args.n):
        sample_id = f"{i:04d}"
        sample_dir = generator.output_dir / sample_id

        if sample_dir.exists() and sample_dir.is_dir() and not args.overwrite:
            skipped += 1
            continue

        generator.generate_one(sample_id, **final_params)
        generated += 1

    print(f"Done! Generated {generated} samples, skipped {skipped} existing")


if __name__ == "__main__":
    main()
