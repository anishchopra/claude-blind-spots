# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project to characterize visual perception limitations in Claude Haiku 4.5. Generates synthetic test images with known ground truth, runs inference, and evaluates accuracy.

## Commands

```bash
# Generate test data
uv run python -m scripts.generate_data <task> --run <run_name> -n <samples>
uv run python -m scripts.generate_data bar_height --run exp_01 -n 100

# List parameters for a task
uv run python -m scripts.generate_data <task> --run x --list-params

# Generate with custom parameters
uv run python -m scripts.generate_data bar_height --run hard -n 50 \
    --param height_diff=0.02

# Run inference (task loaded from run metadata)
uv run python -m scripts.run_inference --run <run_name>
uv run python -m scripts.run_inference --run exp_01 --thinking-budget 1024

# Evaluate results
uv run python -m scripts.evaluate --run <run_name>

# Full pipeline
bash scripts/run_pipeline.sh <task> <run_name> <n_samples>

# Add dependencies (ALWAYS use uv, never edit pyproject.toml directly)
uv add <package>
```

## Architecture

Three-stage pipeline: **Generate → Inference → Evaluate**

### Generators (`src/generators/`)

Each generator extends `BaseGenerator` and defines:
- `task_name`: unique identifier for registration
- `output_model`: Pydantic model for structured output schema
- `get_param_specs()`: returns `ParamSpec` list defining controllable parameters
- `generate_one(sample_id, **params)`: creates image and saves via `_save_sample()`

Register new generators in `src/generators/__init__.py`.

### Data Structure

```
data/{run_name}/
├── task_metadata.json    # {"task": "bar_height"}
├── 0000/
│   ├── image.png
│   └── metadata.json     # prompt, ground_truth, params
├── 0001/
└── ...
```

After inference: `prediction.json` added to each sample directory.
After evaluation: `report.json` added to run directory.

### Inference (`src/inference/run.py`)

- Async with configurable concurrency (`--max-concurrent`)
- Structured outputs via Pydantic models
- Optional extended thinking (`--thinking-budget`)
- Default model: claude-haiku-4-5-20251001

### Utilities (`src/utils.py`)

`load_run_metadata(data_dir, run_name)` - loads task metadata from run directory.

## Creating New Generators

Use the `/create-generator` skill or follow the pattern:

1. Define Pydantic output model (use `str, Enum` for categorical answers)
2. Create generator class with `task_name`, `output_model`, `get_param_specs()`
3. Implement `__init__(output_dir, run_name, seed)` and `generate_one(sample_id, **params)`
4. Register in `src/generators/__init__.py`

See `src/generators/bar_height.py` for a complete example.
