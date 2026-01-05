# Claude Vision Blind Spots

Characterizing and addressing visual perception limitations in Claude Haiku 4.5.

## Project Structure

```
├── src/
│   ├── generators/         # Image generators for each task type
│   ├── inference/          # Claude inference logic
│   └── utils.py            # Shared utilities
├── scripts/                # CLI entry points
└── .claude/skills/         # Claude Code skills for this project
```

## Setup

```bash
uv sync
```

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=your_key_here
```

## Usage

### Quick Start: Full Pipeline

Run the complete generate → inference → evaluate pipeline in one command:

```bash
uv run bash scripts/run_pipeline.sh <task> <run_name> [n_samples] [--param key=value ...]

# Examples
uv run bash scripts/run_pipeline.sh bar_height exp_01 10
uv run bash scripts/run_pipeline.sh bar_height hard_test 50 --param height_diff=0.02
uv run bash scripts/run_pipeline.sh scatter_color_count colors_exp 100 --param num_points=60
```

### Individual Steps

#### Generate Test Data

```bash
# Generate samples for a task
uv run python -m scripts.generate_data <task> --run <run_name> -n <samples>
uv run python -m scripts.generate_data bar_height --run exp_01 -n 100

# List available parameters for a task
uv run python -m scripts.generate_data <task> --run x --list-params

# Generate with custom parameters
uv run python -m scripts.generate_data bar_height --run hard -n 50 --param height_diff=0.02

# With seed for reproducibility
uv run python -m scripts.generate_data bar_height --run exp_01 -n 10 --seed 42
```

Output structure:
```
data/<run_name>/
├── task_metadata.json    # {"task": "bar_height"}
├── 0000/
│   ├── image.png
│   ├── metadata.json     # prompt, ground_truth, params
│   └── prediction.json   # (added by inference)
├── 0001/
│   └── ...
└── report.json           # (added by evaluation)
└── task_metadata.json    # task-level metadata (added by generate_data)
```

#### Run Inference

```bash
# Run inference on a run (task is loaded from metadata)
uv run python -m scripts.run_inference --run <run_name>

# With extended thinking
uv run python -m scripts.run_inference --run exp_01 --thinking-budget 1024

# Use a different model
uv run python -m scripts.run_inference --run exp_01 --model claude-sonnet-4-5-20241022

# Adjust concurrency
uv run python -m scripts.run_inference --run exp_01 --max-concurrent 5
```

#### Evaluate Results

```bash
uv run python -m scripts.evaluate --run <run_name>
```

Generates `report.json` with accuracy and task-specific metrics.

## Available Tasks

| Task | Description | Output |
|------|-------------|--------|
| `bar_height` | Determine which of two bars (red/blue) is taller | `{taller: "red" \| "blue"}` |
| `bar_chart_value` | Read the value label on a colored bar | `{value: int}` |
| `flowchart` | Follow a path to the next step | `{next_step: "A" \| "B" \| ...}` |
| `grid_lookup` | Read the value in a highlighted cell | `{value: int}` |
| `grid_retrieval` | Find which cell contains a red dot | `{cell: str}` |
| `grid_value_search` | Find coordinates of a target value | `{row: int, col: int}` |
| `line_chart` | Identify which line is highest at a given x | `{highest_line: color}` |
| `line_chart_legend_mapping` | Read line value at x=0 using legend | `{value: int}` |
| `line_intersection` | Count intersections between two lines | `{count: int}` |
| `path_following` | Follow colored path from letter to number | `{answer: int}` |
| `pie_chart` | Identify the largest slice | `{biggest_slice: "A" \| "B" \| ...}` |
| `region_containment` | Is the red dot inside the blue box? | `{inside: bool}` |
| `scatter_color_count` | Count points of a specific color | `{count: int}` |
| `scatter_plot_legend_mapping` | Identify highest cluster by shape | `{answer: str}` |
| `scatter_region_count` | Count points in a rectangular region | `{count: int}` |
| `scatter_shape_count` | Count points of a specific shape | `{count: int}` |
| `swimlane` | Identify which lane a step belongs to | `{category: str}` |

## Adding New Tasks

Use the `create-generator` skill in Claude Code, or manually:

1. Create a new generator in `src/generators/` extending `BaseGenerator`
2. Define `task_name`, `output_model` (pydantic), and `get_param_specs()`
3. Implement `generate_one(sample_id, **params)`
4. Register in `src/generators/__init__.py`

See `.claude/skills/create-generator/SKILL.md` for detailed instructions.
