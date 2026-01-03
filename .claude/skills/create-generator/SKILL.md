---
name: create-generator
description: Create a new image generator for visual perception testing. Use when the user wants to add a new generator, create a new visual task, or implement a new perceptual test.
---

# Creating a New Generator

This skill guides you through creating a new image generator for testing Claude's visual perception capabilities.

## Overview

Generators create synthetic test images with known ground truth answers. Each generator:
- Extends `BaseGenerator` from `src/generators/base.py`
- Defines a pydantic output model for structured responses
- Defines custom parameters via `ParamSpec` for controlling generation
- Saves images with metadata for evaluation

## Step-by-Step Process

### Step 1: Define the Output Model

Create a pydantic model that represents the expected answer format. If answers are categorical, use a `str, Enum` for type safety.

```python
from enum import Enum
from pydantic import BaseModel

# For categorical answers, define an enum
class Color(str, Enum):
    RED = "red"
    BLUE = "blue"

class MyTaskOutput(BaseModel):
    answer: Color  # or use primitive types: int, str, float, bool
```

### Step 2: Define the Generator Class

```python
from .base import BaseGenerator, ParamSpec

class MyTaskGenerator(BaseGenerator):
    """Docstring explaining what this generator tests."""

    task_name = "my_task"  # Used for registration
    output_model = MyTaskOutput  # The pydantic model from Step 1
```

### Step 3: Define Parameters with `get_param_specs`

Define what parameters control the generation. These become CLI arguments via `--param key=value`. Prefer single exact values over min/max ranges.

```python
@classmethod
def get_param_specs(cls) -> list[ParamSpec]:
    return [
        ParamSpec(
            name="element_size",
            param_type=int,
            default=30,
            help="Size of elements in pixels",
            min_value=1,
            max_value=100,
        ),
        ParamSpec(
            name="num_elements",
            param_type=int,
            default=5,
            help="Number of elements to generate",
            min_value=1,
            max_value=20,
        ),
    ]
```

### Step 4: Implement `__init__`

```python
def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
    super().__init__(output_dir, run_name, seed)
    if seed is not None:
        random.seed(seed)
```

### Step 5: Implement `generate_one`

```python
def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
    """Generate a single test sample."""
    # 1. Extract params with defaults
    element_size = params.get("element_size", 30)
    num_elements = params.get("num_elements", 5)

    # 2. Create the image
    img = Image.new("RGB", self.image_size, "white")
    draw = ImageDraw.Draw(img)

    # 3. Draw your visual elements based on params
    # ...

    # 4. Determine ground truth
    ground_truth = MyTaskOutput(answer=correct_answer)

    # 5. Create the prompt (can be static or dynamic per sample)
    prompt = "Your question here?"  # or f"Dynamic question about {variable}?"

    # 6. Save and return paths
    generation_params = {
        # Store generation parameters for reproducibility/analysis
        "element_size": element_size,
        "num_elements": num_elements,
    }
    return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)
```

### Step 6: Register the Generator

Add to `src/generators/__init__.py`:

```python
from .my_task import MyTaskGenerator

GENERATORS = {
    # ... existing generators ...
    "my_task": MyTaskGenerator,
}
```

## Key Design Patterns

### Parameter Design

Use `ParamSpec` to define parameters that control task difficulty or variation. Prefer single exact values over min/max ranges to limit variability per run:

| Task | Parameters |
|------|------------|
| Bar comparison | `height_diff` (0.02-0.50) |
| Grid lookup | `rows`, `cols` (5-20) |
| Line chart | `num_lines` (2-6), `gap` (5-60 pixels) |
| Scatter plots | `num_points` (10-150) |
| Flowchart | `num_nodes` (3-20) |

### Randomization

- Always randomize what you can (positions, colors, which item is correct)
- Avoid patterns where certain answers are always correct
- Use `random.shuffle()` to vary ordering

### Prompts

- **Static prompts**: Same question for all samples (e.g., "Which bar is taller?")
- **Dynamic prompts**: Question varies per sample (e.g., "What is the value in cell B2?")

Both are stored in `metadata.json` per sample.

## File Structure

After generation, each run has:
```
data/
└── my_run_name/
    ├── task_metadata.json    # Contains {"task": "my_task"}
    ├── 0000/
    │   ├── image.png
    │   └── metadata.json
    ├── 0001/
    │   └── ...
    └── ...
```

`metadata.json` contains:
```json
{
  "prompt": "The question to ask Claude",
  "ground_truth": {"answer": "value"},
  "run_name": "my_run_name",
  "params": {...}
}
```

## Complete Example: Bar Height Generator

```python
"""Generator for bar height comparison tasks."""

import random
from enum import Enum
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec


class Color(str, Enum):
    RED = "red"
    BLUE = "blue"


class BarHeightOutput(BaseModel):
    taller: Color


class BarHeightGenerator(BaseGenerator):
    """Generate images to test bar height comparison."""

    task_name = "bar_height"
    output_model = BarHeightOutput

    @classmethod
    def get_param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="height_diff",
                param_type=float,
                default=0.08,
                help="Height difference ratio between bars (0.0-1.0)",
                min_value=0.01,
                max_value=0.5,
            ),
        ]

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        super().__init__(output_dir, run_name, seed)
        if seed is not None:
            random.seed(seed)

    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate a bar height comparison image."""
        height_diff = params.get("height_diff", 0.08)

        img = Image.new("RGB", self.image_size, "white")
        draw = ImageDraw.Draw(img)
        width, height = self.image_size

        # Bar dimensions
        bar_width = width // 5
        max_bar_height = height - 80

        # Determine heights based on params
        base_height_ratio = random.uniform(0.4, 0.9 - height_diff)

        # Randomly decide which bar is taller
        red_is_taller = random.choice([True, False])
        if red_is_taller:
            red_height = int(max_bar_height * (base_height_ratio + height_diff))
            blue_height = int(max_bar_height * base_height_ratio)
            taller = Color.RED
        else:
            blue_height = int(max_bar_height * (base_height_ratio + height_diff))
            red_height = int(max_bar_height * base_height_ratio)
            taller = Color.BLUE

        # Draw bars (randomize order)
        # ... drawing code ...

        prompt = "Which bar is taller, red or blue?"
        ground_truth = BarHeightOutput(taller=taller)
        generation_params = {
            "height_diff": height_diff,
            "red_height": red_height,
            "blue_height": blue_height,
        }

        return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)
```

## Optional: Custom Metrics

By default, generators use accuracy (exact match) as the evaluation metric. For tasks like counting where partial credit makes sense, you can override `compute_metrics`:

```python
@classmethod
def compute_metrics(cls, results: list[dict]) -> dict:
    """Compute accuracy and custom error metrics."""
    # Always include base accuracy metrics
    metrics = cls._compute_accuracy(results)

    if not results:
        metrics["mean_absolute_error"] = 0.0
        metrics["mean_percentage_error"] = 0.0
        return metrics

    absolute_errors = []
    percentage_errors = []

    for r in results:
        gt = r["ground_truth"]
        pred = r["prediction"]

        # Handle both dict and raw value formats
        gt_count = gt["count"] if isinstance(gt, dict) else gt
        pred_count = pred["count"] if isinstance(pred, dict) else pred

        # Absolute error
        abs_error = abs(gt_count - pred_count)
        absolute_errors.append(abs_error)

        # Percentage error (avoid division by zero)
        if gt_count > 0:
            pct_error = abs_error / gt_count
        else:
            pct_error = 0.0 if pred_count == 0 else 1.0
        percentage_errors.append(pct_error)

    metrics["mean_absolute_error"] = sum(absolute_errors) / len(absolute_errors)
    metrics["mean_percentage_error"] = sum(percentage_errors) / len(percentage_errors)

    return metrics
```

Key points:
- Always call `cls._compute_accuracy(results)` to include standard accuracy metrics
- Each result dict has `ground_truth` and `prediction` keys
- Handle both dict format (`{"count": 5}`) and raw values (`5`)
- All returned metrics are automatically displayed by the evaluation script

## Testing Your Generator

```bash
# List available parameters
uv run python -m scripts.generate_data my_task --run x --list-params

# Generate samples with default params
uv run python -m scripts.generate_data my_task --run test_run -n 10

# Generate with custom params
uv run python -m scripts.generate_data my_task --run hard_test -n 10 \
    --param height_diff=0.02

# Run inference (task is loaded from metadata)
uv run python -m scripts.run_inference --run test_run

# Run inference with extended thinking
uv run python -m scripts.run_inference --run test_run --thinking-budget 1024

# Evaluate results
uv run python -m scripts.evaluate --run test_run
```

## Common Imports

```python
import random
from enum import Enum
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec
```
