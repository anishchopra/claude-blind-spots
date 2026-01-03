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
        gap = width // 10
        max_bar_height = height - 80  # Leave margin top and bottom
        base_y = height - 40  # Bottom margin

        # Calculate bar positions (centered)
        total_width = 2 * bar_width + gap
        start_x = (width - total_width) // 2

        # Base height between 40-90% of max
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

        # Randomly decide bar order (red-blue or blue-red)
        red_first = random.choice([True, False])

        if red_first:
            bars = [
                (start_x, red_height, "#D94A4A"),  # Red
                (start_x + bar_width + gap, blue_height, "#4A90D9"),  # Blue
            ]
        else:
            bars = [
                (start_x, blue_height, "#4A90D9"),  # Blue
                (start_x + bar_width + gap, red_height, "#D94A4A"),  # Red
            ]

        # Draw bars
        for x, bar_h, color in bars:
            draw.rectangle(
                [x, base_y - bar_h, x + bar_width, base_y],
                fill=color,
                outline="black",
                width=2,
            )

        # Save
        prompt = "Which bar is taller, red or blue?"
        ground_truth = BarHeightOutput(taller=taller)
        generation_params = {
            "height_diff": height_diff,
            "red_height": red_height,
            "blue_height": blue_height,
            "red_first": red_first,
        }

        return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)
