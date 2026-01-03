"""Generator for line chart comparison tasks."""

import random
from enum import Enum
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec


class LineColor(str, Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    ORANGE = "orange"
    PURPLE = "purple"
    CYAN = "cyan"


# Map enum to actual RGB colors for drawing
LINE_COLORS = {
    LineColor.RED: "#E53935",
    LineColor.BLUE: "#1E88E5",
    LineColor.GREEN: "#43A047",
    LineColor.ORANGE: "#FB8C00",
    LineColor.PURPLE: "#8E24AA",
    LineColor.CYAN: "#00ACC1",
}


class LineChartOutput(BaseModel):
    highest_line: LineColor


class LineChartGenerator(BaseGenerator):
    """Generate line chart images to test line comparison at a point."""

    task_name = "line_chart"
    output_model = LineChartOutput

    @classmethod
    def get_param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="num_lines",
                param_type=int,
                default=4,
                help="Number of lines in the chart",
                min_value=2,
                max_value=6,
            ),
            ParamSpec(
                name="gap",
                param_type=int,
                default=20,
                help="Pixel gap between lines",
                min_value=5,
                max_value=60,
            ),
        ]

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        super().__init__(output_dir, run_name, seed)
        if seed is not None:
            random.seed(seed)

    def _generate_line(self, num_points: int, y_range: tuple[int, int]) -> list[float]:
        """Generate a smooth-ish line using random walk with momentum."""
        min_y, max_y = y_range
        mid_y = (min_y + max_y) / 2
        range_size = max_y - min_y

        # Start somewhere in the middle area
        y = mid_y + random.uniform(-range_size * 0.3, range_size * 0.3)
        velocity = 0

        points = []
        for _ in range(num_points):
            points.append(y)
            # Update velocity with some randomness and mean reversion
            velocity = velocity * 0.7 + random.uniform(-5, 5)
            # Add mean reversion to keep line in bounds
            velocity += (mid_y - y) * 0.05
            y += velocity
            # Soft clamp
            y = max(min_y, min(max_y, y))

        return points

    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate a line chart comparison image."""
        num_lines = params.get("num_lines", 4)
        gap = params.get("gap", 20)

        # Image dimensions
        width, height = 500, 400
        margin_left = 50
        margin_right = 30
        margin_top = 30
        margin_bottom = 50

        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        # Number of x points (0 to 50 inclusive = 51 points)
        num_points = 51
        x_max = 50

        # Generate lines with controlled spacing
        colors = list(LineColor)[:num_lines]
        lines_data = []

        # Generate base line
        base_line = self._generate_line(num_points, (50, plot_height - 50))

        # Generate offsets and shuffle them so any color can be on top
        offsets = [gap * (i - num_lines / 2) for i in range(num_lines)]
        random.shuffle(offsets)

        for color, offset in zip(colors, offsets):
            line = [y + offset for y in base_line]
            # Add individual variation
            line = [y + random.uniform(-10, 10) for y in line]
            lines_data.append((color, line))

        # Pick a random x position to ask about
        target_x = random.randint(5, 45)  # Avoid edges

        # Find which line is highest at target_x
        highest_value = float("-inf")
        highest_line = None
        for color, line in lines_data:
            if line[target_x] > highest_value:
                highest_value = line[target_x]
                highest_line = color

        # Draw axes
        # X-axis
        draw.line(
            [(margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom)],
            fill="black",
            width=2,
        )
        # Y-axis
        draw.line(
            [(margin_left, margin_top), (margin_left, height - margin_bottom)],
            fill="black",
            width=2,
        )

        # X-axis labels
        for x_val in range(0, x_max + 1, 10):
            x_pos = margin_left + (x_val / x_max) * plot_width
            draw.line(
                [(x_pos, height - margin_bottom), (x_pos, height - margin_bottom + 5)],
                fill="black",
                width=1,
            )
            draw.text((x_pos - 8, height - margin_bottom + 10), str(x_val), fill="black")

        # Draw vertical line at target x
        target_x_pos = margin_left + (target_x / x_max) * plot_width
        draw.line(
            [(target_x_pos, margin_top), (target_x_pos, height - margin_bottom)],
            fill="#CCCCCC",
            width=1,
        )

        # Draw lines
        for color, line in lines_data:
            points = []
            for x_idx, y_val in enumerate(line):
                x_pos = margin_left + (x_idx / x_max) * plot_width
                # Flip y since image coordinates are top-down
                y_pos = height - margin_bottom - y_val
                points.append((x_pos, y_pos))

            # Draw as connected line segments
            draw.line(points, fill=LINE_COLORS[color], width=2)

        # Draw legend
        legend_x = width - margin_right - 80
        legend_y = margin_top + 10
        for i, (color, _) in enumerate(lines_data):
            y = legend_y + i * 20
            draw.line([(legend_x, y + 6), (legend_x + 20, y + 6)], fill=LINE_COLORS[color], width=2)
            draw.text((legend_x + 25, y), color.value, fill="black")

        # Save
        prompt = f"Which line is the highest at x={target_x}?"
        ground_truth = LineChartOutput(highest_line=highest_line)
        generation_params = {
            "num_lines": num_lines,
            "gap": gap,
            "target_x": target_x,
            "lines": {color.value: line for color, line in lines_data},
        }

        return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)
