"""Generator for legend mapping tasks - identifying lines by their legend labels."""

import random
from enum import Enum
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec


class LineLabel(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


# Distinct colors for different_colors=True
DISTINCT_COLORS = {
    LineLabel.A: "#E53935",  # Red
    LineLabel.B: "#1E88E5",  # Blue
    LineLabel.C: "#FB8C00",  # Orange
    LineLabel.D: "#43A047",  # Green
}

# Base colors for generating shades
BASE_COLORS = {
    "red": [(255, 100, 100), (220, 60, 60), (180, 30, 30), (140, 0, 0)],
    "blue": [(100, 150, 255), (60, 120, 220), (30, 90, 180), (0, 60, 140)],
    "orange": [(255, 200, 100), (255, 165, 60), (220, 130, 30), (180, 100, 0)],
    "green": [(100, 200, 100), (60, 170, 60), (30, 140, 30), (0, 110, 0)],
}


class LegendMappingOutput(BaseModel):
    value: int


class LineChartLegendMappingGenerator(BaseGenerator):
    """Generate line charts to test legend-to-line mapping ability.

    Tests whether the model can correctly identify which line corresponds
    to which legend label and read its value at x=0.
    """

    task_name = "line_chart_legend_mapping"
    output_model = LegendMappingOutput

    @classmethod
    def get_param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="different_colors",
                param_type=bool,
                default=True,
                help="If True, use distinct colors (red, blue, orange, green). If False, use shades of one color.",
            ),
        ]

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        super().__init__(output_dir, run_name, seed)
        if seed is not None:
            random.seed(seed)

    def _generate_line(self, num_points: int, start_y: float, y_range: tuple[int, int]) -> list[float]:
        """Generate a smooth line starting at a specific y value."""
        min_y, max_y = y_range
        mid_y = (min_y + max_y) / 2

        y = start_y
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

    def _get_colors(self, different_colors: bool, labels: list[LineLabel]) -> dict[LineLabel, str]:
        """Get color mapping based on different_colors parameter."""
        if different_colors:
            return DISTINCT_COLORS
        else:
            # Pick a random base color and use its shades
            base_color = random.choice(list(BASE_COLORS.keys()))
            shades = BASE_COLORS[base_color]
            # Shuffle shades so the mapping isn't predictable
            shuffled_shades = shades.copy()
            random.shuffle(shuffled_shades)
            return {
                label: f"#{r:02x}{g:02x}{b:02x}"
                for label, (r, g, b) in zip(labels, shuffled_shades)
            }

    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate a legend mapping test image."""
        different_colors = params.get("different_colors", True)

        # Image dimensions
        width, height = 500, 400
        margin_left = 50
        margin_right = 100  # Extra space for legend
        margin_top = 30
        margin_bottom = 50

        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        # Number of x points
        num_points = 51
        x_max = 50

        # Labels for the 4 lines
        labels = list(LineLabel)

        # Get colors based on parameter
        colors = self._get_colors(different_colors, labels)

        # Generate starting y values with clear separation at x=0
        # Use values that will be well-separated (at least 15 chart units apart)
        y_range = (50, plot_height - 50)

        # Generate 4 well-separated starting points with variance
        min_gap = 15  # Minimum chart units (0-100 scale) between adjacent lines

        # Randomly pick 4 values that are well-separated
        # Strategy: pick a random base, then add increments with some jitter
        max_attempts = 100
        for _ in range(max_attempts):
            # Generate 4 random values
            values = sorted([random.randint(5, 95) for _ in range(4)])

            # Check if all adjacent pairs have enough separation
            valid = True
            for i in range(len(values) - 1):
                if values[i + 1] - values[i] < min_gap:
                    valid = False
                    break

            if valid:
                break
        else:
            # Fallback: evenly spaced with some jitter
            base_values = [15, 40, 65, 90]
            values = [v + random.randint(-8, 8) for v in base_values]
            values = [max(5, min(95, v)) for v in values]

        # Convert chart values (0-100) to internal y coordinates
        start_values = []
        for chart_val in values:
            internal_y = y_range[0] + (chart_val / 100) * (y_range[1] - y_range[0])
            start_values.append(internal_y)

        # Shuffle so the order isn't always A at bottom, D at top
        random.shuffle(start_values)

        # Generate lines
        lines_data = []
        for label, start_y in zip(labels, start_values):
            line = self._generate_line(num_points, start_y, y_range)
            lines_data.append((label, line))

        # Randomly pick which line to ask about
        target_label = random.choice(labels)

        # Get the value at x=0 for the target line
        target_line = next(line for label, line in lines_data if label == target_label)
        # Convert pixel y to a reasonable chart value (0-100 scale)
        # Higher pixel y = lower chart position, so invert
        raw_y = target_line[0]
        # Map from pixel range to 0-100
        chart_value = int(round((raw_y - y_range[0]) / (y_range[1] - y_range[0]) * 100))

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

        # Y-axis labels: show the 4 actual y-values at x=0
        # Store both the raw y value (for positioning) and the chart value (for label)
        line_y_data = []
        for label, line in lines_data:
            raw_y = line[0]
            chart_val = int(round((raw_y - y_range[0]) / (y_range[1] - y_range[0]) * 100))
            line_y_data.append((raw_y, chart_val))

        for raw_y, chart_val in line_y_data:
            # Position tick at exact line position (same calculation as line drawing)
            chart_y_exact = (raw_y - y_range[0]) / (y_range[1] - y_range[0]) * 100
            y_pixel = height - margin_bottom - (chart_y_exact / 100) * plot_height
            draw.line(
                [(margin_left - 5, y_pixel), (margin_left, y_pixel)],
                fill="black",
                width=1,
            )
            draw.text((margin_left - 30, y_pixel - 6), str(chart_val), fill="black")

        # Draw vertical line at x=0 to help identify values
        x0_pos = margin_left
        draw.line(
            [(x0_pos, margin_top), (x0_pos, height - margin_bottom)],
            fill="#DDDDDD",
            width=1,
        )

        # Draw lines
        for label, line in lines_data:
            points = []
            for x_idx, y_val in enumerate(line):
                x_pos = margin_left + (x_idx / x_max) * plot_width
                # Map internal y to pixel position
                # y_val is in range [y_range[0], y_range[1]]
                # Map to chart 0-100, then to pixels
                chart_y = (y_val - y_range[0]) / (y_range[1] - y_range[0]) * 100
                y_pos = height - margin_bottom - (chart_y / 100) * plot_height
                points.append((x_pos, y_pos))

            # Draw as connected line segments
            draw.line(points, fill=colors[label], width=3)

        # Draw legend in top right corner (alphabetical order A, B, C, D)
        legend_x = width - margin_right + 10
        legend_y = margin_top + 10

        legend_order = list(labels)  # Already in order: A, B, C, D

        for i, label in enumerate(legend_order):
            y = legend_y + i * 25
            draw.line([(legend_x, y + 6), (legend_x + 25, y + 6)], fill=colors[label], width=3)
            draw.text((legend_x + 30, y), label.value, fill="black")

        # Save
        prompt = f"What is the value of line {target_label.value} at x=0? Choose from one of the following options: {', '.join([str(y[1]) for y in line_y_data])}"
        ground_truth = LegendMappingOutput(value=chart_value)

        # Determine base color if using shades
        base_color_used = None
        if not different_colors:
            # Find which base color was used by checking the colors
            for base_name, shades in BASE_COLORS.items():
                shade_hexes = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in shades]
                if any(c in shade_hexes for c in colors.values()):
                    base_color_used = base_name
                    break

        generation_params = {
            "different_colors": different_colors,
            "base_color": base_color_used,
            "target_label": target_label.value,
            "target_value": chart_value,
            "legend_order": [l.value for l in legend_order],
            "line_values_at_x0": {
                label.value: int(round((line[0] - y_range[0]) / (y_range[1] - y_range[0]) * 100))
                for label, line in lines_data
            },
        }

        return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)
