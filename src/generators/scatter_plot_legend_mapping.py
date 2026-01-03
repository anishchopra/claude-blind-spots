"""Generator for scatter plot legend mapping tasks - identifying clusters by their legend labels."""

import random
import string
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec


# Available shapes (supports up to 6 clusters)
SHAPES = ["circle", "triangle", "square", "diamond", "star", "cross"]

# Colors for shapes (all use the same color to focus on shape distinction)
SHAPE_COLOR = "#2563EB"  # Blue


class ScatterPlotLegendMappingOutput(BaseModel):
    answer: str


class ScatterPlotLegendMappingGenerator(BaseGenerator):
    """Generate scatter plots to test legend-to-cluster mapping ability.

    Tests whether the model can correctly identify which cluster corresponds
    to which legend label based on shape, and determine which is highest.
    """

    task_name = "scatter_plot_legend_mapping"
    output_model = ScatterPlotLegendMappingOutput

    @classmethod
    def get_param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="num_clusters",
                param_type=int,
                default=4,
                help="Number of clusters/categories (2-6)",
                min_value=2,
                max_value=6,
            ),
            ParamSpec(
                name="points_per_cluster",
                param_type=int,
                default=15,
                help="Number of points per cluster",
                min_value=5,
                max_value=30,
            ),
        ]

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        super().__init__(output_dir, run_name, seed)
        if seed is not None:
            random.seed(seed)

    def _draw_shape(self, draw: ImageDraw, shape: str, x: int, y: int, size: int = 8):
        """Draw a shape at the given position."""
        if shape == "circle":
            draw.ellipse(
                [x - size, y - size, x + size, y + size],
                fill=SHAPE_COLOR,
                outline="black",
                width=1,
            )
        elif shape == "triangle":
            # Equilateral triangle pointing up
            points = [
                (x, y - size),  # Top
                (x - size, y + size),  # Bottom left
                (x + size, y + size),  # Bottom right
            ]
            draw.polygon(points, fill=SHAPE_COLOR, outline="black", width=1)
        elif shape == "square":
            draw.rectangle(
                [x - size, y - size, x + size, y + size],
                fill=SHAPE_COLOR,
                outline="black",
                width=1,
            )
        elif shape == "diamond":
            # Diamond (rotated square)
            points = [
                (x, y - size),  # Top
                (x + size, y),  # Right
                (x, y + size),  # Bottom
                (x - size, y),  # Left
            ]
            draw.polygon(points, fill=SHAPE_COLOR, outline="black", width=1)
        elif shape == "star":
            # 5-pointed star
            import math
            points = []
            for i in range(10):
                angle = math.pi / 2 + i * math.pi / 5
                r = size if i % 2 == 0 else size * 0.4
                points.append((x + r * math.cos(angle), y - r * math.sin(angle)))
            draw.polygon(points, fill=SHAPE_COLOR, outline="black", width=1)
        elif shape == "cross":
            # Plus/cross shape
            arm = size * 0.4
            draw.rectangle([x - arm, y - size, x + arm, y + size], fill=SHAPE_COLOR, outline="black", width=1)
            draw.rectangle([x - size, y - arm, x + size, y + arm], fill=SHAPE_COLOR, outline="black", width=1)

    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate a scatter plot legend mapping test image."""
        num_clusters = params.get("num_clusters", 4)
        points_per_cluster = params.get("points_per_cluster", 15)

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

        # Generate labels (A, B, C, ...) based on num_clusters
        labels = list(string.ascii_uppercase[:num_clusters])

        # Get shapes for these clusters
        shapes = SHAPES[:num_clusters]
        random.shuffle(shapes)
        label_to_shape = dict(zip(labels, shapes))

        # Generate cluster centers with clear vertical separation
        # Y values in chart coordinates (0-100), higher = higher on chart
        # One cluster should be clearly highest (at 85), others spread below

        # Create well-separated y-levels based on num_clusters
        # Highest is always 85, others distributed in 10-60 range
        other_y_levels = []
        spacing = 50 // (num_clusters - 1) if num_clusters > 1 else 0
        for i in range(num_clusters - 1):
            other_y_levels.append(10 + i * spacing)
        y_levels = other_y_levels + [85]  # 85 is clearly highest
        random.shuffle(y_levels)

        # Randomly assign y-levels to labels
        label_to_y_center = dict(zip(labels, y_levels))

        # Find which label has the highest cluster
        highest_label = max(labels, key=lambda l: label_to_y_center[l])

        # X centers spread across the plot
        x_spacing = 80 // num_clusters
        x_centers = [10 + x_spacing // 2 + i * x_spacing for i in range(num_clusters)]
        random.shuffle(x_centers)
        label_to_x_center = dict(zip(labels, x_centers))

        # Generate cluster points
        clusters = {}
        for label in labels:
            x_center = label_to_x_center[label]
            y_center = label_to_y_center[label]

            points = []
            for _ in range(points_per_cluster):
                # Add some spread around the center
                x = x_center + random.gauss(0, 5)
                y = y_center + random.gauss(0, 5)
                # Clamp to valid range
                x = max(5, min(95, x))
                y = max(5, min(95, y))
                points.append((x, y))
            clusters[label] = points

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
        for x_val in range(0, 101, 20):
            x_pos = margin_left + (x_val / 100) * plot_width
            draw.line(
                [(x_pos, height - margin_bottom), (x_pos, height - margin_bottom + 5)],
                fill="black",
                width=1,
            )
            draw.text((x_pos - 8, height - margin_bottom + 10), str(x_val), fill="black")

        # Y-axis labels
        for y_val in range(0, 101, 20):
            y_pixel = height - margin_bottom - (y_val / 100) * plot_height
            draw.line(
                [(margin_left - 5, y_pixel), (margin_left, y_pixel)],
                fill="black",
                width=1,
            )
            draw.text((margin_left - 30, y_pixel - 6), str(y_val), fill="black")

        # Draw points for each cluster
        for label in labels:
            shape = label_to_shape[label]
            for x_chart, y_chart in clusters[label]:
                # Convert chart coordinates to pixel coordinates
                x_pixel = margin_left + (x_chart / 100) * plot_width
                y_pixel = height - margin_bottom - (y_chart / 100) * plot_height
                self._draw_shape(draw, shape, int(x_pixel), int(y_pixel))

        # Draw legend in top right corner (alphabetical order A, B, C, D)
        legend_x = width - margin_right + 10
        legend_y = margin_top + 10

        for i, label in enumerate(labels):
            y = legend_y + i * 25
            shape = label_to_shape[label]
            # Draw shape in legend
            self._draw_shape(draw, shape, legend_x + 8, int(y + 8), size=6)
            # Draw label
            draw.text((legend_x + 22, y), label, fill="black")

        # Save
        # Build prompt dynamically based on number of clusters
        label_list = ", ".join(labels[:-1]) + f", or {labels[-1]}" if num_clusters > 2 else " or ".join(labels)
        prompt = f"Which category ({label_list}) has the highest cluster of points?"
        ground_truth = ScatterPlotLegendMappingOutput(answer=highest_label)

        generation_params = {
            "num_clusters": num_clusters,
            "points_per_cluster": points_per_cluster,
            "highest_label": highest_label,
            "label_to_shape": label_to_shape,
            "label_to_y_center": label_to_y_center,
            "label_to_x_center": label_to_x_center,
        }

        return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)
