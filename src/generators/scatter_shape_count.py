"""Generator for counting points of a specific shape in a scatter plot."""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec


class ScatterShapeCountOutput(BaseModel):
    count: int  # Number of points of the target shape


class ScatterShapeCountGenerator(BaseGenerator):
    """Generate scatter plot images to test counting points of a specific shape."""

    task_name = "scatter_shape_count"
    output_model = ScatterShapeCountOutput

    POINT_SIZE = 12  # Size of shapes
    POINT_COLOR = "#2C3E50"  # Dark blue-gray - same for all shapes
    OUTLINE_COLOR = "#2C3E50"

    # Available shapes with their names
    SHAPE_NAMES = ["circle", "triangle", "square", "diamond", "star"]

    @classmethod
    def compute_metrics(cls, results: list[dict]) -> dict:
        """Compute accuracy and count-based error metrics."""
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

    @classmethod
    def get_param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="num_points",
                param_type=int,
                default=40,
                help="Number of points in the scatter plot",
                min_value=10,
                max_value=150,
            ),
            ParamSpec(
                name="num_shapes",
                param_type=int,
                default=3,
                help="Number of different shapes used",
                min_value=2,
                max_value=5,
            ),
        ]

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        super().__init__(output_dir, run_name, seed)
        if seed is not None:
            random.seed(seed)

    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate a scatter plot with different shaped points."""
        num_points = params.get("num_points", 40)
        num_shapes = params.get("num_shapes", 3)

        width, height = self.image_size
        margin = 30  # Margin from image edges

        # Select shapes for this sample
        shapes_used = self.SHAPE_NAMES[:num_shapes]

        # Generate points with random shapes
        points = self._generate_points(num_points, width, height, margin, num_shapes)

        # Pick a random target shape to count
        target_shape_idx = random.randint(0, num_shapes - 1)
        target_shape_name = shapes_used[target_shape_idx]

        # Count points of target shape
        target_count = sum(1 for _, _, shape_idx in points if shape_idx == target_shape_idx)

        # Ensure we have at least 1 point of the target shape
        if target_count == 0:
            # Replace a random point with the target shape
            random_idx = random.randint(0, len(points) - 1)
            x, y, _ = points[random_idx]
            points[random_idx] = (x, y, target_shape_idx)
            target_count = 1

        # Create image
        img = Image.new("RGB", self.image_size, "white")
        draw = ImageDraw.Draw(img)

        # Draw all points
        for x, y, shape_idx in points:
            shape_name = shapes_used[shape_idx]
            self._draw_shape(draw, x, y, shape_name)

        # Create prompt and ground truth
        prompt = f"How many {target_shape_name}s are in the image?"
        ground_truth = ScatterShapeCountOutput(count=target_count)

        # Count all shapes for params
        shape_counts = {shape: 0 for shape in shapes_used}
        for _, _, shape_idx in points:
            shape_name = shapes_used[shape_idx]
            shape_counts[shape_name] += 1

        generation_params = {
            "num_points": num_points,
            "num_shapes": num_shapes,
            "target_shape": target_shape_name,
            "target_count": target_count,
            "shape_counts": shape_counts,
        }

        return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)

    def _generate_points(
        self,
        num_points: int,
        width: int,
        height: int,
        margin: int,
        num_shapes: int,
    ) -> list[tuple[int, int, int]]:
        """Generate random points with random shapes.

        Args:
            num_points: Number of points to generate
            width: Image width
            height: Image height
            margin: Margin from image edges
            num_shapes: Number of shapes to use

        Returns:
            List of (x, y, shape_index) tuples
        """
        points = []
        attempts = 0
        max_attempts = num_points * 100

        while len(points) < num_points and attempts < max_attempts:
            attempts += 1

            px = random.randint(margin + self.POINT_SIZE, width - margin - self.POINT_SIZE)
            py = random.randint(margin + self.POINT_SIZE, height - margin - self.POINT_SIZE)

            # Check for overlap with existing points
            if self._overlaps_existing(px, py, points):
                continue

            # Assign random shape
            shape_idx = random.randint(0, num_shapes - 1)
            points.append((px, py, shape_idx))

        return points

    def _overlaps_existing(
        self, px: int, py: int, points: list[tuple[int, int, int]]
    ) -> bool:
        """Check if a point overlaps with existing points."""
        min_distance = self.POINT_SIZE * 2.5  # Minimum spacing between points
        for ex, ey, _ in points:
            dist = ((px - ex) ** 2 + (py - ey) ** 2) ** 0.5
            if dist < min_distance:
                return True
        return False

    def _draw_shape(self, draw: ImageDraw.Draw, x: int, y: int, shape: str) -> None:
        """Draw a shape at the given position."""
        size = self.POINT_SIZE
        half = size // 2

        if shape == "circle":
            draw.ellipse(
                [x - half, y - half, x + half, y + half],
                fill=self.POINT_COLOR,
                outline=self.OUTLINE_COLOR,
                width=1,
            )

        elif shape == "triangle":
            # Equilateral triangle pointing up
            points = [
                (x, y - half),  # Top
                (x - half, y + half),  # Bottom left
                (x + half, y + half),  # Bottom right
            ]
            draw.polygon(points, fill=self.POINT_COLOR, outline=self.OUTLINE_COLOR)

        elif shape == "square":
            draw.rectangle(
                [x - half, y - half, x + half, y + half],
                fill=self.POINT_COLOR,
                outline=self.OUTLINE_COLOR,
                width=1,
            )

        elif shape == "diamond":
            # Square rotated 45 degrees
            points = [
                (x, y - half),  # Top
                (x + half, y),  # Right
                (x, y + half),  # Bottom
                (x - half, y),  # Left
            ]
            draw.polygon(points, fill=self.POINT_COLOR, outline=self.OUTLINE_COLOR)

        elif shape == "star":
            # 5-pointed star
            points = []
            for i in range(10):
                angle = math.pi / 2 + i * math.pi / 5  # Start from top
                r = half if i % 2 == 0 else half * 0.4  # Alternate outer/inner
                px = x + r * math.cos(angle)
                py = y - r * math.sin(angle)
                points.append((px, py))
            draw.polygon(points, fill=self.POINT_COLOR, outline=self.OUTLINE_COLOR)
