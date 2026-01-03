"""Generator for counting points of a specific color in a scatter plot."""

import random
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec


class ScatterColorCountOutput(BaseModel):
    count: int  # Number of points of the target color


class ScatterColorCountGenerator(BaseGenerator):
    """Generate scatter plot images to test counting points of a specific color."""

    task_name = "scatter_color_count"
    output_model = ScatterColorCountOutput

    POINT_RADIUS = 6

    # Color palette with distinct, nameable colors
    # Order matters - first N colors are used based on difficulty
    COLOR_PALETTE = [
        ("#E74C3C", "red"),
        ("#3498DB", "blue"),
        ("#2ECC71", "green"),
        ("#F39C12", "orange"),
        ("#9B59B6", "purple"),
        ("#1ABC9C", "teal"),
    ]

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
                name="num_colors",
                param_type=int,
                default=4,
                help="Number of different colors used",
                min_value=2,
                max_value=6,
            ),
        ]

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        super().__init__(output_dir, run_name, seed)
        if seed is not None:
            random.seed(seed)

    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate a scatter plot with colored points."""
        num_points = params.get("num_points", 40)
        num_colors = params.get("num_colors", 4)

        width, height = self.image_size
        margin = 30  # Margin from image edges

        # Select colors for this sample
        colors_used = self.COLOR_PALETTE[:num_colors]

        # Generate points with random colors
        points = self._generate_points(num_points, width, height, margin, num_colors)

        # Pick a random target color to count
        target_color_idx = random.randint(0, num_colors - 1)
        target_color_hex, target_color_name = colors_used[target_color_idx]

        # Count points of target color
        target_count = sum(1 for _, _, color_idx in points if color_idx == target_color_idx)

        # Ensure we have at least 1 point of the target color
        if target_count == 0:
            # Replace a random point with the target color
            random_idx = random.randint(0, len(points) - 1)
            x, y, _ = points[random_idx]
            points[random_idx] = (x, y, target_color_idx)
            target_count = 1

        # Create image
        img = Image.new("RGB", self.image_size, "white")
        draw = ImageDraw.Draw(img)

        # Draw all points
        for x, y, color_idx in points:
            color_hex, _ = colors_used[color_idx]
            self._draw_point(draw, x, y, color_hex)

        # Create prompt and ground truth
        prompt = f"How many {target_color_name} points are in the image?"
        ground_truth = ScatterColorCountOutput(count=target_count)

        # Count all colors for params
        color_counts = {}
        for _, color_name in colors_used:
            color_counts[color_name] = 0
        for _, _, color_idx in points:
            _, color_name = colors_used[color_idx]
            color_counts[color_name] += 1

        generation_params = {
            "num_points": num_points,
            "num_colors": num_colors,
            "target_color": target_color_name,
            "target_count": target_count,
            "color_counts": color_counts,
        }

        return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)

    def _generate_points(
        self,
        num_points: int,
        width: int,
        height: int,
        margin: int,
        num_colors: int,
    ) -> list[tuple[int, int, int]]:
        """Generate random points with random colors.

        Args:
            num_points: Number of points to generate
            width: Image width
            height: Image height
            margin: Margin from image edges
            num_colors: Number of colors to use

        Returns:
            List of (x, y, color_index) tuples
        """
        points = []
        attempts = 0
        max_attempts = num_points * 100

        while len(points) < num_points and attempts < max_attempts:
            attempts += 1

            px = random.randint(margin + self.POINT_RADIUS, width - margin - self.POINT_RADIUS)
            py = random.randint(margin + self.POINT_RADIUS, height - margin - self.POINT_RADIUS)

            # Check for overlap with existing points
            if self._overlaps_existing(px, py, points):
                continue

            # Assign random color
            color_idx = random.randint(0, num_colors - 1)
            points.append((px, py, color_idx))

        return points

    def _overlaps_existing(
        self, px: int, py: int, points: list[tuple[int, int, int]]
    ) -> bool:
        """Check if a point overlaps with existing points."""
        min_distance = self.POINT_RADIUS * 2.5  # Minimum spacing between points
        for ex, ey, _ in points:
            dist = ((px - ex) ** 2 + (py - ey) ** 2) ** 0.5
            if dist < min_distance:
                return True
        return False

    def _draw_point(self, draw: ImageDraw.Draw, x: int, y: int, color: str) -> None:
        """Draw a single colored point."""
        draw.ellipse(
            [
                x - self.POINT_RADIUS,
                y - self.POINT_RADIUS,
                x + self.POINT_RADIUS,
                y + self.POINT_RADIUS,
            ],
            fill=color,
            outline="#2C3E50",
            width=1,
        )
