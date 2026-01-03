"""Generator for counting line intersections."""

import random
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec


class LineIntersectionOutput(BaseModel):
    count: int  # Number of intersections between the two lines


class LineIntersectionGenerator(BaseGenerator):
    """Generate images with two lines to test counting intersections."""

    task_name = "line_intersection"
    output_model = LineIntersectionOutput

    RED_COLOR = "#E74C3C"
    BLUE_COLOR = "#3498DB"
    LINE_WIDTH = 3

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
                name="num_intersections",
                param_type=int,
                default=3,
                help="Number of intersections to generate",
                min_value=0,
                max_value=12,
            ),
            ParamSpec(
                name="num_segments",
                param_type=int,
                default=8,
                help="Number of segments per line (more segments = more potential intersections)",
                min_value=3,
                max_value=15,
            ),
        ]

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        super().__init__(output_dir, run_name, seed)
        if seed is not None:
            random.seed(seed)

    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate an image with two intersecting lines."""
        target_intersections = params.get("num_intersections", 3)
        num_segments = params.get("num_segments", 8)

        # Validate: can't have more intersections than segments
        if target_intersections > num_segments:
            raise ValueError(
                f"num_intersections ({target_intersections}) cannot exceed num_segments ({num_segments}). "
                f"Each line segment can intersect the other line at most once."
            )

        # Generate two polylines with the target number of intersections
        red_points, blue_points, actual_intersections = self._generate_lines(
            target_intersections, num_segments
        )

        # Create image
        img = Image.new("RGB", self.image_size, "white")
        draw = ImageDraw.Draw(img)

        # Draw lines
        self._draw_line(draw, red_points, self.RED_COLOR)
        self._draw_line(draw, blue_points, self.BLUE_COLOR)

        # Create prompt and ground truth
        prompt = "How many times do these lines intersect?"
        ground_truth = LineIntersectionOutput(count=actual_intersections)

        generation_params = {
            "num_intersections": target_intersections,
            "num_segments": num_segments,
            "actual_intersections": actual_intersections,
            "red_points": red_points,
            "blue_points": blue_points,
        }

        return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)

    def _generate_lines(
        self, target_intersections: int, num_segments: int
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
        """Generate two polylines with approximately the target number of intersections.

        Returns:
            Tuple of (red_points, blue_points, actual_intersection_count)
        """
        width, height = self.image_size
        margin = 30

        max_attempts = 100
        best_result = None
        best_diff = float("inf")

        for _ in range(max_attempts):
            # Generate wavy lines that go from left to right
            red_points = self._generate_wavy_line(width, height, margin, "top", num_segments)
            blue_points = self._generate_wavy_line(width, height, margin, "bottom", num_segments)

            # Count actual intersections
            intersections = self._count_intersections(red_points, blue_points)

            diff = abs(intersections - target_intersections)
            if diff < best_diff:
                best_diff = diff
                best_result = (red_points, blue_points, intersections)

            if intersections == target_intersections:
                break

        return best_result

    def _generate_wavy_line(
        self, width: int, height: int, margin: int, start_position: str, num_segments: int
    ) -> list[tuple[int, int]]:
        """Generate a wavy polyline from left to right.

        Args:
            width: Image width
            height: Image height
            margin: Margin from edges
            start_position: 'top' or 'bottom' - where the line starts
            num_segments: Number of line segments

        Returns:
            List of (x, y) points defining the polyline
        """
        points = []

        # X positions evenly distributed
        x_step = (width - 2 * margin) / num_segments

        # Starting Y position
        if start_position == "top":
            base_y = margin + random.randint(20, 80)
        else:
            base_y = height - margin - random.randint(20, 80)

        for i in range(num_segments + 1):
            x = margin + int(i * x_step)

            # Add waviness
            if i == 0:
                y = base_y
            else:
                # Random oscillation that tends to cross the middle
                amplitude = (height - 2 * margin) * 0.4
                # Use sine-like pattern with randomness
                wave = amplitude * (0.5 - random.random())
                center_y = height / 2
                # Pull towards center with some randomness
                y = center_y + wave + random.randint(-30, 30)
                y = max(margin, min(height - margin, int(y)))

            points.append((x, int(y)))

        return points

    def _count_intersections(
        self, line1: list[tuple[int, int]], line2: list[tuple[int, int]]
    ) -> int:
        """Count the number of intersections between two polylines."""
        intersections = 0

        # Check each segment of line1 against each segment of line2
        for i in range(len(line1) - 1):
            seg1 = (line1[i], line1[i + 1])
            for j in range(len(line2) - 1):
                seg2 = (line2[j], line2[j + 1])
                if self._segments_intersect(seg1, seg2):
                    intersections += 1

        return intersections

    def _segments_intersect(
        self,
        seg1: tuple[tuple[int, int], tuple[int, int]],
        seg2: tuple[tuple[int, int], tuple[int, int]],
    ) -> bool:
        """Check if two line segments intersect."""
        (x1, y1), (x2, y2) = seg1
        (x3, y3), (x4, y4) = seg2

        def ccw(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> bool:
            return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

        # Check if segments intersect (excluding endpoints)
        if ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and ccw(
            x1, y1, x2, y2, x3, y3
        ) != ccw(x1, y1, x2, y2, x4, y4):
            return True

        return False

    def _draw_line(
        self, draw: ImageDraw.Draw, points: list[tuple[int, int]], color: str
    ) -> None:
        """Draw a polyline."""
        if len(points) < 2:
            return

        # Draw line segments
        for i in range(len(points) - 1):
            draw.line(
                [points[i], points[i + 1]],
                fill=color,
                width=self.LINE_WIDTH,
            )

        # Draw small circles at each point for smoother appearance
        for point in points:
            x, y = point
            r = self.LINE_WIDTH // 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
