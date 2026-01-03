"""Generator for scatter plot point counting tasks within a region."""

import random
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec


class ScatterRegionCountOutput(BaseModel):
    count: int  # Number of points inside the region


class ScatterRegionCountGenerator(BaseGenerator):
    """Generate scatter plot images to test counting points within a region."""

    task_name = "scatter_region_count"
    output_model = ScatterRegionCountOutput

    POINT_RADIUS = 5
    POINT_COLOR = "#2C3E50"  # Dark blue-gray
    REGION_COLOR = (52, 152, 219, 80)  # Translucent blue (RGBA)
    REGION_BORDER_COLOR = "#2980B9"  # Darker blue for border

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
                name="min_boundary_distance",
                param_type=int,
                default=8,
                help="Minimum distance from points to region boundary (larger = easier)",
                min_value=2,
                max_value=20,
            ),
        ]

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        super().__init__(output_dir, run_name, seed)
        if seed is not None:
            random.seed(seed)

    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate a scatter plot with a rectangular counting region."""
        num_points = params.get("num_points", 40)
        min_boundary_dist = params.get("min_boundary_distance", 8)

        width, height = self.image_size
        margin = 30  # Margin from image edges

        # Define the rectangular region (random position and size)
        region = self._generate_region(width, height, margin)

        # Generate points ensuring none are on the boundary
        points, inside_count = self._generate_points(
            num_points, width, height, margin, region, min_boundary_dist
        )

        # Create base image with white background
        img = Image.new("RGBA", self.image_size, "white")
        draw = ImageDraw.Draw(img)

        # Draw the translucent region first (as overlay)
        overlay = Image.new("RGBA", self.image_size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        self._draw_region_fill(overlay_draw, region)

        # Composite the translucent region onto base
        img = Image.alpha_composite(img, overlay)

        # Now draw points on top so they're visible
        draw = ImageDraw.Draw(img)
        for x, y in points:
            self._draw_point(draw, x, y)

        # Draw region border on top of everything
        self._draw_region_border(draw, region)

        # Convert to RGB for saving
        img_rgb = img.convert("RGB")

        # Create prompt and ground truth
        prompt = "How many points are contained within the blue rectangular region? Count only points that are fully inside the region."
        ground_truth = ScatterRegionCountOutput(count=inside_count)

        generation_params = {
            "num_points": num_points,
            "inside_count": inside_count,
            "region": region,
            "min_boundary_distance": min_boundary_dist,
        }

        return self._save_sample(sample_id, img_rgb, prompt, ground_truth, generation_params)

    def _generate_region(
        self, width: int, height: int, margin: int
    ) -> tuple[int, int, int, int]:
        """Generate a random rectangular region.

        Returns:
            Tuple of (x1, y1, x2, y2) defining the region bounds.
        """
        # Region should be between 20-40% of image dimensions
        min_region_w = int(width * 0.2)
        max_region_w = int(width * 0.4)
        min_region_h = int(height * 0.2)
        max_region_h = int(height * 0.4)

        region_w = random.randint(min_region_w, max_region_w)
        region_h = random.randint(min_region_h, max_region_h)

        # Random position (ensure region fits within margins)
        x1 = random.randint(margin, width - margin - region_w)
        y1 = random.randint(margin, height - margin - region_h)
        x2 = x1 + region_w
        y2 = y1 + region_h

        return (x1, y1, x2, y2)

    def _generate_points(
        self,
        num_points: int,
        width: int,
        height: int,
        margin: int,
        region: tuple[int, int, int, int],
        min_boundary_dist: int,
    ) -> tuple[list[tuple[int, int]], int]:
        """Generate random points, ensuring none are on the region boundary.

        Args:
            num_points: Number of points to generate
            width: Image width
            height: Image height
            margin: Margin from image edges
            region: Region bounds (x1, y1, x2, y2)
            min_boundary_dist: Minimum distance from any point to region boundary

        Returns:
            Tuple of (list of points, count of points inside region)
        """
        x1, y1, x2, y2 = region
        points = []
        inside_count = 0

        # We want a mix of inside and outside points
        # Aim for roughly 20-50% of points inside
        target_inside_ratio = random.uniform(0.2, 0.5)
        target_inside = int(num_points * target_inside_ratio)

        attempts = 0
        max_attempts = num_points * 100  # Prevent infinite loops

        while len(points) < num_points and attempts < max_attempts:
            attempts += 1

            # Decide if we want this point inside or outside
            need_inside = inside_count < target_inside and (
                num_points - len(points) <= target_inside - inside_count
                or random.random() < target_inside_ratio
            )

            if need_inside:
                # Generate point clearly inside the region
                px = random.randint(
                    x1 + min_boundary_dist + self.POINT_RADIUS,
                    x2 - min_boundary_dist - self.POINT_RADIUS,
                )
                py = random.randint(
                    y1 + min_boundary_dist + self.POINT_RADIUS,
                    y2 - min_boundary_dist - self.POINT_RADIUS,
                )
            else:
                # Generate point outside the region
                px = random.randint(margin + self.POINT_RADIUS, width - margin - self.POINT_RADIUS)
                py = random.randint(margin + self.POINT_RADIUS, height - margin - self.POINT_RADIUS)

                # Check if it's in the "forbidden zone" (too close to boundary)
                if self._is_near_boundary(px, py, region, min_boundary_dist):
                    continue

                # If it ended up inside, skip
                if self._is_inside(px, py, region):
                    continue

            # Check for overlap with existing points
            if self._overlaps_existing(px, py, points):
                continue

            points.append((px, py))
            if self._is_inside(px, py, region):
                inside_count += 1

        return points, inside_count

    def _is_inside(self, px: int, py: int, region: tuple[int, int, int, int]) -> bool:
        """Check if a point is inside the region."""
        x1, y1, x2, y2 = region
        return x1 < px < x2 and y1 < py < y2

    def _is_near_boundary(
        self, px: int, py: int, region: tuple[int, int, int, int], min_dist: int
    ) -> bool:
        """Check if a point is too close to the region boundary."""
        x1, y1, x2, y2 = region

        # Distance to each edge
        dist_left = abs(px - x1)
        dist_right = abs(px - x2)
        dist_top = abs(py - y1)
        dist_bottom = abs(py - y2)

        # Check if within the boundary zone
        in_horizontal_band = y1 - min_dist <= py <= y2 + min_dist
        in_vertical_band = x1 - min_dist <= px <= x2 + min_dist

        # Near left or right edge
        if in_horizontal_band and (dist_left < min_dist or dist_right < min_dist):
            return True

        # Near top or bottom edge
        if in_vertical_band and (dist_top < min_dist or dist_bottom < min_dist):
            return True

        return False

    def _overlaps_existing(
        self, px: int, py: int, points: list[tuple[int, int]]
    ) -> bool:
        """Check if a point overlaps with existing points."""
        min_distance = self.POINT_RADIUS * 3  # Minimum spacing between points
        for ex, ey in points:
            dist = ((px - ex) ** 2 + (py - ey) ** 2) ** 0.5
            if dist < min_distance:
                return True
        return False

    def _draw_point(self, draw: ImageDraw.Draw, x: int, y: int) -> None:
        """Draw a single point."""
        draw.ellipse(
            [
                x - self.POINT_RADIUS,
                y - self.POINT_RADIUS,
                x + self.POINT_RADIUS,
                y + self.POINT_RADIUS,
            ],
            fill=self.POINT_COLOR,
        )

    def _draw_region_fill(self, draw: ImageDraw.Draw, region: tuple[int, int, int, int]) -> None:
        """Draw the translucent fill of the rectangular region."""
        x1, y1, x2, y2 = region
        draw.rectangle([x1, y1, x2, y2], fill=self.REGION_COLOR)

    def _draw_region_border(self, draw: ImageDraw.Draw, region: tuple[int, int, int, int]) -> None:
        """Draw the border of the rectangular region."""
        x1, y1, x2, y2 = region
        draw.rectangle([x1, y1, x2, y2], outline=self.REGION_BORDER_COLOR, width=2)
