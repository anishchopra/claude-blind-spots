"""Generator for pie chart slice comparison tasks."""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec


class PieChartOutput(BaseModel):
    biggest_slice: str  # The label of the biggest slice (A, B, C, etc.)


class PieChartGenerator(BaseGenerator):
    """Generate pie chart images to test identification of the largest slice."""

    task_name = "pie_chart"
    output_model = PieChartOutput

    # Distinct colors for pie slices
    SLICE_COLORS = [
        "#E74C3C",  # Red
        "#3498DB",  # Blue
        "#2ECC71",  # Green
        "#F39C12",  # Orange
        "#9B59B6",  # Purple
        "#1ABC9C",  # Teal
        "#E91E63",  # Pink
        "#795548",  # Brown
    ]

    @classmethod
    def get_param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="num_slices",
                param_type=int,
                default=5,
                help="Number of pie slices",
                min_value=2,
                max_value=8,
            ),
            ParamSpec(
                name="size_advantage",
                param_type=float,
                default=0.20,
                help="Size advantage of largest slice (ratio above average)",
                min_value=0.05,
                max_value=0.6,
            ),
        ]

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        super().__init__(output_dir, run_name, seed)
        if seed is not None:
            random.seed(seed)

    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate a pie chart image with labeled slices."""
        num_slices = params.get("num_slices", 5)
        advantage = params.get("size_advantage", 0.20)

        # Create slice values: start with random base values, then boost one
        slices = self._generate_slice_values(num_slices, advantage)

        # Find the biggest slice index
        biggest_idx = slices.index(max(slices))
        biggest_label = chr(ord("A") + biggest_idx)

        # Create image
        img = Image.new("RGB", self.image_size, "white")
        draw = ImageDraw.Draw(img)

        # Draw the pie chart
        self._draw_pie_chart(draw, slices)

        # Create prompt and ground truth
        prompt = "Which slice of the pie chart is the biggest? Answer with just the letter (A, B, C, etc.)."
        ground_truth = PieChartOutput(biggest_slice=biggest_label)

        generation_params = {
            "num_slices": num_slices,
            "size_advantage": advantage,
            "slice_values": slices,
            "biggest_slice_idx": biggest_idx,
        }

        return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)

    def _generate_slice_values(self, num_slices: int, advantage: float) -> list[float]:
        """Generate slice values where one slice is larger by the given advantage.

        Args:
            num_slices: Number of slices to generate
            advantage: How much larger the biggest slice is (as ratio above average)

        Returns:
            List of slice values (as proportions that sum to 1.0)
        """
        # Start with random values for all slices
        base_values = [random.uniform(0.5, 1.5) for _ in range(num_slices)]

        # Pick a random slice to be the biggest
        biggest_idx = random.randint(0, num_slices - 1)

        # Calculate what the biggest slice needs to be
        # Average slice = 1/num_slices of the pie
        # We want biggest to be (1 + advantage) * average
        # But we need to account for the existing distribution

        # Normalize to sum to 1
        total = sum(base_values)
        normalized = [v / total for v in base_values]

        # Find current max
        current_max_idx = normalized.index(max(normalized))

        # Make our chosen slice the biggest
        if current_max_idx != biggest_idx:
            # Swap values so our chosen index becomes the biggest
            normalized[biggest_idx], normalized[current_max_idx] = (
                normalized[current_max_idx],
                normalized[biggest_idx],
            )

        # Now adjust to ensure the advantage is correct
        avg = 1.0 / num_slices
        target_biggest = avg * (1 + advantage)

        # Ensure the biggest is at least target_biggest
        current_biggest = normalized[biggest_idx]
        if current_biggest < target_biggest:
            # Need to increase the biggest slice
            deficit = target_biggest - current_biggest
            normalized[biggest_idx] = target_biggest

            # Take equally from other slices
            reduction_per_slice = deficit / (num_slices - 1)
            for i in range(num_slices):
                if i != biggest_idx:
                    normalized[i] -= reduction_per_slice

        # Make sure no slice is negative or too small
        min_slice = 0.03  # Minimum 3% for visibility
        for i in range(num_slices):
            if normalized[i] < min_slice:
                normalized[i] = min_slice

        # Re-normalize
        total = sum(normalized)
        normalized = [v / total for v in normalized]

        # Ensure biggest is still biggest after adjustments
        biggest_val = normalized[biggest_idx]
        second_biggest = max(v for i, v in enumerate(normalized) if i != biggest_idx)

        if biggest_val <= second_biggest:
            # Force it to be bigger
            diff = second_biggest - biggest_val + avg * advantage
            normalized[biggest_idx] += diff
            # Reduce from the second biggest
            second_idx = normalized.index(second_biggest)
            normalized[second_idx] -= diff

        # Final normalization
        total = sum(normalized)
        return [v / total for v in normalized]

    def _draw_pie_chart(self, draw: ImageDraw.Draw, slices: list[float]) -> None:
        """Draw a pie chart with labeled slices.

        Args:
            draw: PIL ImageDraw object
            slices: List of slice proportions (should sum to 1.0)
        """
        width, height = self.image_size
        center_x = width // 2
        center_y = height // 2
        radius = min(width, height) // 2 - 50  # Leave margin for labels

        # Bounding box for the pie
        bbox = [
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        ]

        # Draw slices
        start_angle = -90  # Start from top (12 o'clock)
        label_positions = []

        for i, proportion in enumerate(slices):
            extent = proportion * 360
            color = self.SLICE_COLORS[i % len(self.SLICE_COLORS)]

            # Draw the slice
            draw.pieslice(bbox, start_angle, start_angle + extent, fill=color, outline="white", width=2)

            # Calculate label position (middle of the arc, outside the pie)
            mid_angle = start_angle + extent / 2
            mid_angle_rad = math.radians(mid_angle)

            # Position label outside the pie
            label_radius = radius + 25
            label_x = center_x + label_radius * math.cos(mid_angle_rad)
            label_y = center_y + label_radius * math.sin(mid_angle_rad)

            label_positions.append((label_x, label_y, chr(ord("A") + i)))

            start_angle += extent

        # Draw labels
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except (OSError, IOError):
                font = ImageFont.load_default()

        for label_x, label_y, label in label_positions:
            # Draw label with background for visibility
            text_bbox = draw.textbbox((label_x, label_y), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Center the label
            text_x = label_x - text_width / 2
            text_y = label_y - text_height / 2

            # Draw text
            draw.text((text_x, text_y), label, fill="black", font=font)
