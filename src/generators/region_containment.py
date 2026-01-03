"""Generator for region containment tasks (is the dot inside the box?)."""

import random
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec


class RegionContainmentOutput(BaseModel):
    inside: bool  # True if the dot is inside the box


class RegionContainmentGenerator(BaseGenerator):
    """Generate images with a box and dot to test containment detection."""

    task_name = "region_containment"
    output_model = RegionContainmentOutput

    DOT_COLOR = "#E74C3C"  # Red
    DOT_RADIUS = 5
    BOX_COLOR = "#3498DB"  # Blue
    BOX_WIDTH = 150
    BOX_HEIGHT = 150

    @classmethod
    def compute_metrics(cls, results: list[dict]) -> dict:
        """Compute accuracy metrics broken down by inside/outside."""
        metrics = cls._compute_accuracy(results)

        if not results:
            return metrics

        # Separate results by ground truth (inside vs outside)
        inside_results = [r for r in results if r["ground_truth"].get("inside", r["ground_truth"]) is True]
        outside_results = [r for r in results if r["ground_truth"].get("inside", r["ground_truth"]) is False]

        # Compute accuracy for inside cases
        inside_correct = sum(
            1 for r in inside_results
            if r["prediction"].get("inside", r["prediction"]) == r["ground_truth"].get("inside", r["ground_truth"])
        )
        inside_total = len(inside_results)
        metrics["inside_correct"] = inside_correct
        metrics["inside_total"] = inside_total
        metrics["inside_accuracy"] = inside_correct / inside_total if inside_total > 0 else 0.0

        # Compute accuracy for outside cases
        outside_correct = sum(
            1 for r in outside_results
            if r["prediction"].get("inside", r["prediction"]) == r["ground_truth"].get("inside", r["ground_truth"])
        )
        outside_total = len(outside_results)
        metrics["outside_correct"] = outside_correct
        metrics["outside_total"] = outside_total
        metrics["outside_accuracy"] = outside_correct / outside_total if outside_total > 0 else 0.0

        return metrics

    @classmethod
    def get_param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="epsilon",
                param_type=int,
                default=10,
                help="Exact distance from box boundary for dot placement",
                min_value=1,
                max_value=50,
            ),
        ]

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        super().__init__(output_dir, run_name, seed)
        if seed is not None:
            random.seed(seed)

    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate an image with a box and a dot near its boundary."""
        epsilon = params.get("epsilon", 10)

        width, height = self.image_size
        margin = 50  # Margin from image edges for box placement

        # Random box position (top-left corner)
        box_x = random.randint(margin, width - margin - self.BOX_WIDTH)
        box_y = random.randint(margin, height - margin - self.BOX_HEIGHT)

        # Box bounds
        box_left = box_x
        box_right = box_x + self.BOX_WIDTH
        box_top = box_y
        box_bottom = box_y + self.BOX_HEIGHT

        # Decide if dot is inside or outside (50/50)
        is_inside = random.choice([True, False])

        # Place dot within epsilon of the boundary
        dot_x, dot_y = self._place_dot(
            box_left, box_right, box_top, box_bottom,
            epsilon, is_inside, width, height
        )

        # Create image
        img = Image.new("RGB", self.image_size, "white")
        draw = ImageDraw.Draw(img)

        # Draw box
        draw.rectangle(
            [box_left, box_top, box_right, box_bottom],
            outline=self.BOX_COLOR,
            width=3,
        )

        # Draw dot
        draw.ellipse(
            [
                dot_x - self.DOT_RADIUS,
                dot_y - self.DOT_RADIUS,
                dot_x + self.DOT_RADIUS,
                dot_y + self.DOT_RADIUS,
            ],
            fill=self.DOT_COLOR,
        )

        # Create prompt and ground truth
        prompt = "Is the red dot inside the blue box? Answer true or false."
        ground_truth = RegionContainmentOutput(inside=is_inside)

        generation_params = {
            "epsilon": epsilon,
            "box": {"left": box_left, "right": box_right, "top": box_top, "bottom": box_bottom},
            "dot": {"x": dot_x, "y": dot_y},
            "inside": is_inside,
        }

        return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)

    def _place_dot(
        self,
        box_left: int,
        box_right: int,
        box_top: int,
        box_bottom: int,
        epsilon: int,
        is_inside: bool,
        img_width: int,
        img_height: int,
    ) -> tuple[int, int]:
        """Place a dot exactly epsilon pixels from the box boundary.

        Args:
            box_left, box_right, box_top, box_bottom: Box bounds
            epsilon: Exact distance from boundary
            is_inside: Whether dot should be inside the box
            img_width, img_height: Image dimensions

        Returns:
            (x, y) coordinates of the dot center
        """
        # Try edges in random order until we find one that works
        edges = ["left", "right", "top", "bottom"]
        random.shuffle(edges)

        for edge in edges:
            if edge == "left":
                if is_inside:
                    dot_x = box_left + epsilon
                    # When inside, must be epsilon from ALL edges
                    y_min = box_top + epsilon
                    y_max = box_bottom - epsilon
                else:
                    dot_x = box_left - epsilon
                    # When outside, y can be anywhere along the box
                    y_min = box_top + self.DOT_RADIUS
                    y_max = box_bottom - self.DOT_RADIUS

                # Check image bounds
                if dot_x < self.DOT_RADIUS or dot_x > img_width - self.DOT_RADIUS:
                    continue
                if y_min > y_max:
                    continue

                dot_y = random.randint(y_min, y_max)
                return dot_x, dot_y

            elif edge == "right":
                if is_inside:
                    dot_x = box_right - epsilon
                    # When inside, must be epsilon from ALL edges
                    y_min = box_top + epsilon
                    y_max = box_bottom - epsilon
                else:
                    dot_x = box_right + epsilon
                    # When outside, y can be anywhere along the box
                    y_min = box_top + self.DOT_RADIUS
                    y_max = box_bottom - self.DOT_RADIUS

                # Check image bounds
                if dot_x < self.DOT_RADIUS or dot_x > img_width - self.DOT_RADIUS:
                    continue
                if y_min > y_max:
                    continue

                dot_y = random.randint(y_min, y_max)
                return dot_x, dot_y

            elif edge == "top":
                if is_inside:
                    dot_y = box_top + epsilon
                    # When inside, must be epsilon from ALL edges
                    x_min = box_left + epsilon
                    x_max = box_right - epsilon
                else:
                    dot_y = box_top - epsilon
                    # When outside, x can be anywhere along the box
                    x_min = box_left + self.DOT_RADIUS
                    x_max = box_right - self.DOT_RADIUS

                # Check image bounds
                if dot_y < self.DOT_RADIUS or dot_y > img_height - self.DOT_RADIUS:
                    continue
                if x_min > x_max:
                    continue

                dot_x = random.randint(x_min, x_max)
                return dot_x, dot_y

            else:  # bottom
                if is_inside:
                    dot_y = box_bottom - epsilon
                    # When inside, must be epsilon from ALL edges
                    x_min = box_left + epsilon
                    x_max = box_right - epsilon
                else:
                    dot_y = box_bottom + epsilon
                    # When outside, x can be anywhere along the box
                    x_min = box_left + self.DOT_RADIUS
                    x_max = box_right - self.DOT_RADIUS

                # Check image bounds
                if dot_y < self.DOT_RADIUS or dot_y > img_height - self.DOT_RADIUS:
                    continue
                if x_min > x_max:
                    continue

                dot_x = random.randint(x_min, x_max)
                return dot_x, dot_y

        # Fallback: place inside the box if no valid edge found
        dot_x = (box_left + box_right) // 2
        dot_y = (box_top + box_bottom) // 2
        return dot_x, dot_y
