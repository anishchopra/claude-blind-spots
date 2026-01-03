"""Generator for swimlane chart category identification tasks."""

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec


class SwimlaneOutput(BaseModel):
    category: str  # The name of the lane/category


class SwimlaneGenerator(BaseGenerator):
    """Generate swimlane chart images to test category identification."""

    task_name = "swimlane"
    output_model = SwimlaneOutput

    BOX_WIDTH = 45
    BOX_HEIGHT = 25
    BOX_COLOR = "#E8F4FD"
    BOX_BORDER_COLOR = "#2980B9"
    LANE_COLORS = [
        "#FFF9E6",  # Light yellow
        "#E6F3FF",  # Light blue
        "#E6FFE6",  # Light green
        "#FFE6E6",  # Light pink
        "#F0E6FF",  # Light purple
        "#E6FFFF",  # Light cyan
    ]
    LANE_BORDER_COLOR = "#7F8C8D"
    TEXT_COLOR = "#2C3E50"
    LABEL_BG_COLOR = "#ECF0F1"

    # Category names for lanes
    CATEGORY_NAMES = [
        "Sales",
        "Engineering",
        "Marketing",
        "Support",
        "Finance",
        "Operations",
    ]

    @classmethod
    def get_param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="num_lanes",
                param_type=int,
                default=3,
                help="Number of swimlanes",
                min_value=2,
                max_value=6,
            ),
            ParamSpec(
                name="num_boxes",
                param_type=int,
                default=8,
                help="Number of boxes",
                min_value=3,
                max_value=25,
            ),
        ]

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        super().__init__(output_dir, run_name, seed)
        if seed is not None:
            random.seed(seed)

    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate a swimlane chart image with a category identification question."""
        num_lanes = params.get("num_lanes", 3)
        num_boxes = params.get("num_boxes", 8)

        # Select category names for lanes
        lane_names = random.sample(self.CATEGORY_NAMES, num_lanes)

        # Distribute boxes across lanes (ensure each lane has at least 1 box)
        box_assignments = self._distribute_boxes(num_boxes, num_lanes)

        # Create box labels (A, B, C, ...)
        box_labels = [chr(ord("A") + i) for i in range(num_boxes)]

        # Assign boxes to lanes
        lane_boxes: list[list[str]] = [[] for _ in range(num_lanes)]
        box_to_lane: dict[str, int] = {}

        box_idx = 0
        for lane_idx, count in enumerate(box_assignments):
            for _ in range(count):
                label = box_labels[box_idx]
                lane_boxes[lane_idx].append(label)
                box_to_lane[label] = lane_idx
                box_idx += 1

        # Sort boxes within each lane alphabetically so flow goes left-to-right
        for lane in lane_boxes:
            lane.sort()

        # Select a random box to ask about
        query_box = random.choice(box_labels)
        answer_lane_idx = box_to_lane[query_box]
        answer_category = lane_names[answer_lane_idx]

        # Create image
        img = Image.new("RGB", self.image_size, "white")
        draw = ImageDraw.Draw(img)

        # Calculate layout
        positions = self._calculate_layout(lane_boxes, num_lanes)

        # Draw lanes
        self._draw_lanes(draw, num_lanes, lane_names)

        # Draw boxes
        self._draw_boxes(draw, lane_boxes, positions)

        # Draw arrows between sequential boxes (optional visual complexity)
        self._draw_arrows(draw, box_labels, positions, box_to_lane)

        # Create prompt and ground truth
        prompt = f"Which category does step {query_box} belong to?"
        ground_truth = SwimlaneOutput(category=answer_category)

        generation_params = {
            "num_lanes": num_lanes,
            "num_boxes": num_boxes,
            "lane_names": lane_names,
            "query_box": query_box,
            "answer_category": answer_category,
            "box_assignments": {lane_names[i]: lane_boxes[i] for i in range(num_lanes)},
        }

        return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)

    def _distribute_boxes(self, num_boxes: int, num_lanes: int) -> list[int]:
        """Distribute boxes across lanes, ensuring each lane has at least 1.

        Returns:
            List of box counts per lane
        """
        # Start with 1 box per lane
        distribution = [1] * num_lanes
        remaining = num_boxes - num_lanes

        # Distribute remaining boxes randomly
        for _ in range(remaining):
            lane_idx = random.randint(0, num_lanes - 1)
            distribution[lane_idx] += 1

        return distribution

    def _calculate_layout(
        self, lane_boxes: list[list[str]], num_lanes: int
    ) -> dict[str, tuple[int, int]]:
        """Calculate positions for all boxes.

        Returns:
            Dictionary mapping box labels to (x, y) positions
        """
        width, height = self.image_size
        label_width = 80  # Width reserved for lane labels
        content_width = width - label_width - 20  # Margin on right

        lane_height = (height - 20) // num_lanes  # 20px margin

        positions = {}

        for lane_idx, boxes in enumerate(lane_boxes):
            if not boxes:
                continue

            # Lane vertical center
            lane_y = 10 + lane_idx * lane_height + lane_height // 2

            # Distribute boxes horizontally within the lane
            num_boxes_in_lane = len(boxes)
            box_spacing = content_width // (num_boxes_in_lane + 1)

            for i, box_label in enumerate(boxes):
                box_x = label_width + (i + 1) * box_spacing
                positions[box_label] = (box_x, lane_y)

        return positions

    def _draw_lanes(
        self, draw: ImageDraw.Draw, num_lanes: int, lane_names: list[str]
    ) -> None:
        """Draw the swimlane backgrounds and labels."""
        width, height = self.image_size
        label_width = 80
        lane_height = (height - 20) // num_lanes

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except (OSError, IOError):
                font = ImageFont.load_default()

        for lane_idx in range(num_lanes):
            y_start = 10 + lane_idx * lane_height
            y_end = y_start + lane_height

            # Draw lane background
            lane_color = self.LANE_COLORS[lane_idx % len(self.LANE_COLORS)]
            draw.rectangle(
                [label_width, y_start, width - 10, y_end],
                fill=lane_color,
                outline=self.LANE_BORDER_COLOR,
                width=1,
            )

            # Draw label background
            draw.rectangle(
                [10, y_start, label_width, y_end],
                fill=self.LABEL_BG_COLOR,
                outline=self.LANE_BORDER_COLOR,
                width=1,
            )

            # Draw lane label (rotated text would be ideal, but PIL doesn't support it easily)
            # So we'll draw it horizontally, centered
            label = lane_names[lane_idx]
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_x = 10 + (label_width - 10 - text_width) // 2
            text_y = y_start + (lane_height - text_height) // 2

            draw.text((text_x, text_y), label, fill=self.TEXT_COLOR, font=font)

    def _draw_boxes(
        self,
        draw: ImageDraw.Draw,
        lane_boxes: list[list[str]],
        positions: dict[str, tuple[int, int]],
    ) -> None:
        """Draw all boxes with their labels."""
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except (OSError, IOError):
                font = ImageFont.load_default()

        for lane in lane_boxes:
            for box_label in lane:
                x, y = positions[box_label]

                # Draw box
                box_coords = [
                    x - self.BOX_WIDTH // 2,
                    y - self.BOX_HEIGHT // 2,
                    x + self.BOX_WIDTH // 2,
                    y + self.BOX_HEIGHT // 2,
                ]
                draw.rectangle(
                    box_coords,
                    fill=self.BOX_COLOR,
                    outline=self.BOX_BORDER_COLOR,
                    width=2,
                )

                # Draw label
                text_bbox = draw.textbbox((0, 0), box_label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = x - text_width // 2
                text_y = y - text_height // 2

                draw.text((text_x, text_y), box_label, fill=self.TEXT_COLOR, font=font)

    def _draw_arrows(
        self,
        draw: ImageDraw.Draw,
        box_labels: list[str],
        positions: dict[str, tuple[int, int]],
        box_to_lane: dict[str, int],
    ) -> None:
        """Draw arrows between sequential boxes to show flow."""
        import math

        arrow_color = "#7F8C8D"

        # Connect boxes in alphabetical order (A->B->C->...)
        for i in range(len(box_labels) - 1):
            source = box_labels[i]
            target = box_labels[i + 1]

            x1, y1 = positions[source]
            x2, y2 = positions[target]

            # Adjust start/end points based on relative positions
            if x2 > x1:
                # Target is to the right
                x1 += self.BOX_WIDTH // 2
                x2 -= self.BOX_WIDTH // 2
            elif x2 < x1:
                # Target is to the left
                x1 -= self.BOX_WIDTH // 2
                x2 += self.BOX_WIDTH // 2
            else:
                # Same column, adjust vertically
                if y2 > y1:
                    y1 += self.BOX_HEIGHT // 2
                    y2 -= self.BOX_HEIGHT // 2
                else:
                    y1 -= self.BOX_HEIGHT // 2
                    y2 += self.BOX_HEIGHT // 2

            # Draw line
            draw.line([(x1, y1), (x2, y2)], fill=arrow_color, width=1)

            # Draw arrowhead
            arrow_length = 8
            arrow_angle = math.pi / 6

            angle = math.atan2(y2 - y1, x2 - x1)
            left_x = x2 - arrow_length * math.cos(angle - arrow_angle)
            left_y = y2 - arrow_length * math.sin(angle - arrow_angle)
            right_x = x2 - arrow_length * math.cos(angle + arrow_angle)
            right_y = y2 - arrow_length * math.sin(angle + arrow_angle)

            draw.polygon(
                [(x2, y2), (left_x, left_y), (right_x, right_y)],
                fill=arrow_color,
            )
