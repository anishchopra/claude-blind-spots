"""Generator for path following tasks - tracing paths from labeled boxes."""

import random
import string
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec


# Distinct colors for paths
PATH_COLORS = [
    "#E53935",  # Red
    "#1E88E5",  # Blue
    "#43A047",  # Green
    "#FB8C00",  # Orange
    "#8E24AA",  # Purple
    "#00ACC1",  # Cyan
    "#F4511E",  # Deep Orange
    "#3949AB",  # Indigo
]


class PathFollowingOutput(BaseModel):
    answer: int


class PathFollowingGenerator(BaseGenerator):
    """Generate path following images to test visual tracking ability.

    Tests whether the model can follow a winding path from a labeled
    letter box to its corresponding numbered box.
    """

    task_name = "path_following"
    output_model = PathFollowingOutput

    @classmethod
    def get_param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="num_boxes",
                param_type=int,
                default=4,
                help="Number of boxes on each side (2-8)",
                min_value=2,
                max_value=8,
            ),
            ParamSpec(
                name="num_bends",
                param_type=int,
                default=4,
                help="Number of bends in each path (must be even, 2-8)",
                min_value=2,
                max_value=8,
            ),
        ]

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        super().__init__(output_dir, run_name, seed)
        if seed is not None:
            random.seed(seed)

    def _generate_path(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        num_bends: int,
        all_paths: list[list[tuple[int, int]]],
        y_positions: list[int],
    ) -> list[tuple[int, int]]:
        """Generate a path with the specified number of bends.

        The path alternates between horizontal and vertical segments.
        For even num_bends, we need num_bends + 1 segments.
        """
        points = [(start_x, start_y)]

        # We need to go from start to end with num_bends turns
        # Path structure: H -> V -> H -> V -> ... -> H (for even bends starting horizontal)
        # Or: start horizontal, then alternate

        # Calculate intermediate x positions for vertical segments
        total_horizontal = end_x - start_x
        num_horizontal_segments = (num_bends // 2) + 1

        # Divide the horizontal space into segments
        segment_width = total_horizontal // num_horizontal_segments

        # Generate x positions for bends (where vertical segments occur)
        x_positions = []
        for i in range(1, num_horizontal_segments):
            # Add some randomness to x positions
            base_x = start_x + i * segment_width
            jitter = random.randint(-segment_width // 4, segment_width // 4)
            x_pos = max(start_x + 30, min(end_x - 30, base_x + jitter))
            x_positions.append(x_pos)

        # Generate y positions for horizontal segments (between bends)
        # Avoid y positions too close to box positions
        available_y_range = []
        min_y = min(start_y, end_y) - 50
        max_y = max(start_y, end_y) + 50

        # Current position
        curr_x, curr_y = start_x, start_y

        for i in range(num_bends):
            if i % 2 == 0:
                # Horizontal segment followed by vertical turn
                if i // 2 < len(x_positions):
                    next_x = x_positions[i // 2]
                else:
                    next_x = end_x
                points.append((next_x, curr_y))
                curr_x = next_x
            else:
                # Vertical segment followed by horizontal turn
                # Pick a y position that's different from current
                if i == num_bends - 1:
                    # Last vertical segment should go to end_y
                    next_y = end_y
                else:
                    # Pick a random y that creates visual separation
                    attempts = 0
                    while attempts < 50:
                        next_y = random.randint(
                            min(start_y, end_y) - 30,
                            max(start_y, end_y) + 30
                        )
                        # Make sure it's different enough from current y
                        if abs(next_y - curr_y) > 20:
                            break
                        attempts += 1
                    else:
                        next_y = curr_y + random.choice([-30, 30])

                points.append((curr_x, next_y))
                curr_y = next_y

        # Final segment to end point
        points.append((end_x, curr_y))
        if curr_y != end_y:
            points.append((end_x, end_y))

        return points

    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate a path following test image."""
        num_boxes = params.get("num_boxes", 4)
        num_bends = params.get("num_bends", 4)

        # Ensure num_bends is even
        if num_bends % 2 != 0:
            num_bends = num_bends + 1

        # Image dimensions
        width, height = 600, 500
        margin = 60
        box_size = 35

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        # Calculate vertical spacing for boxes
        usable_height = height - 2 * margin
        if num_boxes > 1:
            vertical_spacing = usable_height // (num_boxes - 1)
        else:
            vertical_spacing = 0

        # Box positions
        left_x = margin
        right_x = width - margin - box_size

        # Generate labels
        letters = list(string.ascii_uppercase[:num_boxes])
        numbers = list(range(1, num_boxes + 1))

        # Random mapping from letters to numbers
        number_assignment = numbers.copy()
        random.shuffle(number_assignment)
        letter_to_number = dict(zip(letters, number_assignment))

        # Calculate y positions for each box
        y_positions = []
        for i in range(num_boxes):
            if num_boxes > 1:
                y = margin + i * vertical_spacing
            else:
                y = height // 2 - box_size // 2
            y_positions.append(y)

        # Get colors for paths
        colors = PATH_COLORS[:num_boxes]
        if len(colors) < num_boxes:
            # Generate additional colors if needed
            for i in range(num_boxes - len(colors)):
                colors.append(f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}")

        random.shuffle(colors)
        letter_to_color = dict(zip(letters, colors))

        # Generate and draw paths
        all_paths = []
        for i, letter in enumerate(letters):
            target_number = letter_to_number[letter]
            target_idx = target_number - 1  # Convert to 0-indexed

            # Start and end points (center of boxes)
            start_x = left_x + box_size
            start_y = y_positions[i] + box_size // 2
            end_x = right_x
            end_y = y_positions[target_idx] + box_size // 2

            # Generate path
            path_points = self._generate_path(
                start_x, start_y, end_x, end_y, num_bends, all_paths, y_positions
            )
            all_paths.append(path_points)

            # Draw path
            color = letter_to_color[letter]
            for j in range(len(path_points) - 1):
                draw.line(
                    [path_points[j], path_points[j + 1]],
                    fill=color,
                    width=3,
                )

        # Draw boxes on top of paths
        for i, letter in enumerate(letters):
            y = y_positions[i]

            # Left box (letter) - black outline
            draw.rectangle(
                [left_x, y, left_x + box_size, y + box_size],
                fill="white",
                outline="black",
                width=2,
            )
            # Center the letter in the box (larger text)
            draw.text(
                (left_x + box_size // 2 - 7, y + box_size // 2 - 10),
                letter,
                fill="black",
                font_size=20,
            )

            # Right box (number)
            draw.rectangle(
                [right_x, y, right_x + box_size, y + box_size],
                fill="white",
                outline="black",
                width=2,
            )
            draw.text(
                (right_x + box_size // 2 - 7, y + box_size // 2 - 10),
                str(i + 1),
                fill="black",
                font_size=20,
            )

        # Pick a random letter to ask about
        target_letter = random.choice(letters)
        target_answer = letter_to_number[target_letter]

        # Save
        # prompt = f"Follow the path starting from box {target_letter}. Which numbered box does it lead to?"
        prompt = f"Follow the path starting from box {target_letter}. The path will only be one continuous color, but other paths may overlap with it, obscuring part of the path. Be sure to follow the path with the correct color, and determine the answer to this question; Which numbered box does it lead to?"
        ground_truth = PathFollowingOutput(answer=target_answer)

        generation_params = {
            "num_boxes": num_boxes,
            "num_bends": num_bends,
            "target_letter": target_letter,
            "target_answer": target_answer,
            "letter_to_number": letter_to_number,
            "letter_to_color": letter_to_color,
        }

        return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)
