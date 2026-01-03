"""Generator for grid dot location tasks."""

import random
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec


class GridRetrievalOutput(BaseModel):
    cell: str  # Cell name like "B2"


class GridRetrievalGenerator(BaseGenerator):
    """Generate grid images with a red dot to test location identification."""

    task_name = "grid_retrieval"
    output_model = GridRetrievalOutput

    DOT_COLOR = "#E74C3C"  # Red

    @classmethod
    def get_param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="grid_size",
                param_type=int,
                default=10,
                help="Size of the NxN grid",
                min_value=3,
                max_value=26,  # Limited by alphabet for column labels
            ),
        ]

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        super().__init__(output_dir, run_name, seed)
        if seed is not None:
            random.seed(seed)

    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate a grid with a red dot."""
        grid_size = params.get("grid_size", 10)

        # Calculate dimensions
        cell_size = 40
        header_size = 25
        margin = 15

        width = margin + header_size + grid_size * cell_size + margin
        height = margin + header_size + grid_size * cell_size + margin

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        # Pick a random target cell for the dot
        target_row = random.randint(0, grid_size - 1)
        target_col = random.randint(0, grid_size - 1)

        # Cell name (e.g., "B2")
        target_col_letter = chr(65 + target_col)  # A, B, C, ...
        target_row_number = target_row + 1  # 1-indexed
        cell_name = f"{target_col_letter}{target_row_number}"

        # Starting positions
        grid_x = margin + header_size
        grid_y = margin + header_size

        # Draw column headers (A, B, C, ...)
        for col in range(grid_size):
            x = grid_x + col * cell_size + cell_size // 2
            y = margin + header_size // 2
            letter = chr(65 + col)
            draw.text((x - 5, y - 8), letter, fill="black")

        # Draw row headers (1, 2, 3, ...)
        for row in range(grid_size):
            x = margin + header_size // 2
            y = grid_y + row * cell_size + cell_size // 2
            row_label = str(row + 1)
            # Adjust x position for two-digit numbers
            offset = 5 if row + 1 < 10 else 8
            draw.text((x - offset, y - 8), row_label, fill="black")

        # Draw cells
        for row in range(grid_size):
            for col in range(grid_size):
                x1 = grid_x + col * cell_size
                y1 = grid_y + row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                # Alternating colors for readability
                if (row + col) % 2 == 0:
                    fill_color = "#F5F5F5"
                else:
                    fill_color = "#FFFFFF"

                draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline="#CCCCCC", width=1)

        # Draw red dot in target cell
        dot_x = grid_x + target_col * cell_size + cell_size // 2
        dot_y = grid_y + target_row * cell_size + cell_size // 2
        dot_radius = cell_size // 4
        draw.ellipse(
            [dot_x - dot_radius, dot_y - dot_radius, dot_x + dot_radius, dot_y + dot_radius],
            fill=self.DOT_COLOR,
        )

        # Draw outer border
        draw.rectangle(
            [grid_x, grid_y, grid_x + grid_size * cell_size, grid_y + grid_size * cell_size],
            outline="black",
            width=2,
        )

        # Save
        prompt = "Which cell contains the red dot? Answer with the cell name (e.g., B2)."
        ground_truth = GridRetrievalOutput(cell=cell_name)
        generation_params = {
            "grid_size": grid_size,
            "target_cell": cell_name,
            "target_row": target_row,
            "target_col": target_col,
        }

        return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)
