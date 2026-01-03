"""Generator for grid/table cell lookup tasks."""

import random
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec


class GridLookupOutput(BaseModel):
    value: int


class GridLookupGenerator(BaseGenerator):
    """Generate grid images to test cell lookup ability."""

    task_name = "grid_lookup"
    output_model = GridLookupOutput

    @classmethod
    def get_param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="rows",
                param_type=int,
                default=10,
                help="Number of rows in the grid",
                min_value=3,
                max_value=30,
            ),
            ParamSpec(
                name="cols",
                param_type=int,
                default=10,
                help="Number of columns in the grid",
                min_value=3,
                max_value=30,
            ),
        ]

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        super().__init__(output_dir, run_name, seed)
        if seed is not None:
            random.seed(seed)

    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate a grid lookup test image."""
        rows = params.get("rows", 10)
        cols = params.get("cols", 10)

        # Calculate dimensions
        cell_size = 50
        header_size = 30
        margin = 20

        width = margin + header_size + cols * cell_size + margin
        height = margin + header_size + rows * cell_size + margin

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        # Generate random values for each cell (two-digit numbers)
        values = [[random.randint(10, 99) for _ in range(cols)] for _ in range(rows)]

        # Pick a random target cell
        target_row = random.randint(0, rows - 1)
        target_col = random.randint(0, cols - 1)
        target_value = values[target_row][target_col]

        # Cell name (e.g., "B2")
        target_col_letter = chr(65 + target_col)  # A, B, C, ...
        target_row_number = target_row + 1  # 1-indexed
        cell_name = f"{target_col_letter}{target_row_number}"

        # Starting positions
        grid_x = margin + header_size
        grid_y = margin + header_size

        # Draw column headers (A, B, C, ...)
        for col in range(cols):
            x = grid_x + col * cell_size + cell_size // 2
            y = margin + header_size // 2
            letter = chr(65 + col)
            draw.text((x - 5, y - 8), letter, fill="black")

        # Draw row headers (1, 2, 3, ...)
        for row in range(rows):
            x = margin + header_size // 2
            y = grid_y + row * cell_size + cell_size // 2
            draw.text((x - 5, y - 8), str(row + 1), fill="black")

        # Draw cells
        for row in range(rows):
            for col in range(cols):
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

                # Draw value centered in cell
                value_str = str(values[row][col])
                text_x = x1 + cell_size // 2 - len(value_str) * 4
                text_y = y1 + cell_size // 2 - 8
                draw.text((text_x, text_y), value_str, fill="black")

        # Draw outer border
        draw.rectangle(
            [grid_x, grid_y, grid_x + cols * cell_size, grid_y + rows * cell_size],
            outline="black",
            width=2,
        )

        # Save
        prompt = f"What is the value in cell {cell_name}?"
        ground_truth = GridLookupOutput(value=target_value)
        generation_params = {
            "rows": rows,
            "cols": cols,
            "target_cell": cell_name,
            "all_values": values,
        }

        return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)
