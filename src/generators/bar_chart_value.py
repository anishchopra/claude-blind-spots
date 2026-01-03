"""Generator for bar chart value reading tasks."""

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from .base import BaseGenerator, ParamSpec


class BarChartValueOutput(BaseModel):
    value: int  # The value shown on top of the target bar


class BarChartValueGenerator(BaseGenerator):
    """Generate bar chart images to test reading values from labeled bars."""

    task_name = "bar_chart_value"
    output_model = BarChartValueOutput

    # Colors for bars with their names
    BAR_COLORS = [
        ("#E74C3C", "red"),
        ("#3498DB", "blue"),
        ("#2ECC71", "green"),
        ("#F39C12", "orange"),
        ("#9B59B6", "purple"),
        ("#1ABC9C", "teal"),
        ("#E91E63", "pink"),
        ("#795548", "brown"),
        ("#607D8B", "gray"),
    ]

    TEXT_COLOR = "#2C3E50"
    AXIS_COLOR = "#7F8C8D"

    @classmethod
    def get_param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="num_bars",
                param_type=int,
                default=5,
                help="Number of bars in the chart",
                min_value=2,
                max_value=9,
            ),
            ParamSpec(
                name="label_font_size",
                param_type=int,
                default=12,
                help="Font size for value labels (smaller = harder)",
                min_value=8,
                max_value=20,
            ),
        ]

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        super().__init__(output_dir, run_name, seed)
        if seed is not None:
            random.seed(seed)

    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate a bar chart image with labeled bars."""
        num_bars = params.get("num_bars", 5)
        label_font_size = params.get("label_font_size", 12)

        # Select colors for bars
        colors_used = random.sample(self.BAR_COLORS, num_bars)

        # Generate random values for each bar
        values = [random.randint(10, 99) for _ in range(num_bars)]

        # Select a random bar to ask about
        query_idx = random.randint(0, num_bars - 1)
        query_color_hex, query_color_name = colors_used[query_idx]
        query_value = values[query_idx]

        # Create image
        img = Image.new("RGB", self.image_size, "white")
        draw = ImageDraw.Draw(img)

        # Draw the bar chart
        self._draw_chart(draw, colors_used, values, label_font_size)

        # Create prompt and ground truth
        prompt = f"What is the value of the {query_color_name} bar?"
        ground_truth = BarChartValueOutput(value=query_value)

        generation_params = {
            "num_bars": num_bars,
            "label_font_size": label_font_size,
            "query_color": query_color_name,
            "query_value": query_value,
            "all_values": {colors_used[i][1]: values[i] for i in range(num_bars)},
        }

        return self._save_sample(sample_id, img, prompt, ground_truth, generation_params)

    def _draw_chart(
        self,
        draw: ImageDraw.Draw,
        colors: list[tuple[str, str]],
        values: list[int],
        label_font_size: int,
    ) -> None:
        """Draw the bar chart with value labels."""
        width, height = self.image_size
        margin_left = 50
        margin_right = 30
        margin_top = 40
        margin_bottom = 50

        chart_width = width - margin_left - margin_right
        chart_height = height - margin_top - margin_bottom

        num_bars = len(values)
        max_value = max(values)

        # Calculate bar dimensions
        total_bar_space = chart_width * 0.8  # 80% for bars, 20% for gaps
        bar_width = total_bar_space / num_bars
        gap_width = (chart_width - total_bar_space) / (num_bars + 1)

        # Load fonts
        try:
            label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", label_font_size)
            axis_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except (OSError, IOError):
            try:
                label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", label_font_size)
                axis_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            except (OSError, IOError):
                label_font = ImageFont.load_default()
                axis_font = ImageFont.load_default()

        # Draw axes
        # Y-axis
        draw.line(
            [(margin_left, margin_top), (margin_left, height - margin_bottom)],
            fill=self.AXIS_COLOR,
            width=2,
        )
        # X-axis
        draw.line(
            [(margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom)],
            fill=self.AXIS_COLOR,
            width=2,
        )

        # Draw Y-axis ticks and labels
        num_ticks = 5
        for i in range(num_ticks + 1):
            tick_value = int(max_value * i / num_ticks)
            tick_y = height - margin_bottom - (chart_height * i / num_ticks)

            # Tick mark
            draw.line(
                [(margin_left - 5, tick_y), (margin_left, tick_y)],
                fill=self.AXIS_COLOR,
                width=1,
            )

            # Label
            label = str(tick_value)
            text_bbox = draw.textbbox((0, 0), label, font=axis_font)
            text_width = text_bbox[2] - text_bbox[0]
            draw.text(
                (margin_left - 10 - text_width, tick_y - 5),
                label,
                fill=self.TEXT_COLOR,
                font=axis_font,
            )

        # Draw bars with labels
        for i, (value, (color_hex, color_name)) in enumerate(zip(values, colors)):
            # Calculate bar position
            bar_x = margin_left + gap_width + i * (bar_width + gap_width)
            bar_height = (value / max_value) * chart_height
            bar_y = height - margin_bottom - bar_height

            # Draw bar
            draw.rectangle(
                [bar_x, bar_y, bar_x + bar_width, height - margin_bottom],
                fill=color_hex,
                outline="#2C3E50",
                width=1,
            )

            # Draw value label on top of bar
            label = str(value)
            text_bbox = draw.textbbox((0, 0), label, font=label_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            label_x = bar_x + (bar_width - text_width) / 2
            label_y = bar_y - text_height - 3

            draw.text(
                (label_x, label_y),
                label,
                fill=self.TEXT_COLOR,
                font=label_font,
            )
