"""Base class for all image generators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from PIL import Image
from pydantic import BaseModel


@dataclass
class ParamSpec:
    """Specification for a generator parameter."""

    name: str
    param_type: type  # int, float, str, bool
    default: Any
    help: str
    min_value: Any = None
    max_value: Any = None


class BaseGenerator(ABC):
    """Abstract base class for test image generators."""

    task_name: str = "base"
    output_model: type[BaseModel]  # Subclasses must define this
    image_size: tuple[int, int] = (400, 400)

    @classmethod
    def get_param_specs(cls) -> list[ParamSpec]:
        """Return list of parameter specifications for this generator.

        Subclasses should override this to define their custom parameters.
        """
        return []

    @classmethod
    def _compute_accuracy(cls, results: list[dict]) -> dict:
        """Compute accuracy metrics. Shared helper for all generators.

        Each result dict should have 'ground_truth' and 'prediction' keys.
        Correctness is determined by exact equality.
        """
        total_correct = sum(
            1 for r in results if r["prediction"] == r["ground_truth"]
        )
        total_samples = len(results)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        return {
            "correct": total_correct,
            "total": total_samples,
            "accuracy": accuracy,
        }

    @classmethod
    def compute_metrics(cls, results: list[dict]) -> dict:
        """Compute aggregate metrics for all samples.

        Override in subclasses to add task-specific metrics.
        Call cls._compute_accuracy(results) to include accuracy.
        """
        return cls._compute_accuracy(results)

    def __init__(self, output_dir: Path | str, run_name: str, seed: int | None = None):
        """Initialize generator.

        Args:
            output_dir: Base output directory (e.g., 'data')
            run_name: User-provided run identifier (required)
            seed: Random seed for reproducibility
        """
        self.run_name = run_name
        self.output_dir = Path(output_dir) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        # Save task metadata for downstream scripts
        task_metadata_path = self.output_dir / "task_metadata.json"
        task_metadata = {"task": self.task_name}
        with open(task_metadata_path, "w") as f:
            json.dump(task_metadata, f, indent=2)

    @abstractmethod
    def generate_one(self, sample_id: str, **params) -> tuple[Path, Path]:
        """Generate a single test sample.

        Implementations should:
        1. Create the image
        2. Call self._save_sample() to write image + JSON sidecar
        3. Return the paths

        Args:
            sample_id: Unique identifier for this sample
            **params: Generator-specific parameters

        Returns:
            Tuple of (image_path, json_path)
        """
        pass

    def _save_sample(
        self,
        sample_id: str,
        image: Image.Image,
        prompt: str,
        ground_truth: BaseModel,
        params: dict,
    ) -> tuple[Path, Path]:
        """Save image and metadata to disk.

        Args:
            sample_id: Unique identifier for this sample
            image: PIL Image to save
            prompt: The prompt to ask Claude for this sample
            ground_truth: The correct answer as a pydantic model instance
            params: Generation parameters (for reproducibility/analysis)

        Returns:
            Tuple of (image_path, metadata_path)
        """
        # Create sample directory: run_name/sample_id/
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        image_path = sample_dir / "image.png"
        metadata_path = sample_dir / "metadata.json"

        # Save image
        image.save(image_path)

        # Save metadata
        metadata = {
            "prompt": prompt,
            "ground_truth": ground_truth.model_dump(),
            "run_name": self.run_name,
            "params": params,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return image_path, metadata_path
