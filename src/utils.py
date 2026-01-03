"""Shared utilities."""

import json
from pathlib import Path


def load_run_metadata(data_dir: Path, run_name: str) -> dict:
    """Load metadata from task_metadata.json in the run directory.

    Args:
        data_dir: Root data directory
        run_name: Run identifier

    Returns:
        Metadata dict containing at least 'task' key

    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    metadata_path = data_dir / run_name / "task_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Run metadata not found: {metadata_path}\n"
            f"Make sure you ran generate_data first for run '{run_name}'"
        )
    with open(metadata_path) as f:
        return json.load(f)
