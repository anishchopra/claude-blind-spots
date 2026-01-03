"""Run Claude inference on generated samples."""

import asyncio
import base64
import json
from pathlib import Path
from typing import Type

from anthropic import AsyncAnthropic, RateLimitError
from pydantic import BaseModel
from tqdm import tqdm

from src.generators import GENERATORS


async def predict_sample(
    client: AsyncAnthropic,
    sample_dir: Path,
    output_model: Type[BaseModel],
    model: str,
    thinking_budget: int | None = None,
) -> dict:
    """Run inference on a single sample with Claude.

    Args:
        client: Async Anthropic client
        sample_dir: Path to sample directory containing image.png and metadata.json
        output_model: Pydantic model for structured output
        model: Model to use
        thinking_budget: If provided, enable extended thinking with this token budget

    Returns:
        dict with prediction and usage info
    """
    image_path = sample_dir / "image.png"
    metadata_path = sample_dir / "metadata.json"

    # Load metadata to get prompt
    with open(metadata_path) as f:
        metadata = json.load(f)
    prompt = metadata["prompt"]

    # Load image as base64
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Build API call kwargs
    api_kwargs = {
        "model": model,
        "max_tokens": thinking_budget + 256 if thinking_budget else 256,
        "betas": ["structured-outputs-2025-11-13"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        "output_format": output_model,
    }

    if thinking_budget is not None:
        api_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }

    # Call Claude with structured output
    response = await client.beta.messages.parse(**api_kwargs)

    # Extract prediction
    prediction = response.parsed_output.model_dump()

    # Build result
    result = {
        "prediction": prediction,
        "model": model,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }

    # Save prediction
    prediction_path = sample_dir / "prediction.json"
    with open(prediction_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


async def run_inference(
    task_name: str,
    run_name: str,
    data_dir: Path,
    model: str,
    max_concurrent: int = 10,
    rerun: bool = False,
    thinking_budget: int | None = None,
) -> list[dict]:
    """Run inference on all samples for a task run.

    Args:
        task_name: Name of the task (must be in GENERATORS)
        run_name: Run identifier
        data_dir: Root data directory
        model: Model to use
        max_concurrent: Max concurrent API calls
        rerun: If True, re-run inference even if prediction.json exists
        thinking_budget: If provided, enable extended thinking with this token budget

    Returns:
        List of result dicts
    """
    if task_name not in GENERATORS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(GENERATORS.keys())}")

    generator_cls = GENERATORS[task_name]
    output_model = generator_cls.output_model

    # Path: data/{run_name}/
    task_dir = data_dir / run_name

    if not task_dir.exists():
        print(f"Task directory not found: {task_dir}")
        return []

    # Find all sample directories (no difficulty subdirectories)
    sample_dirs = [
        d for d in task_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ]

    if not sample_dirs:
        print(f"No samples found in {task_dir}")
        return []

    # Filter out samples that already have predictions (unless rerun)
    if not rerun:
        total_found = len(sample_dirs)
        sample_dirs = [d for d in sample_dirs if not (d / "prediction.json").exists()]
        skipped = total_found - len(sample_dirs)
        if skipped > 0:
            print(f"Found {total_found} samples, skipping {skipped} with existing predictions")
        else:
            print(f"Found {len(sample_dirs)} samples")
    else:
        print(f"Found {len(sample_dirs)} samples")

    if not sample_dirs:
        print("No samples to process")
        return []

    # Create client
    client = AsyncAnthropic()

    # Semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Progress bar and error tracking
    pbar = tqdm(total=len(sample_dirs), desc="Running inference")
    errors = []

    async def predict_with_semaphore(sample_dir: Path) -> dict:
        async with semaphore:
            max_retries = 5
            base_delay = 1.0

            for attempt in range(max_retries):
                try:
                    result = await predict_sample(
                        client=client,
                        sample_dir=sample_dir,
                        output_model=output_model,
                        model=model,
                        thinking_budget=thinking_budget,
                    )
                    pbar.update(1)
                    return result
                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                        continue
                    errors.append((sample_dir.name, f"Rate limited after {max_retries} retries"))
                    pbar.update(1)
                    return {
                        "error": str(e),
                        "sample_dir": str(sample_dir),
                    }
                except Exception as e:
                    errors.append((sample_dir.name, str(e)))
                    pbar.update(1)
                    return {
                        "error": str(e),
                        "sample_dir": str(sample_dir),
                    }

    # Run all predictions concurrently
    results = await asyncio.gather(*[
        predict_with_semaphore(sample_dir)
        for sample_dir in sample_dirs
    ])
    pbar.close()

    # Print errors if any
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for sample_name, error_msg in errors:
            print(f"  {sample_name}: {error_msg}")

    # Summary
    error_count = sum(1 for r in results if "error" in r)
    success = len(results) - error_count

    print(f"Completed: {success}/{len(results)} ({error_count} errors)")

    return results
