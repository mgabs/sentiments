#!/usr/bin/env python3
"""Standalone runner for Tier 4 SetFit training & INT8 quantization export."""

import argparse
import sys
from pathlib import Path

from sentiments.providers.tier4.packaging import load_artifact_bundle
from sentiments.training.setfit_pipeline import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_DIR,
    SetFitTrainingConfig,
    run_setfit_training_pipeline,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune compact Sentence Transformer with SetFit and export to quantized INT8 ONNX."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Base Sentence Transformer model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="nlp-datasets/trustpilot-reviews",
        help="Path to labeled review CSVs (default: nlp-datasets/trustpilot-reviews)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Target directory for exported ONNX bundle (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="setfit-minilm-int8",
        help="Exported model name identifier in config.json",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Tokenizer max sequence length (default: {DEFAULT_MAX_LENGTH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Training epochs per iteration (default: 1)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=20,
        help="Number of contrastive pairs generation iterations (default: 20)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Body learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data split and training (default: 42)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable INT8 dynamic quantization export",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    print(f"=== Starting Tier 4 SetFit Training Pipeline ===")
    print(f"  Base Model:    {parsed.model_id}")
    print(f"  Dataset Path:  {parsed.dataset_path}")
    print(f"  Output Dir:    {parsed.output_dir}")
    print(f"  Quantize INT8: {not parsed.no_quantize}")

    config = SetFitTrainingConfig(
        model_id=parsed.model_id,
        dataset_path=parsed.dataset_path,
        output_dir=Path(parsed.output_dir),
        model_name=parsed.model_name,
        max_length=parsed.max_length,
        batch_size=parsed.batch_size,
        num_epochs=parsed.num_epochs,
        num_iterations=parsed.num_iterations,
        learning_rate=parsed.learning_rate,
        seed=parsed.seed,
        quantize=not parsed.no_quantize,
    )

    bundle = run_setfit_training_pipeline(config)

    print(f"\n=== Artifact Bundle Successfully Exported & Validated ===")
    print(f"  Model Path:     {bundle.model_path}")
    print(f"  Tokenizer Path: {bundle.tokenizer_path}")
    print(f"  Model Name:     {bundle.model_name}")
    print(f"  Max Length:     {bundle.max_length}")
    print(f"  ID to Label:    {bundle.id2label}")
    if bundle.model_path.exists():
        print(f"  Model Size:     {bundle.model_path.stat().st_size / (1024 * 1024):.2f} MB")

    # Double verify using load_artifact_bundle
    verified = load_artifact_bundle(parsed.output_dir)
    print(f"  Bundle Check:   OK (model={verified.model_name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
