"""Sentiments training package."""

from sentiments.training.setfit_pipeline import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_DIR,
    SetFitONNXWrapper,
    SetFitTrainingConfig,
    export_onnx_bundle,
    extract_linear_head,
    prepare_dataset_splits,
    resolve_dataset_path,
    run_setfit_training_pipeline,
    train_setfit_model,
)

__all__ = [
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_MODEL_ID",
    "DEFAULT_OUTPUT_DIR",
    "SetFitONNXWrapper",
    "SetFitTrainingConfig",
    "export_onnx_bundle",
    "extract_linear_head",
    "prepare_dataset_splits",
    "resolve_dataset_path",
    "run_setfit_training_pipeline",
    "train_setfit_model",
]
