"""SetFit training and INT8 dynamic quantization export pipeline for Tier 4."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sentiments.core.models import SentimentLabel
from sentiments.dataset.loader import DatasetLoader
from sentiments.dataset.splitter import StratifiedSplitter
from sentiments.providers.tier4.packaging import ModelArtifactBundle, load_artifact_bundle

DEFAULT_MODEL_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"
DEFAULT_MAX_LENGTH = 128
DEFAULT_OUTPUT_DIR = Path("models/tier4")

LABEL_TO_ID: dict[SentimentLabel, int] = {
    SentimentLabel.NEGATIVE: 0,
    SentimentLabel.NEUTRAL: 1,
    SentimentLabel.POSITIVE: 2,
}

ID_TO_LABEL_STR: dict[str, str] = {
    "0": "NEGATIVE",
    "1": "NEUTRAL",
    "2": "POSITIVE",
}


@dataclass
class SetFitTrainingConfig:
    """Configuration for SetFit fine-tuning and ONNX export."""

    model_id: str = DEFAULT_MODEL_ID
    dataset_path: Path | str = "nlp-datasets/trustpilot-reviews"
    output_dir: Path | str = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    model_name: str = "setfit-minilm-int8"
    max_length: int = DEFAULT_MAX_LENGTH
    batch_size: int = 16
    num_epochs: int = 1
    num_iterations: int = 20
    learning_rate: float = 2e-5
    seed: int = 42
    quantize: bool = True


def resolve_dataset_path(dataset_path: Path | str) -> Path:
    """Resolve dataset directory checking current dir and parent directories."""
    p = Path(dataset_path)
    if p.exists() and any(p.iterdir()):
        return p
    for parent in Path(__file__).resolve().parents:
        cand = parent / dataset_path
        if cand.exists() and any(cand.iterdir()):
            return cand
    for parent in Path.cwd().resolve().parents:
        cand = parent / dataset_path
        if cand.exists() and any(cand.iterdir()):
            return cand
    return p


def prepare_dataset_splits(
    dataset_path: Path | str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Any, Any, list[Any]]:
    """Load and split records into HuggingFace Datasets for training and validation."""
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for training. Install sentiments[train]."
        ) from exc

    resolved_path = resolve_dataset_path(dataset_path)
    loader = DatasetLoader(resolved_path)
    records = loader.load_all()
    if not records:
        raise ValueError(f"No records found in {resolved_path}")

    splitter = StratifiedSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    split = splitter.split(records)

    train_data = {
        "text": [r.text for r in split.train],
        "label": [LABEL_TO_ID[r.label] for r in split.train],
    }
    val_data = {
        "text": [r.text for r in split.val],
        "label": [LABEL_TO_ID[r.label] for r in split.val],
    }

    train_ds = Dataset.from_dict(train_data)
    val_ds = Dataset.from_dict(val_data)
    return train_ds, val_ds, split.test


def extract_linear_head(
    head_obj: Any,
    embedding_dim: int,
    id2label: dict[int, str] | None = None,
) -> Any:
    """Extract or convert a classification head into a PyTorch nn.Linear module."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("PyTorch is required for head extraction. Install sentiments[train].") from exc

    if isinstance(head_obj, nn.Linear):
        return head_obj

    if isinstance(head_obj, nn.Module):
        if hasattr(head_obj, "linear") and isinstance(head_obj.linear, nn.Linear):
            return head_obj.linear
        for sub in head_obj.modules():
            if isinstance(sub, nn.Linear):
                return sub
        return head_obj

    # Handle scikit-learn models (e.g. LogisticRegression)
    if hasattr(head_obj, "coef_") and hasattr(head_obj, "intercept_"):
        classes = list(getattr(head_obj, "classes_", [0, 1, 2]))
        num_classes = len(classes)
        coef = np.zeros((num_classes, embedding_dim), dtype=np.float32)
        intercept = np.zeros((num_classes,), dtype=np.float32)

        label2id = {v: k for k, v in id2label.items()} if id2label else {}

        for src_idx, c in enumerate(classes):
            if isinstance(c, (int, np.integer)):
                dst_idx = int(c)
            elif str(c) in label2id:
                dst_idx = label2id[str(c)]
            elif str(c).isdigit():
                dst_idx = int(str(c))
            else:
                dst_idx = src_idx

            coef[dst_idx] = head_obj.coef_[src_idx]
            intercept[dst_idx] = head_obj.intercept_[src_idx]

        linear = nn.Linear(embedding_dim, num_classes)
        linear.weight.data = torch.from_numpy(coef).float()
        linear.bias.data = torch.from_numpy(intercept).float()
        return linear

    raise TypeError(f"Unsupported head type: {type(head_obj)}")


try:
    import torch
    import torch.nn as nn

    class SetFitONNXWrapper(nn.Module):
        """End-to-end PyTorch module combining Sentence Transformer backbone and linear head."""

        def __init__(self, sentence_transformer: nn.Module, head: nn.Module):
            super().__init__()
            self.sentence_transformer = sentence_transformer
            self.head = head

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            features = {"input_ids": input_ids, "attention_mask": attention_mask}
            features = self.sentence_transformer(features)
            embeddings = features["sentence_embedding"]
            logits = self.head(embeddings)
            return logits

except ImportError:
    SetFitONNXWrapper = None  # type: ignore


def save_tokenizer(tokenizer: Any, output_dir: Path) -> Path:
    """Save tokenizer to tokenizer.json in target directory."""
    tok_json_path = output_dir / "tokenizer.json"
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(output_dir))

    if not tok_json_path.exists():
        backend = (
            getattr(tokenizer, "backend_tokenizer", None)
            or getattr(tokenizer, "_tokenizer", None)
        )
        if backend and hasattr(backend, "save"):
            backend.save(str(tok_json_path))
        elif hasattr(tokenizer, "save"):
            tokenizer.save(str(tok_json_path))

    if not tok_json_path.exists():
        raise RuntimeError(f"Failed to generate tokenizer.json in {output_dir}")
    return tok_json_path


def export_onnx_bundle(
    model: Any,
    output_dir: Path | str,
    tokenizer: Any | None = None,
    model_name: str = "setfit-minilm-int8",
    max_length: int = DEFAULT_MAX_LENGTH,
    quantize: bool = True,
    base_model_name: str = "",
) -> ModelArtifactBundle:
    """Export fine-tuned model into an ONNX artifact bundle with INT8 quantization."""
    try:
        import torch
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise ImportError(
            "torch, onnx, and onnxruntime are required for ONNX export. Install sentiments[train,tier4]."
        ) from exc

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Determine sentence transformer body and classification head
    if hasattr(model, "model_body"):
        sentence_trans = model.model_body
        head_obj = getattr(model, "model_head", None)
    elif isinstance(model, torch.nn.Module):
        sentence_trans = model
        head_obj = getattr(model, "head", None)
    else:
        raise TypeError(f"Unsupported model type for export: {type(model)}")

    # Extract embedding dimension
    embedding_dim = 384
    if hasattr(sentence_trans, "get_sentence_embedding_dimension"):
        dim = sentence_trans.get_sentence_embedding_dimension()
        if dim:
            embedding_dim = int(dim)
    elif hasattr(head_obj, "coef_"):
        embedding_dim = head_obj.coef_.shape[1]
    elif hasattr(head_obj, "in_features"):
        embedding_dim = head_obj.in_features

    id2label_int = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    head = extract_linear_head(head_obj, embedding_dim=embedding_dim, id2label=id2label_int)

    # Build end-to-end wrapper and ensure CPU placement
    wrapper = SetFitONNXWrapper(sentence_trans, head)
    wrapper.to("cpu")
    wrapper.eval()

    # 2. Export FP32 ONNX graph
    fp32_model_path = out_path / "model.onnx"
    dummy_input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device="cpu")
    dummy_attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long, device="cpu")

    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        str(fp32_model_path),
        dynamo=False,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )

    # 3. Dynamic INT8 Quantization
    if quantize:
        quantized_model_path = out_path / "model_quantized.onnx"
        quantize_dynamic(
            model_input=str(fp32_model_path),
            model_output=str(quantized_model_path),
            weight_type=QuantType.QInt8,
        )

    # 4. Save Tokenizer
    tok = tokenizer
    if tok is None and hasattr(sentence_trans, "tokenizer"):
        tok = sentence_trans.tokenizer
    if tok is not None:
        save_tokenizer(tok, out_path)

    # 5. Write config.json
    config_dict = {
        "model_name": model_name,
        "max_length": max_length,
        "id2label": ID_TO_LABEL_STR,
        "metadata": {
            "quantization": "int8_dynamic" if quantize else "none",
            "base_model": base_model_name or getattr(sentence_trans, "model_name_or_path", "minilm"),
            "num_classes": 3,
            "classes": ["NEGATIVE", "NEUTRAL", "POSITIVE"],
        },
    }
    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # 6. Validate via packaging contract
    return load_artifact_bundle(out_path)


def train_setfit_model(
    config: SetFitTrainingConfig | None = None,
    train_dataset: Any = None,
    eval_dataset: Any = None,
) -> Any:
    """Fine-tune a Sentence Transformer using SetFit."""
    try:
        from setfit import SetFitModel, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError(
            "The 'setfit' package is required for training. Install sentiments[train]."
        ) from exc

    cfg = config or SetFitTrainingConfig()

    # Load base model with balanced class weights for imbalanced review datasets
    model = SetFitModel.from_pretrained(
        cfg.model_id,
        labels=["NEGATIVE", "NEUTRAL", "POSITIVE"],
        head_params={"class_weight": "balanced"},
    )

    args = TrainingArguments(
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        num_iterations=cfg.num_iterations,
        seed=cfg.seed,
        max_length=cfg.max_length,
        body_learning_rate=cfg.learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    return model


def run_setfit_training_pipeline(
    config: SetFitTrainingConfig | None = None,
) -> ModelArtifactBundle:
    """Execute the full training, INT8 quantization, and export pipeline."""
    cfg = config or SetFitTrainingConfig()

    train_ds, val_ds, _ = prepare_dataset_splits(
        dataset_path=cfg.dataset_path,
        seed=cfg.seed,
    )

    model = train_setfit_model(
        config=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    bundle = export_onnx_bundle(
        model=model,
        output_dir=cfg.output_dir,
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        quantize=cfg.quantize,
        base_model_name=cfg.model_id,
    )
    return bundle
