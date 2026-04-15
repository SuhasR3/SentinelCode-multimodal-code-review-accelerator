from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import warnings
warnings.filterwarnings("ignore")

from src.promise_model import PromiseMLPClassifier


CODE_MODEL_DIR = "outputs/checkpoints/code-model-best"
PROMISE_CKPT_PATH = "outputs/checkpoints/promise-best/promise_mlp.pt"
PROMISE_SCALER_PATH = "data/processed/promise_scaler.pkl"
PROMISE_FEATURE_COLS_PATH = "data/processed/promise_feature_cols.json"

MAX_LENGTH = 256


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "num_samples": len(y_true),
    }


def load_models(device: torch.device):
    code_model_dir = Path(CODE_MODEL_DIR)
    promise_ckpt_path = Path(PROMISE_CKPT_PATH)
    promise_scaler_path = Path(PROMISE_SCALER_PATH)
    promise_feature_cols_path = Path(PROMISE_FEATURE_COLS_PATH)

    if not code_model_dir.exists():
        raise FileNotFoundError(f"Code model dir not found: {code_model_dir}")
    if not promise_ckpt_path.exists():
        raise FileNotFoundError(f"PROMISE checkpoint not found: {promise_ckpt_path}")
    if not promise_scaler_path.exists():
        raise FileNotFoundError(f"PROMISE scaler not found: {promise_scaler_path}")
    if not promise_feature_cols_path.exists():
        raise FileNotFoundError(f"PROMISE feature cols file not found: {promise_feature_cols_path}")

    code_tokenizer = AutoTokenizer.from_pretrained(code_model_dir, use_fast=True)
    code_model = AutoModelForSequenceClassification.from_pretrained(code_model_dir).to(device).eval()

    promise_ckpt = torch.load(promise_ckpt_path, map_location=device)
    promise_feature_cols = json.loads(promise_feature_cols_path.read_text(encoding="utf-8"))
    promise_scaler = joblib.load(promise_scaler_path)

    promise_model = PromiseMLPClassifier(
        input_dim=promise_ckpt["input_dim"],
        hidden_dims=tuple(promise_ckpt["hidden_dims"]),
        emb_dim=promise_ckpt["emb_dim"],
        num_labels=2,
        dropout=promise_ckpt["dropout"],
    ).to(device).eval()

    state_dict = promise_ckpt["model_state_dict"]
    state_dict.pop("loss_fn.weight", None)
    promise_model.load_state_dict(state_dict, strict=False)

    return code_tokenizer, code_model, promise_model, promise_scaler, promise_feature_cols


@torch.no_grad()
def predict_fused(
    text: str,
    metrics_row: Dict[str, float],
    code_tokenizer,
    code_model,
    promise_model,
    promise_scaler,
    promise_feature_cols: List[str],
    device: torch.device,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    code_inputs = code_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    code_inputs = {k: v.to(device) for k, v in code_inputs.items()}

    raw_metrics = pd.DataFrame([metrics_row], columns=promise_feature_cols)
    scaled_metrics = promise_scaler.transform(raw_metrics)
    metrics_tensor = torch.tensor(scaled_metrics, dtype=torch.float32, device=device)

    code_logits = code_model(**code_inputs).logits
    promise_logits = promise_model(metrics=metrics_tensor)["logits"]

    code_probs = torch.softmax(code_logits, dim=-1)
    promise_probs = torch.softmax(promise_logits, dim=-1)

    fused_probs = alpha * code_probs + (1.0 - alpha) * promise_probs
    fused_pred = int(torch.argmax(fused_probs, dim=-1).item())

    return {
        "code_probs": code_probs.squeeze(0).cpu().tolist(),
        "promise_probs": promise_probs.squeeze(0).cpu().tolist(),
        "fused_probs": fused_probs.squeeze(0).cpu().tolist(),
        "fused_pred": fused_pred,
        "fused_label": "buggy" if fused_pred == 1 else "clean",
    }


def evaluate_fusion_csv(
    csv_path: str,
    code_tokenizer,
    code_model,
    promise_model,
    promise_scaler,
    promise_feature_cols: List[str],
    device: torch.device,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    required = ["text", "label"] + promise_feature_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    y_true: List[int] = []
    y_pred: List[int] = []

    for _, row in df.iterrows():
        text = str(row["text"])
        label = int(row["label"])
        metrics_row = {col: float(row[col]) for col in promise_feature_cols}

        out = predict_fused(
            text=text,
            metrics_row=metrics_row,
            code_tokenizer=code_tokenizer,
            code_model=code_model,
            promise_model=promise_model,
            promise_scaler=promise_scaler,
            promise_feature_cols=promise_feature_cols,
            device=device,
            alpha=alpha,
        )

        y_true.append(label)
        y_pred.append(out["fused_pred"])

    metrics = compute_metrics(y_true, y_pred)
    metrics["labels"] = dict(zip(*np.unique(y_true, return_counts=True)))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Test the final fused model.")
    parser.add_argument("--paired_csv", type=str, required=True, help="Paired fusion CSV with text + metrics + label.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for CodeBERT branch.")
    parser.add_argument("--text", type=str, default=None, help="Optional single sample text.")
    parser.add_argument("--metrics_json", type=str, default=None, help="Optional JSON dict of PROMISE metrics.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    code_tokenizer, code_model, promise_model, promise_scaler, promise_feature_cols = load_models(device)

    if args.text and args.metrics_json:
        metrics_row = json.loads(args.metrics_json)
        result = predict_fused(
            text=args.text,
            metrics_row=metrics_row,
            code_tokenizer=code_tokenizer,
            code_model=code_model,
            promise_model=promise_model,
            promise_scaler=promise_scaler,
            promise_feature_cols=promise_feature_cols,
            device=device,
            alpha=args.alpha,
        )
        print(json.dumps(result, indent=2))
        return

    results = evaluate_fusion_csv(
        csv_path=args.paired_csv,
        code_tokenizer=code_tokenizer,
        code_model=code_model,
        promise_model=promise_model,
        promise_scaler=promise_scaler,
        promise_feature_cols=promise_feature_cols,
        device=device,
        alpha=args.alpha,
    )

    print("\nFinal fused model test results")
    print(f"Samples:   {results['num_samples']}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1:        {results['f1']:.4f}")
    print("Confusion matrix:")
    print(results["confusion_matrix"])


if __name__ == "__main__":
    main()