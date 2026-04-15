from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

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


# -----------------------------
# Metric extraction
# -----------------------------
JAVA_KEYWORDS_AS_OPS = {
    "if", "else", "for", "while", "do", "switch", "case", "default", "break",
    "continue", "return", "try", "catch", "finally", "throw", "throws", "new",
    "this", "super", "instanceof", "class", "extends", "implements", "abstract",
    "static", "final", "public", "private", "protected", "void", "int", "long",
    "double", "float", "boolean", "char", "byte", "short", "null", "true",
    "false", "enum", "interface", "package", "import", "synchronized",
    "volatile", "transient", "native", "strictfp"
}

OPERATOR_SYMBOLS = {
    "==", "!=", "<=", ">=", "&&", "||", "<<", ">>", ">>>",
    "+", "-", "*", "/", "%", "&", "|", "^", "~", "!", "=",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", ">>>=",
    "++", "--", "?", "::", "->", ".", ",", ";", ":", "(", ")", "{", "}", "[", "]"
}

TOKEN_RE = re.compile(
    r"""
    "(?:\\.|[^"])*"            |   # double-quoted string
    '(?:\\.|[^'])*'            |   # single-quoted char/string
    \b\d+\.\d+\b               |   # float
    \b\d+\b                    |   # int
    ==|!=|<=|>=|&&|\|\||<<|>>|>>>|
    \+\+|--|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|>>>=|
    [A-Za-z_][A-Za-z0-9_]*     |   # identifiers / keywords
    [{}()\[\];,.:?+\-*/%&|^~!=<>]   # single-char operators / punctuation
    """,
    re.VERBOSE,
)

LINE_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_comments(code: str) -> str:
    code = BLOCK_COMMENT_RE.sub("", code)
    code = LINE_COMMENT_RE.sub("", code)
    return code


def tokenize_java(code: str) -> List[str]:
    return TOKEN_RE.findall(code)


def count_line_metrics(code: str) -> Tuple[int, int, int, int]:
    lines = code.splitlines()
    if not lines:
        return 0, 0, 0, 0

    loc = sum(1 for line in lines if line.strip())
    blank = sum(1 for line in lines if not line.strip())
    comment = 0
    code_and_comment = 0

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("//") or s.startswith("/*") or s.startswith("*"):
            comment += 1
            continue
        if "//" in line:
            comment += 1
            before, _after = line.split("//", 1)
            if before.strip():
                code_and_comment += 1

    code_lines = max(0, loc - comment)
    return loc, code_lines, comment, blank


def extract_metrics(code: str) -> Dict[str, float]:
    loc, lOCode, lOComment, lOBlank = count_line_metrics(code)

    locCodeAndComment = 0
    for line in code.splitlines():
        if not line.strip():
            continue
        if "//" in line:
            before, _after = line.split("//", 1)
            if before.strip():
                locCodeAndComment += 1

    branch_keywords = re.findall(
        r"\b(?:if|for|while|case|catch|switch|do|else\s+if)\b", code
    )
    branch_ops = code.count("&&") + code.count("||") + code.count("?")
    branchCount = len(branch_keywords) + branch_ops

    vg = branchCount + 1
    evg = vg + max(0, code.count("&&") + code.count("||"))
    ivg = max(1, vg - 1)

    clean_code = strip_comments(code)
    tokens = tokenize_java(clean_code)

    op_tokens: List[str] = []
    opnd_tokens: List[str] = []

    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        if t in OPERATOR_SYMBOLS:
            op_tokens.append(t)
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t):
            if t in JAVA_KEYWORDS_AS_OPS:
                op_tokens.append(t)
            else:
                opnd_tokens.append(t)
        elif re.fullmatch(r'"(?:\\.|[^"])*"|\'(?:\\.|[^\'])*\'|\b\d+\.\d+\b|\b\d+\b', t):
            opnd_tokens.append(t)
        else:
            op_tokens.append(t)

    total_Op = len(op_tokens)
    total_Opnd = len(opnd_tokens)
    uniq_Op = len(set(op_tokens))
    uniq_Opnd = len(set(opnd_tokens))

    n = uniq_Op + uniq_Opnd
    N = total_Op + total_Opnd

    if n > 0 and N > 0:
        v = float(N * math.log2(max(n, 2)))
    else:
        v = 0.0

    if uniq_Opnd > 0:
        d = float((uniq_Op / 2.0) * (total_Opnd / max(1, uniq_Opnd)))
    else:
        d = 0.0

    l = float(1.0 / d) if d > 0 else 0.0
    i = float(v / d) if d > 0 else 0.0
    e = float(d * v)
    b = float(v / 3000.0)
    t = float(e / 18.0) if e > 0 else 0.0

    return {
        "loc": float(loc),
        "v(g)": float(vg),
        "ev(g)": float(evg),
        "iv(g)": float(ivg),
        "n": float(n),
        "v": float(v),
        "l": float(l),
        "d": float(d),
        "i": float(i),
        "e": float(e),
        "b": float(b),
        "t": float(t),
        "lOCode": float(lOCode),
        "lOComment": float(lOComment),
        "lOBlank": float(lOBlank),
        "locCodeAndComment": float(locCodeAndComment),
        "uniq_Op": float(uniq_Op),
        "uniq_Opnd": float(uniq_Opnd),
        "total_Op": float(total_Op),
        "total_Opnd": float(total_Opnd),
        "branchCount": float(branchCount),
    }


# -----------------------------
# Metrics / model loading
# -----------------------------
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


# -----------------------------
# Prediction
# -----------------------------
@torch.no_grad()
def predict_fused(
    text: str,
    code_tokenizer,
    code_model,
    promise_model,
    promise_scaler,
    promise_feature_cols: List[str],
    device: torch.device,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    # This is the input path from the frontend:
    # user uploads code snippet -> this function receives it as `text`
    metrics_row = extract_metrics(text)

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
        "metrics_used": metrics_row,
    }


# -----------------------------
# Evaluation on paired CSV
# -----------------------------
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

        y_true.append(label)
        y_pred.append(fused_pred)

    metrics = compute_metrics(y_true, y_pred)
    metrics["labels"] = dict(zip(*np.unique(y_true, return_counts=True)))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Test the final fused model.")
    parser.add_argument("--paired_csv", type=str, required=False, help="Paired fusion CSV with text + metrics + label.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for CodeBERT branch.")
    parser.add_argument("--text", type=str, default=None, help="Single code snippet to test from frontend or CLI.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    code_tokenizer, code_model, promise_model, promise_scaler, promise_feature_cols = load_models(device)

    # Frontend / single-snippet path
    if args.text is not None:
        result = predict_fused(
            text=args.text,
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

    # Evaluation path
    if not args.paired_csv:
        raise ValueError("Provide either --text for a single snippet or --paired_csv for batch testing.")

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