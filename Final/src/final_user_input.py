from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Any, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
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
OPENAI_RESPONSES_PATH = "/v1/responses"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com"
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


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


@lru_cache(maxsize=1)
def load_runtime() -> Dict[str, Any]:
    load_dotenv(ENV_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    (
        code_tokenizer,
        code_model,
        promise_model,
        promise_scaler,
        promise_feature_cols,
    ) = load_models(device)
    return {
        "device": device,
        "code_tokenizer": code_tokenizer,
        "code_model": code_model,
        "promise_model": promise_model,
        "promise_scaler": promise_scaler,
        "promise_feature_cols": promise_feature_cols,
    }


def build_llm_review_prompt(
    text: str,
    prediction: Dict[str, Any],
    metrics_row: Dict[str, float],
) -> str:
    fused_buggy = float(prediction["fused_probs"][1])
    code_buggy = float(prediction["code_probs"][1])
    promise_buggy = float(prediction["promise_probs"][1])
    confidence_label = classify_confidence(fused_buggy)
    branch_alignment = describe_branch_alignment(code_buggy, promise_buggy)

    return (
        "You are a senior secure code review assistant.\n"
        "You are given a code snippet, a defect-classifier prediction, and structural metrics.\n"
        "Use the model output only as evidence, never as proof.\n"
        "Do not invent a defect when the code is simple or the evidence is weak.\n"
        "If confidence is low or the model branches disagree, say that explicitly and downgrade the verdict.\n"
        "Prefer cautious language such as 'possible concern' or 'no concrete defect visible from snippet alone'.\n"
        "Only provide corrected code when there is a plausible concrete improvement.\n"
        "Optimize the response for a project demo: concise, clear, non-repetitive, and easy to scan.\n\n"
        "Write a short code review in plain text.\n"
        "Use clear section headings, but you do not need to follow any exact schema.\n"
        "Good sections are: Verdict, Summary, Likely Issue, and Recommended Action.\n"
        "Do not include sections named 'Why It Matters' or 'Suggested Fix'.\n"
        "Keep it brief, readable, and presentation-friendly.\n"
        "Limit the total response to about 120-180 words.\n"
        "Use short bullets and short sentences.\n"
        "If there is no clear issue, say that directly.\n"
        "If there is a concrete defect, include at least one actionable fix.\n\n"
        f"Derived confidence band: {confidence_label}\n"
        f"Branch agreement: {branch_alignment}\n\n"
        f"Classifier prediction:\n{json.dumps(prediction, indent=2)}\n\n"
        f"Extracted metrics:\n{json.dumps(metrics_row, indent=2)}\n\n"
        "Code snippet:\n"
        "```java\n"
        f"{text}\n"
        "```"
    )


def extract_response_text(response_json: Dict[str, Any]) -> str:
    output_items = response_json.get("output", [])

    texts: List[str] = []
    for item in output_items:
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))

    if texts:
        return "\n".join(part for part in texts if part.strip()).strip()

    return response_json.get("output_text", "").strip()


def default_llm_review(raw_text: str = "") -> Dict[str, Any]:
    return {
        "verdict": "",
        "headline": "",
        "risk_level": "low",
        "issue_locations": [],
        "why_it_matters": [],
        "recommended_action": [],
        "corrected_code": "",
        "confidence_note": "",
        "presentation_note": "",
        "raw_text": raw_text.strip(),
        "parse_status": "plain_text",
        "is_truncated": False,
        "sections": [],
    }


def normalize_llm_review(parsed: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    review = default_llm_review(raw_text)
    review.update(parsed)

    for key in ("issue_locations", "why_it_matters", "recommended_action", "sections"):
        value = review.get(key, [])
        if key == "sections":
            if not isinstance(value, list):
                review[key] = []
            continue
        if isinstance(value, str):
            review[key] = [value] if value.strip() else []
        elif not isinstance(value, list):
            review[key] = []

    for key in ("verdict", "headline", "risk_level", "corrected_code", "confidence_note", "presentation_note"):
        value = review.get(key, "")
        if not isinstance(value, str):
            review[key] = str(value)

    review["risk_level"] = review["risk_level"].strip().lower() or "low"
    if review["risk_level"] not in {"low", "moderate", "high"}:
        review["risk_level"] = "low"

    review["raw_text"] = raw_text.strip()
    review["parse_status"] = parsed.get("parse_status", review.get("parse_status", "parsed"))
    review["is_truncated"] = bool(parsed.get("is_truncated", False))
    return review


def parse_llm_review(response_text: str) -> Dict[str, Any]:
    cleaned = response_text.strip()
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)
    review = default_llm_review(cleaned)
    review["headline"] = cleaned.splitlines()[0].strip()[:160] if cleaned else "LLM review unavailable."
    review["is_truncated"] = looks_truncated(cleaned)
    review["sections"] = extract_review_sections(cleaned)
    return normalize_llm_review(review, cleaned)


def extract_review_sections(text: str) -> List[Dict[str, Any]]:
    canonical_titles = {
        "verdict": "Verdict",
        "summary": "Summary",
        "likely issue": "Likely Issue",
        "likely issue (possible concern)": "Likely Issue",
        "recommended action": "Recommended Action",
    }

    sections: List[Dict[str, Any]] = []
    current_title = "Review"
    current_lines: List[str] = []

    def flush_section() -> None:
        nonlocal current_title, current_lines, sections
        body_lines = [line.rstrip() for line in current_lines if line.strip()]
        if not body_lines:
            current_lines = []
            return

        bullets = []
        paragraphs = []
        for line in body_lines:
            stripped = line.strip()
            if stripped.startswith("- "):
                bullets.append(stripped[2:].strip())
            else:
                paragraphs.append(stripped)

        sections.append(
            {
                "title": current_title,
                "paragraphs": paragraphs,
                "bullets": bullets,
            }
        )
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            current_lines.append("")
            continue

        normalized = line.rstrip(":").strip().lower()
        if normalized in canonical_titles:
            flush_section()
            current_title = canonical_titles[normalized]
            continue

        current_lines.append(raw_line)

    flush_section()
    return sections


def looks_truncated(text: str) -> bool:
    if not text:
        return False

    stripped = text.rstrip()
    if stripped.endswith(("because", "due to", "for example", "such as", "including")):
        return True

    last_line = stripped.splitlines()[-1].strip()
    if last_line.startswith("- ") and len(last_line) < 24:
        return True

    if stripped[-1] not in ".!?`":
        word_count = len(stripped.split())
        if word_count > 20:
            return True

    return False


def classify_confidence(fused_buggy_prob: float) -> str:
    if fused_buggy_prob >= 0.8:
        return "high"
    if fused_buggy_prob >= 0.6:
        return "moderate"
    return "low"


def describe_branch_alignment(code_buggy_prob: float, promise_buggy_prob: float) -> str:
    code_label = "buggy" if code_buggy_prob >= 0.5 else "clean"
    promise_label = "buggy" if promise_buggy_prob >= 0.5 else "clean"

    if code_label == promise_label:
        return f"aligned ({code_label})"
    return "disagreement between semantic and structural branches"


def summarize_prediction_for_display(result: Dict[str, Any], alpha: float) -> Dict[str, str]:
    fused_buggy = float(result["fused_probs"][1])
    confidence = classify_confidence(fused_buggy)
    branch_alignment = describe_branch_alignment(
        float(result["code_probs"][1]),
        float(result["promise_probs"][1]),
    )

    if confidence == "low":
        interpretation = "Weak signal. Treat this as a triage hint, not a confirmed defect."
    elif confidence == "moderate":
        interpretation = "Moderate signal. Review the flagged code before trusting the result."
    else:
        interpretation = "Strong signal. The snippet likely deserves review."

    return {
        "label": result["fused_label"],
        "confidence": confidence,
        "fusion_buggy_probability": f"{fused_buggy:.2%}",
        "semantic_buggy_probability": f"{float(result['code_probs'][1]):.2%}",
        "structural_buggy_probability": f"{float(result['promise_probs'][1]):.2%}",
        "branch_alignment": branch_alignment,
        "alpha": f"{alpha:.2f}",
        "interpretation": interpretation,
    }


def combine_review_priority(
    summary: Dict[str, str],
    llm_review: Dict[str, Any] | None,
) -> Dict[str, str]:
    model_confidence = summary["confidence"]
    llm_risk = "none"

    if llm_review:
        llm_risk = llm_review.get("risk_level", "low")

    if llm_risk == "high":
        priority = "high"
        explanation = "Escalate for human review. The LLM identified a strong risk pattern even if the classifier signal is weak."
    elif llm_risk == "moderate":
        priority = "moderate"
        explanation = "Review recommended. The LLM identified a plausible issue and the model should be treated as supporting evidence."
    elif model_confidence == "high":
        priority = "high"
        explanation = "Escalate for human review. The classifier shows a strong learned risk signal."
    elif model_confidence == "moderate":
        priority = "moderate"
        explanation = "Review recommended. The classifier shows a meaningful but not conclusive signal."
    else:
        priority = "low"
        explanation = "Low immediate priority. Treat this as a weak triage hint unless broader context suggests otherwise."

    return {
        "model_signal": model_confidence,
        "llm_security_reasoning": llm_risk,
        "final_review_priority": priority,
        "priority_explanation": explanation,
    }


def analyze_text(
    text: str,
    alpha: float = 0.5,
    explain_with_llm: bool = False,
    llm_model: str | None = None,
    llm_max_output_tokens: int = 900,
) -> Dict[str, Any]:
    runtime = load_runtime()
    result = predict_fused(
        text=text,
        code_tokenizer=runtime["code_tokenizer"],
        code_model=runtime["code_model"],
        promise_model=runtime["promise_model"],
        promise_scaler=runtime["promise_scaler"],
        promise_feature_cols=runtime["promise_feature_cols"],
        device=runtime["device"],
        alpha=alpha,
    )
    display_summary = summarize_prediction_for_display(result, alpha)

    output = {
        "prediction": result,
        "summary": display_summary,
        "llm_review": None,
        "review_priority": combine_review_priority(display_summary, None),
    }

    if explain_with_llm:
        model_name = llm_model or os.environ.get("OPENAI_MODEL")
        if not model_name:
            raise ValueError(
                "Provide --llm-model or set OPENAI_MODEL when using LLM explanation."
            )

        llm_prompt = build_llm_review_prompt(
            text=text,
            prediction={
                "fused_label": result["fused_label"],
                "fused_pred": result["fused_pred"],
                "fused_probs": result["fused_probs"],
                "code_probs": result["code_probs"],
                "promise_probs": result["promise_probs"],
                "alpha": alpha,
            },
            metrics_row=result["metrics"],
        )
        llm_result = request_llm_explanation(
            prompt=llm_prompt,
            model=model_name,
            max_output_tokens=llm_max_output_tokens,
        )
        output["llm_review"] = parse_llm_review(llm_result["text"])
        output["review_priority"] = combine_review_priority(display_summary, output["llm_review"])

    return output


def request_llm_explanation(
    prompt: str,
    model: str,
    max_output_tokens: int = 900,
) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).rstrip("/")
    payload = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }

    request = urllib.request.Request(
        f"{base_url}{OPENAI_RESPONSES_PATH}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LLM request failed with HTTP {exc.code}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

    return {
        "model": model,
        "text": extract_response_text(response_json),
        "raw_response": response_json,
    }


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
        "metrics": metrics_row,
        "code_probs": code_probs.squeeze(0).cpu().tolist(),
        "promise_probs": promise_probs.squeeze(0).cpu().tolist(),
        "fused_probs": fused_probs.squeeze(0).cpu().tolist(),
        "fused_pred": fused_pred,
        "fused_label": "buggy" if fused_pred == 1 else "clean",
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
    parser.add_argument(
        "--explain-with-llm",
        action="store_true",
        help="After prediction, call an OpenAI-compatible LLM for issue explanation and remediation guidance.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.environ.get("OPENAI_MODEL"),
        help="Model name for the post-prediction LLM call. Defaults to OPENAI_MODEL.",
    )
    parser.add_argument(
        "--llm-max-output-tokens",
        type=int,
        default=1400,
        help="Maximum tokens for the LLM explanation response.",
    )
    args = parser.parse_args()

    # Frontend / single-snippet path
    if args.text is not None:
        analysis = analyze_text(
            text=args.text,
            alpha=args.alpha,
            explain_with_llm=args.explain_with_llm,
            llm_model=args.llm_model,
            llm_max_output_tokens=args.llm_max_output_tokens,
        )
        result = analysis["prediction"]
        code_buggy = result["code_probs"][1]
        promise_buggy = result["promise_probs"][1]
        fusion_buggy = result["fused_probs"][1]

        print(f"CodeBERT: {'buggy' if code_buggy >= 0.5 else 'clean'} ({code_buggy:.2%})")
        print(f"PROMISE:  {'buggy' if promise_buggy >= 0.5 else 'clean'} ({promise_buggy:.2%})")
        print(f"Fusion:   {result['fused_label']} ({fusion_buggy:.2%})")
        display_summary = analysis["summary"]
        print("\nPrediction Summary")
        print(f"Label:              {display_summary['label']}")
        print(f"Confidence:         {display_summary['confidence']}")
        print(f"Fusion buggy prob:  {display_summary['fusion_buggy_probability']}")
        print(f"Semantic branch:    {display_summary['semantic_buggy_probability']}")
        print(f"Structural branch:  {display_summary['structural_buggy_probability']}")
        print(f"Branch alignment:   {display_summary['branch_alignment']}")
        print(f"Interpretation:     {display_summary['interpretation']}")

        if args.explain_with_llm:
            print("\nLLM Review")
            print(json.dumps(analysis["llm_review"], indent=2))
        return

    # Evaluation path
    if not args.paired_csv:
        raise ValueError("Provide either --text for a single snippet or --paired_csv for batch testing.")

    runtime = load_runtime()
    results = evaluate_fusion_csv(
        csv_path=args.paired_csv,
        code_tokenizer=runtime["code_tokenizer"],
        code_model=runtime["code_model"],
        promise_model=runtime["promise_model"],
        promise_scaler=runtime["promise_scaler"],
        promise_feature_cols=runtime["promise_feature_cols"],
        device=runtime["device"],
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
