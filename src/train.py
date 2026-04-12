"""
Fine-tune a CodeBERT classifier using the CodeSearchNet-pretrained checkpoint.

This script:
- loads the pretrained MLM checkpoint from outputs/pretrained_codesearchnet
- initializes a new classification head
- trains on binary buggy/clean labels
- saves the best model checkpoint
"""

import os
import time
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data.dataset import JsonlCodeDataset
from src.data.collate import CodeCollator
from src.models.transformers_classifier import (
    freeze_backbone,
    unfreeze_last_n_layers,
    count_trainable_parameters,
)

# =========================
# Model / data config
# =========================
MODEL_NAME = "outputs/pretrained_codesearchnet"
NUM_LABELS = 2
DROPOUT = 0.1

TRAIN_PATH = "data/processed/train.jsonl"
VAL_PATH = "data/processed/val.jsonl"
OUTPUT_DIR = "outputs/checkpoints/code-model-best"

# training hyperparameters
BATCH_SIZE = 8
MAX_LENGTH = 256
EPOCHS = 5
LR = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

# freezing policy
FREEZE_ENCODER = False
UNFREEZE_LAST_N = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset) -> torch.Tensor:
    """Compute inverse frequency weights for imbalanced classification."""
    labels = [sample["label"] for sample in dataset]
    label_counts = Counter(labels)

    if len(label_counts) < 2:
        raise ValueError(
            f"Training set must contain both classes. Found label distribution: {dict(label_counts)}"
        )

    total = len(labels)
    num_classes = NUM_LABELS

    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weights.append(total / (num_classes * count))

    weights = torch.tensor(weights, dtype=torch.float32)
    print(f"Class distribution: {dict(label_counts)}")
    print(f"Class weights: {weights.tolist()}")
    return weights


@torch.no_grad()
def evaluate_model(current_model, dataloader, device):
    current_model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = current_model(**batch)

        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        total_loss += loss.item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())

    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary",
        zero_division=0,
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_NAME):
        raise FileNotFoundError(
            f"Pretrained checkpoint folder not found: {MODEL_NAME}"
        )

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Train file not found: {TRAIN_PATH}")

    if not os.path.exists(VAL_PATH):
        raise FileNotFoundError(f"Validation file not found: {VAL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Load classification model from MLM-pretrained checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )

    # Optional freezing
    if FREEZE_ENCODER:
        freeze_backbone(model)

    if UNFREEZE_LAST_N > 0:
        unfreeze_last_n_layers(model, n=UNFREEZE_LAST_N)

    trainable_params, total_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    model = model.to(device)

    train_dataset = JsonlCodeDataset(TRAIN_PATH)
    val_dataset = JsonlCodeDataset(VAL_PATH)

    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    collator = CodeCollator(tokenizer=tokenizer, max_length=MAX_LENGTH)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        pin_memory=pin_memory,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        pin_memory=pin_memory,
        num_workers=2,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    total_training_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_val_f1 = -1.0
    training_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)

            logits = outputs.logits
            loss = criterion(logits, batch["labels"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if step % 10 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"Batch Loss: {loss.item():.4f}"
                )

        avg_train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate_model(model, val_loader, device)
        epoch_elapsed = time.time() - epoch_start

        print("\nValidation results")
        print(f"Epoch:      {epoch + 1}/{EPOCHS}")
        print(f"Epoch Time: {epoch_elapsed / 60:.1f} min")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {val_metrics['loss']:.4f}")
        print(f"Accuracy:   {val_metrics['accuracy']:.4f}")
        print(f"Precision:  {val_metrics['precision']:.4f}")
        print(f"Recall:     {val_metrics['recall']:.4f}")
        print(f"F1:         {val_metrics['f1']:.4f}\n")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"Saved new best model to: {OUTPUT_DIR}")

    print("Training complete.")
    total_elapsed = time.time() - training_start
    print(f"Total training time: {total_elapsed / 60:.1f} min")
    print(f"Best validation F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
