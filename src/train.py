# getting the model and make it ready for the training

from src.models.transformers_classifier import (
    build_model,
    freeze_backbone,
    unfreeze_last_n_layers,
    count_trainable_parameters,
)

# waht we can use for good/bad software detection/quality classification
# 1) "microsoft/codebert-base" (Best for:Defects4J,Bugs.jar,Function-level classification)
# 2) "microsoft/codebert-base-mlm" (best if we want to pretrain before fine tunning)
# 3) "microsoft/graphcodebert-base" (Best for:Bug detection, Code vulnerability classification,When semantic structure matters)
# 4) "huggingface/CodeBERTa-small-v1" ( Best for:Fast experiments, Limited GPU, Quick iteration)
### I guess the best option would be the 4th one + freezing the transformer since we have no GPU to fine tune. with shorter sequence length can be done
### Our choice is dictated by the computational power mostly...correct me if I am wrong

MODEL_NAME = "huggingface/CodeBERTa-small-v1"

# how many classes we have
NUM_LABELS = 2

# Dataset is small → dropout increase (0.2–0.3)
# Dataset is large → dropout default 0.1
DROPOUT = 0.1


model = build_model(
    pretrained_name=MODEL_NAME,
    num_labels=NUM_LABELS,
    dropout=DROPOUT
)

# =========================
# Added below: actual training pipeline
# =========================

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data.dataset import JsonlCodeDataset
from src.data.collate import CodeCollator


# paths
TRAIN_PATH = "data/processed/train.jsonl"
VAL_PATH = "data/processed/val.jsonl"
OUTPUT_DIR = "outputs/checkpoints/code-model-best"

# training hyperparameters
BATCH_SIZE = 4
MAX_LENGTH = 128
EPOCHS = 3
LR = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

# freezing policy
FREEZE_ENCODER = True
UNFREEZE_LAST_N = 0  # set to 1 later if you want to unfreeze the last transformer block


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    global model

    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # optionally freeze encoder for CPU / low-compute setup
    if FREEZE_ENCODER:
        freeze_backbone(model)

    # optionally unfreeze the last n transformer blocks
    if UNFREEZE_LAST_N > 0:
        unfreeze_last_n_layers(model, n=UNFREEZE_LAST_N)

    trainable_params, total_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    model = model.to(device)

    train_dataset = JsonlCodeDataset(TRAIN_PATH)
    val_dataset = JsonlCodeDataset(VAL_PATH)

    collator = CodeCollator(tokenizer=tokenizer, max_length=MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
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

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
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

        print("\nValidation results")
        print(f"Epoch:      {epoch + 1}/{EPOCHS}")
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
    print(f"Best validation F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()