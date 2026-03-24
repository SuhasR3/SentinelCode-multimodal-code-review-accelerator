import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from src.data.dataset import JsonlCodeDataset
from src.data.collate import CodeCollator


# =========================
# Configuration
# =========================
CHECKPOINT_DIR = "outputs/checkpoints/code-model-best"
TEST_PATH = "data/processed/test.jsonl"

BATCH_SIZE = 4
MAX_LENGTH = 128


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR).to(device)
    model.eval()

    test_dataset = JsonlCodeDataset(TEST_PATH)
    collator = CodeCollator(tokenizer=tokenizer, max_length=MAX_LENGTH)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        total_loss += loss.item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())

    avg_loss = total_loss / max(1, len(test_loader))
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary",
        zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    print("Test results")
    print(f"Loss:      {avg_loss:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
