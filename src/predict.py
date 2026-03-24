import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================
# Configuration
# =========================
CHECKPOINT_DIR = "outputs/checkpoints/code-model-best"
MAX_LENGTH = 128


@torch.no_grad()
def predict_texts(texts, checkpoint_dir=CHECKPOINT_DIR, max_length=MAX_LENGTH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
    model.eval()

    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    outputs = model(**encodings)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(probabilities, dim=-1)

    results = []
    for text, pred, probs in zip(texts, predictions.cpu().tolist(), probabilities.cpu().tolist()):
        results.append({
            "text": text,
            "predicted_label": pred,
            "probabilities": probs,
        })

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on one or more code snippets.")
    parser.add_argument(
        "--text",
        type=str,
        nargs="+",
        required=True,
        help="One or more code snippets to classify."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT_DIR,
        help="Path to a fine-tuned checkpoint directory."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=MAX_LENGTH,
        help="Maximum tokenized sequence length."
    )

    args = parser.parse_args()

    results = predict_texts(
        texts=args.text,
        checkpoint_dir=args.checkpoint,
        max_length=args.max_length,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
