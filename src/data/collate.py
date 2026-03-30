from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer


class CodeBERTCollator:
    """
    Collate function that tokenizes raw code samples and builds a batch.

    Input samples:
        [{"text": "...", "label": 0}, ...]

    Output batch:
        {
            "input_ids": ...,
            "attention_mask": ...,
            "labels": ...
        }
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        max_length: int = 256,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]

        tokenized = self.tokenizer(
            texts,
            padding=True,              # dynamic padding per batch
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        tokenized["labels"] = torch.tensor(labels, dtype=torch.long)
        return tokenized
