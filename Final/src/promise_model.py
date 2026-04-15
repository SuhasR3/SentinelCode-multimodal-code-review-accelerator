from __future__ import annotations

import torch
import torch.nn as nn


class PromiseMLPClassifier(nn.Module):
    """
    Structural branch for PROMISE metrics.
    Encodes metrics -> embedding -> logits.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        emb_dim: int = 64,
        num_labels: int = 2,
        dropout: float = 0.15,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h

        self.encoder = nn.Sequential(*layers)
        self.embedding = nn.Linear(prev_dim, emb_dim)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(emb_dim, num_labels)

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        metrics: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        x = self.encoder(metrics)
        emb = self.activation(self.embedding(x))
        logits = self.classifier(emb)

        output = {
            "logits": logits,
            "embedding": emb,
        }

        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)

        return output