import json
from pathlib import Path
from typing import Any, Dict, List

from torch.utils.data import Dataset


class CodeReviewDataset(Dataset):
    """
    Dataset for JSONL files with the format:
    {"text": "...source code...", "label": 0}
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        self.samples = self._load_jsonl(self.file_path)

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in: {self.file_path}")

    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        with file_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} in {file_path}: {e}"
                    ) from e

                if "text" not in record:
                    raise KeyError(
                        f"Missing 'text' field on line {line_number} in {file_path}"
                    )
                if "label" not in record:
                    raise KeyError(
                        f"Missing 'label' field on line {line_number} in {file_path}"
                    )

                text = record["text"]
                label = record["label"]

                if not isinstance(text, str):
                    raise TypeError(
                        f"'text' must be a string on line {line_number} in {file_path}"
                    )

                if text.strip() == "":
                    continue

                try:
                    label = int(label)
                except (TypeError, ValueError) as e:
                    raise TypeError(
                        f"'label' must be convertible to int on line {line_number} in {file_path}"
                    ) from e

                samples.append(
                    {
                        "text": text,
                        "label": label,
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
