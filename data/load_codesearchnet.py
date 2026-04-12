from datasets import load_dataset
import json
import os

dataset = load_dataset("code_search_net", "java")

print(dataset["train"][0])

output_dir = "data/codesearchnet"
os.makedirs(output_dir, exist_ok=True)

def save_jsonl(split, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset[split]:
            record = {
                "text": item["func_code_string"]
            }
            f.write(json.dumps(record) + "\n")

save_jsonl("train", os.path.join(output_dir, "train.jsonl"))
save_jsonl("validation", os.path.join(output_dir, "val.jsonl"))
save_jsonl("test", os.path.join(output_dir, "test.jsonl"))

print("Saved CodeSearchNet data")