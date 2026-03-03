#this code will create a df with 0 and 1 labels for clean and buggy code. The data link is: https://www.kaggle.com/datasets/arunabhaalphadutta/defects4j-extracted-dataset
import os
import pandas as pd
from pathlib import Path

def load_defects4j_kaggle(root_dir):
    dataset = []
    root_path = Path(root_dir)

    # Iterating through each bug directory
    for bug_dir in root_path.iterdir():
        if not bug_dir.is_dir():
            continue
            
        buggy_path = bug_dir / "buggy"
        if buggy_path.exists():
            for java_file in buggy_path.rglob("*.java"):
                with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                    dataset.append({
                        'project': bug_dir.name,
                        'code': f.read(),
                        'label': 1  # Defect-prone
                    })

        fixed_path = bug_dir / "fixed"
        if fixed_path.exists():
            for java_file in fixed_path.rglob("*.java"):
                with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                    dataset.append({
                        'project': bug_dir.name,
                        'code': f.read(),
                        'label': 0  # Clean
                    })

    df = pd.DataFrame(dataset)
    print(f"Successfully loaded {len(df)} samples.")
    return df

df = load_defects4j_kaggle('data/initial_data')
df.head(100).to_csv('data/raw_extracted.csv', index=False) #currently I have set it to 100 for testing purposes