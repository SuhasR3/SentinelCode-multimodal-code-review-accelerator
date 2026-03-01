"""
PROMISE Dataset Loader & Preprocessing Pipeline
-----------------------------------------------
Steps implemented:
1. Download KC1, PC1, JM1 from OpenML
2. Convert labels to binary (buggy=1, clean=0)
3. Merge datasets
4. Remove duplicates
5. Median imputation for missing values
6. Stratified 70/15/15 split
7. Save processed datasets
"""

import os
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# ---------------------------------------------------
# STEP 1 — LOAD SINGLE PROMISE DATASET
# ---------------------------------------------------
def load_promise_dataset(name: str) -> pd.DataFrame:
    print(f"\nLoading PROMISE dataset: {name}")

    data = fetch_openml(name=name, as_frame=True)
    df = data.frame.copy()

    # ---- Find defect label column automatically ----
    label_col = None
    for col in df.columns:
        if col.lower() in ["defects", "defect", "bug", "class"]:
            label_col = col
            break

    if label_col is None:
        raise ValueError(f"No defect label column found in {name}")

    # ---- Convert to binary labels ----
    def to_binary(x):
        if pd.isna(x):
            return 0
        if isinstance(x, (int, float)):
            return 1 if x > 0 else 0
        x_str = str(x).lower().strip()
        if x_str in ("true", "yes", "y", "1"):
            return 1
        if x_str in ("false", "no", "n", "0"):
            return 0
        try:
            return 1 if float(x_str) > 0 else 0
        except ValueError:
            return 0

    df["label"] = df[label_col].apply(to_binary)

    # Remove original label column
    df.drop(columns=[label_col], inplace=True)

    # Add module identifier
    df["module_id"] = [f"{name}_{i}" for i in range(len(df))]

    print(f"{name} loaded: {len(df)} samples")
    return df


# ---------------------------------------------------
# STEP 2 — LOAD ALL PROMISE DATASETS
# ---------------------------------------------------
def load_all_promise():
    kc1 = load_promise_dataset("kc1")
    pc1 = load_promise_dataset("pc1")
    jm1 = load_promise_dataset("jm1")

    promise_df = pd.concat([kc1, pc1, jm1], ignore_index=True)

    print("\nCombined dataset shape:", promise_df.shape)
    return promise_df


# ---------------------------------------------------
# STEP 3 — CLEAN DATA
# ---------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\nCleaning dataset...")

    # Remove duplicates
    df = df.drop_duplicates()

    # Median imputation for numeric columns
    numeric_cols = df.select_dtypes(include="number").columns

    df[numeric_cols] = df[numeric_cols].fillna(
        df[numeric_cols].median()
    )

    print("Cleaning complete.")
    return df


# ---------------------------------------------------
# STEP 4 — STRATIFIED SPLIT (70/15/15)
# ---------------------------------------------------
def stratified_split(df: pd.DataFrame):

    print("\nPerforming stratified split...")

    X = df.drop(columns=["label"])
    y = df["label"]

    # 70% train
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    # 15% validation / 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print("Split sizes:")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    return train_df, val_df, test_df


# ---------------------------------------------------
# STEP 5 — SAVE DATASETS
# ---------------------------------------------------
def save_datasets(raw_df, train_df, val_df, test_df):

    os.makedirs("data", exist_ok=True)

    raw_df.to_csv("data/promise_raw.csv", index=False)
    train_df.to_csv("data/promise_train.csv", index=False)
    val_df.to_csv("data/promise_val.csv", index=False)
    test_df.to_csv("data/promise_test.csv", index=False)

    print("\nDatasets saved to /data folder ✅")


# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------
if __name__ == "__main__":

    print("Starting PROMISE preprocessing pipeline...")

    # Load
    promise_df = load_all_promise()

    # Clean
    promise_df = clean_data(promise_df)

    # Split
    train_df, val_df, test_df = stratified_split(promise_df)

    # Save
    save_datasets(promise_df, train_df, val_df, test_df)

    print("\n✅ PROMISE pipeline completed successfully.")