import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, classification_report

from ml.features import FEATURE_NAMES, prepare_featured_dataset

def train_and_save_model(data_dir: str = "data", model_path: str = "ml/model.joblib") -> str:
    s1_df = pd.read_csv(os.path.join(data_dir, "s1.csv"))
    s2_df = pd.read_csv(os.path.join(data_dir, "s2.csv"))
    s3_df = pd.read_csv(os.path.join(data_dir, "s3.csv"))
    links_df = pd.read_csv(os.path.join(data_dir, "links.csv"))

    cands_train = pd.read_csv(os.path.join(data_dir, "candidates_train.csv"))
    cands_val = pd.read_csv(os.path.join(data_dir, "candidates_val.csv"))

    train_featured = prepare_featured_dataset(cands_train, s1_df, s2_df, s3_df, links_df)
    val_featured = prepare_featured_dataset(cands_val, s1_df, s2_df, s3_df, links_df)

    X_train = train_featured[FEATURE_NAMES]
    y_train = train_featured["target"]

    X_val = val_featured[FEATURE_NAMES]
    y_val = val_featured["target"]

    # Verify both classes exist in train and val
    train_classes = set(y_train.unique())
    val_classes = set(y_val.unique())

    print(f"Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"Val class distribution  : {y_val.value_counts().to_dict()}")

    if len(train_classes) < 2 or len(val_classes) < 2:
        raise ValueError("Error: Both classes (0 and 1) must be present in train and validation sets!")

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, random_state=42))
        ]),
        "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    }

    best_name = None
    best_model = None
    best_f05 = -1.0

    print("\n--- Model Evaluation on Validation Set ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        val_f05 = fbeta_score(y_val, val_preds, beta=0.5, zero_division=0)
        print(f"Model: {name:20s} | Val F0.5 Score: {val_f05:.4f}")

        if val_f05 > best_f05:
            best_f05 = val_f05
            best_name = name
            best_model = model

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # Save dictionary with model and feature names for robustness
    model_artifact = {
        "model": best_model,
        "feature_names": FEATURE_NAMES,
        "model_type": best_name,
        "val_f05": best_f05
    }
    joblib.dump(model_artifact, model_path)
    print(f"\nSaved best model '{best_name}' (F0.5 = {best_f05:.4f}) to {model_path}")
    return best_name

if __name__ == "__main__":
    train_and_save_model()
