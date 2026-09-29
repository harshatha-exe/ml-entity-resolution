import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, confusion_matrix
)

from ml.features import FEATURE_NAMES, prepare_featured_dataset

def evaluate_and_save_metrics(
    data_dir: str = "data",
    model_path: str = "ml/model.joblib",
    metrics_path: str = "ml/metrics.json"
) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Run ml.train first.")

    artifact = joblib.load(model_path)
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    selected_model_name = artifact.get("model_type", type(model).__name__) if isinstance(artifact, dict) else type(model).__name__

    s1_df = pd.read_csv(os.path.join(data_dir, "s1.csv"))
    s2_df = pd.read_csv(os.path.join(data_dir, "s2.csv"))
    s3_df = pd.read_csv(os.path.join(data_dir, "s3.csv"))
    links_df = pd.read_csv(os.path.join(data_dir, "links.csv"))

    cands_val = pd.read_csv(os.path.join(data_dir, "candidates_val.csv"))
    cands_test = pd.read_csv(os.path.join(data_dir, "candidates_test.csv"))

    val_featured = prepare_featured_dataset(cands_val, s1_df, s2_df, s3_df, links_df)
    test_featured = prepare_featured_dataset(cands_test, s1_df, s2_df, s3_df, links_df)

    X_val = val_featured[FEATURE_NAMES]
    y_val = val_featured["target"]

    X_test = test_featured[FEATURE_NAMES]
    y_test = test_featured["target"]

    # Predict probabilities if supported, else decision function or binary predict
    if hasattr(model, "predict_proba"):
        val_probs = model.predict_proba(X_val)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
    else:
        val_probs = model.predict(X_val).astype(float)
        test_probs = model.predict(X_test).astype(float)

    # Threshold selection using validation set F0.5
    best_thresh = 0.5
    best_val_f05 = -1.0
    threshold_grid = np.linspace(0.1, 0.9, 81)

    for thresh in threshold_grid:
        preds = (val_probs >= thresh).astype(int)
        f05 = fbeta_score(y_val, preds, beta=0.5, zero_division=0)
        if f05 > best_val_f05:
            best_val_f05 = f05
            best_thresh = float(thresh)

    # Compute validation metrics at best_thresh
    val_preds = (val_probs >= best_thresh).astype(int)
    val_cm = confusion_matrix(y_val, val_preds).tolist()
    val_metrics = {
        "accuracy": float(accuracy_score(y_val, val_preds)),
        "precision": float(precision_score(y_val, val_preds, zero_division=0)),
        "recall": float(recall_score(y_val, val_preds, zero_division=0)),
        "f1": float(f1_score(y_val, val_preds, zero_division=0)),
        "f0_5": float(fbeta_score(y_val, val_preds, beta=0.5, zero_division=0)),
        "confusion_matrix": val_cm
    }

    # Evaluate on untouched test set at frozen best_thresh
    test_preds = (test_probs >= best_thresh).astype(int)
    test_cm = confusion_matrix(y_test, test_preds).tolist()

    test_featured["pred"] = test_preds

    # Per S1 entity evaluation
    per_s1_scores = {}
    for s1_id, group in test_featured.groupby("s1_id"):
        y_g = group["target"]
        p_g = group["pred"]
        per_s1_scores[s1_id] = float(fbeta_score(y_g, p_g, beta=0.5, zero_division=0))

    # Singleton error rate on test set (S1 entities with 0 true links)
    test_links_s1 = set(links_df[links_df["s1_id"].isin(test_featured["s1_id"])]["s1_id"])
    singleton_s1s = set(test_featured["s1_id"].unique()) - test_links_s1
    singleton_errors = 0
    for s1_id in singleton_s1s:
        s1_preds = test_featured[test_featured["s1_id"] == s1_id]["pred"]
        if (s1_preds == 1).any():
            singleton_errors += 1
    singleton_error_rate = float(singleton_errors / len(singleton_s1s)) if singleton_s1s else 0.0

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, test_preds)),
        "precision": float(precision_score(y_test, test_preds, zero_division=0)),
        "recall": float(recall_score(y_test, test_preds, zero_division=0)),
        "f1": float(f1_score(y_test, test_preds, zero_division=0)),
        "f0_5": float(fbeta_score(y_test, test_preds, beta=0.5, zero_division=0)),
        "confusion_matrix": test_cm,
        "mean_per_s1_f0_5": float(np.mean(list(per_s1_scores.values()))) if per_s1_scores else 0.0,
        "singleton_error_rate": singleton_error_rate
    }

    metrics_payload = {
        "selected_model": selected_model_name,
        "threshold": round(best_thresh, 4),
        "feature_names": FEATURE_NAMES,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics
    }

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nEvaluation Complete:")
    print(f"  Selected Model  : {selected_model_name}")
    print(f"  Optimal Threshold: {best_thresh:.4f} (selected on Validation Set F0.5)")
    print(f"  Validation F0.5 : {val_metrics['f0_5']:.4f}")
    print(f"  Untouched Test F0.5: {test_metrics['f0_5']:.4f} (Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f})")
    print(f"  Metrics saved to {metrics_path}")

    return metrics_payload

if __name__ == "__main__":
    evaluate_and_save_metrics()
