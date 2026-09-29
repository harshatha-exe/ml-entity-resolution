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

    if hasattr(model, "predict_proba"):
        val_probs = model.predict_proba(X_val)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
    else:
        val_probs = model.predict(X_val).astype(float)
        test_probs = model.predict(X_test).astype(float)

    # Optimal threshold selection on validation set F0.5
    best_thresh = 0.5
    best_val_f05 = -1.0
    threshold_grid = np.linspace(0.1, 0.9, 81)

    for thresh in threshold_grid:
        preds = (val_probs >= thresh).astype(int)
        f05 = fbeta_score(y_val, preds, beta=0.5, zero_division=0)
        if f05 > best_val_f05:
            best_val_f05 = f05
            best_thresh = float(thresh)

    # Compute validation metrics
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

    # Evaluate test set
    test_preds = (test_probs >= best_thresh).astype(int)
    test_cm = confusion_matrix(y_test, test_preds).tolist()
    test_featured["pred"] = test_preds

    # Detailed per-S1 entity metrics analysis
    from ml.split import split_s1_ids
    _, _, _, test_s1_ids = split_s1_ids(s1_df["entity_id"].tolist(), seed=42, data_dir=data_dir)

    links_test_s1 = set(links_df[links_df["s1_id"].isin(test_s1_ids)]["s1_id"])
    no_link_test_s1s = set(test_s1_ids) - links_test_s1
    linked_test_s1s = set(test_s1_ids).intersection(links_test_s1)

    correct_abstentions = 0
    false_matches = 0
    for sid in no_link_test_s1s:
        preds_for_sid = test_featured[test_featured["s1_id"] == sid]["pred"]
        if len(preds_for_sid) == 0 or not (preds_for_sid == 1).any():
            correct_abstentions += 1
        else:
            false_matches += 1

    correct_accepted_links = 0
    missed_links = 0
    per_s1_f05_linked = []

    for sid in linked_test_s1s:
        group = test_featured[test_featured["s1_id"] == sid]
        if len(group) == 0:
            missed_links += 1
            per_s1_f05_linked.append(0.0)
            continue
        y_g = group["target"]
        p_g = group["pred"]
        score = float(fbeta_score(y_g, p_g, beta=0.5, zero_division=0))
        per_s1_f05_linked.append(score)

        if ((y_g == 1) & (p_g == 1)).any():
            correct_accepted_links += 1
        else:
            missed_links += 1

    total_no_link_s1s = len(no_link_test_s1s)
    total_linked_s1s = len(linked_test_s1s)

    singleton_accuracy = float(correct_abstentions / total_no_link_s1s) if total_no_link_s1s > 0 else 1.0
    singleton_false_acc_rate = float(false_matches / total_no_link_s1s) if total_no_link_s1s > 0 else 0.0

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, test_preds)),
        "precision": float(precision_score(y_test, test_preds, zero_division=0)),
        "recall": float(recall_score(y_test, test_preds, zero_division=0)),
        "f1": float(f1_score(y_test, test_preds, zero_division=0)),
        "f0_5": float(fbeta_score(y_test, test_preds, beta=0.5, zero_division=0)),
        "confusion_matrix": test_cm,
        "counts": {
            "total_test_s1": len(test_s1_ids),
            "total_linked_s1": total_linked_s1s,
            "total_no_link_s1": total_no_link_s1s,
            "correct_abstentions": correct_abstentions,
            "false_matches": false_matches,
            "correct_accepted_links": correct_accepted_links,
            "missed_links": missed_links
        },
        "mean_per_s1_f0_5_linked": float(np.mean(per_s1_f05_linked)) if per_s1_f05_linked else 0.0,
        "linked_s1_denominator": total_linked_s1s,
        "singleton_no_match_accuracy": singleton_accuracy,
        "singleton_false_acceptance_rate": singleton_false_acc_rate,
        "singleton_no_link_denominator": total_no_link_s1s
    }

    metrics_payload = {
        "selected_model": selected_model_name,
        "threshold": round(best_thresh, 4),
        "feature_names": FEATURE_NAMES,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "notes": "Per-S1 F0.5 is calculated exclusively on S1 entities with at least 1 true positive link. No-match singletons are evaluated separately using no-match accuracy and false acceptance rate."
    }

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nEvaluation Complete:")
    print(f"  Selected Model  : {selected_model_name}")
    print(f"  Optimal Threshold: {best_thresh:.4f}")
    print(f"  Validation F0.5 : {val_metrics['f0_5']:.4f}")
    print(f"  Test Pair F0.5  : {test_metrics['f0_5']:.4f} (Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f})")
    print(f"  Mean Linked Per-S1 F0.5: {test_metrics['mean_per_s1_f0_5_linked']:.4f} (denom: {total_linked_s1s})")
    print(f"  Singleton No-Match Acc  : {singleton_accuracy:.4f} (denom: {total_no_link_s1s})")
    print(f"  Metrics saved to {metrics_path}")

    return metrics_payload

if __name__ == "__main__":
    evaluate_and_save_metrics()
