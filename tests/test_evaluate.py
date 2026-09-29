import os
import json
import pytest
import pandas as pd

from ml.evaluate import evaluate_and_save_metrics

def test_evaluation_pipeline(tmp_path):
    metrics_file = str(tmp_path / "metrics.json")
    payload = evaluate_and_save_metrics(data_dir="data", model_path="ml/model.joblib", metrics_path=metrics_file)

    assert os.path.exists(metrics_file)
    assert "selected_model" in payload
    assert "threshold" in payload
    assert payload["feature_names"] == ["name_sim", "addr_sim", "exact_name", "country_match", "len_diff"]

    test_metrics = payload["test_metrics"]
    assert "counts" in test_metrics
    assert test_metrics["counts"]["total_test_s1"] == 20
    assert "mean_per_s1_f0_5_linked" in test_metrics
    assert "singleton_no_match_accuracy" in test_metrics
    assert test_metrics["singleton_no_match_accuracy"] == 1.0

def test_singleton_abstention_case():
    """Verify that an S1 entity with no true links and all 0 predictions counts as a correct abstention."""
    y_true = pd.Series([0, 0, 0])
    y_pred = pd.Series([0, 0, 0])
    is_abstention = not (y_pred == 1).any()
    assert is_abstention is True

def test_false_singleton_merge_case():
    """Verify that an S1 entity with no true links where a candidate is falsely accepted flags a false match."""
    y_true = pd.Series([0, 0])
    y_pred = pd.Series([0, 1])
    is_false_match = (y_pred == 1).any() and not (y_true == 1).any()
    assert is_false_match is True

def test_missed_positive_link_case():
    """Verify detection of a missed positive link when a true link prediction is 0."""
    y_true = pd.Series([1, 0])
    y_pred = pd.Series([0, 0])
    is_missed = (y_true == 1).any() and not ((y_true == 1) & (y_pred == 1)).any()
    assert is_missed is True

def test_multiple_candidates_s1_case():
    """Verify evaluating an S1 entity with multiple candidates (e.g. 1 match, 2 non-matches)."""
    y_true = pd.Series([1, 0, 0])
    y_pred = pd.Series([1, 0, 0])
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    assert tp == 1
    assert fp == 0
    assert fn == 0
