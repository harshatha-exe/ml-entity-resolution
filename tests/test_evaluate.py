import os
import json
import pytest

from ml.evaluate import evaluate_and_save_metrics

def test_evaluation_pipeline(tmp_path):
    metrics_file = str(tmp_path / "metrics.json")
    res = evaluate_and_save_metrics(data_dir="data", model_path="ml/model.joblib", metrics_path=metrics_file)

    assert os.path.exists(metrics_file)
    with open(metrics_file, "r") as f:
        payload = json.load(f)

    assert "selected_model" in payload
    assert "threshold" in payload
    assert "feature_names" in payload
    assert payload["feature_names"] == ["name_sim", "addr_sim", "exact_name", "country_match", "len_diff"]
    assert "val_metrics" in payload
    assert "test_metrics" in payload
    assert "f0_5" in payload["test_metrics"]
    assert "singleton_error_rate" in payload["test_metrics"]
