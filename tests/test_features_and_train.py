import os
import joblib
import pandas as pd
import pytest

from ml.features import extract_pair_features, FEATURE_NAMES
from ml.train import train_and_save_model

def test_feature_extraction():
    feats = extract_pair_features(
        s1_name="Acme Global Solutions", s1_addr="123 Innovation Way Suite 400", s1_country="USA",
        cand_name="Acme Global Solutions Inc.", cand_addr="123 Innovation Way Ste 400", cand_country="USA"
    )
    assert list(feats.keys()) == FEATURE_NAMES
    assert feats["name_sim"] > 0.8
    assert feats["addr_sim"] > 0.8
    assert feats["country_match"] == 1.0

def test_model_training_and_loading(tmp_path):
    model_file = str(tmp_path / "model.joblib")
    best_name = train_and_save_model(data_dir="data", model_path=model_file)

    assert os.path.exists(model_file)
    artifact = joblib.load(model_file)
    assert "model" in artifact
    assert artifact["feature_names"] == FEATURE_NAMES
    assert artifact["model_type"] == best_name
