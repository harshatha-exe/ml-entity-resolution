import os
import pandas as pd
import pytest

from ml.normalize import normalize_name, strip_company_suffixes, normalize_address, clean_text
from ml.split import split_s1_ids
from ml.seed_data import generate_seed_data
from ml.blocking import run_blocking

def test_normalization():
    assert normalize_name("Acme Global Solutions Inc.") == "acme global solutions inc"
    assert strip_company_suffixes("Acme Global Solutions Inc.") == "acme global solutions"
    assert normalize_address("123 Innovation Way Ste 400") == "123 innovation way suite 400"
    assert normalize_address("300 Doctors Dr Apt 2B") == "300 doctors drive apartment 2b"

def test_s1_split_exclusivity():
    s1_ids = [f"S1_{i:03d}" for i in range(1, 101)]
    split_map, train_ids, val_ids, test_ids = split_s1_ids(s1_ids, seed=42)

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    # Check pair-wise intersections are empty
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0
    assert len(train_set) + len(val_set) + len(test_set) == 100

def test_blocking_and_demo_ids(tmp_path):
    data_dir = str(tmp_path)
    generate_seed_data(data_dir=data_dir, seed=42)

    recall_results, train_cands, val_cands, test_cands = run_blocking(data_dir=data_dir)

    all_cands = pd.concat([train_cands, val_cands, test_cands], ignore_index=True)

    # Demo 1: S1_001 (Match Demo) -> Should include S2_001 and S3_001
    s1_001_cands = set(all_cands[all_cands["s1_id"] == "S1_001"]["candidate_id"].tolist())
    assert "S2_001" in s1_001_cands
    assert "S3_001" in s1_001_cands

    # Demo 2: S1_002 (Collision Demo) -> Should block same-name candidates S2_002, S3_002
    s1_002_cands = set(all_cands[all_cands["s1_id"] == "S1_002"]["candidate_id"].tolist())
    assert "S2_002" in s1_002_cands or "S3_002" in s1_002_cands

    # Demo 3: S1_003 (Singleton Demo) -> Has candidate pairs generated, but ground truth links in links.csv is empty
    links_df = pd.read_csv(os.path.join(data_dir, "links.csv"))
    s1_003_links = links_df[links_df["s1_id"] == "S1_003"]
    assert len(s1_003_links) == 0

    # Recall check (target >= 85%)
    for split_name, recall in recall_results.items():
        assert recall >= 0.85, f"Recall for {split_name} was {recall:.2%}, expected >= 85%"
