import os
import pandas as pd
import pytest

from ml.seed_data import generate_seed_data

def test_seed_data_uniqueness_and_endpoints(tmp_path):
    data_dir = str(tmp_path)
    generate_seed_data(data_dir=data_dir, seed=42)

    s1_df = pd.read_csv(os.path.join(data_dir, "s1.csv"))
    s2_df = pd.read_csv(os.path.join(data_dir, "s2.csv"))
    s3_df = pd.read_csv(os.path.join(data_dir, "s3.csv"))
    links_df = pd.read_csv(os.path.join(data_dir, "links.csv"))

    # Check ID uniqueness within each source
    assert s1_df["entity_id"].is_unique, "s1.csv entity_ids are not unique"
    assert s2_df["entity_id"].is_unique, "s2.csv entity_ids are not unique"
    assert s3_df["entity_id"].is_unique, "s3.csv entity_ids are not unique"

    # Check candidate counts >= 200 combined
    total_cands = len(s2_df) + len(s3_df)
    assert total_cands >= 200, f"Expected S2+S3 >= 200, got {total_cands}"

    # Check link endpoints exist in s1 and s2/s3
    s1_ids = set(s1_df["entity_id"])
    s2_ids = set(s2_df["entity_id"])
    s3_ids = set(s3_df["entity_id"])

    for _, row in links_df.iterrows():
        assert row["s1_id"] in s1_ids, f"Link s1_id {row['s1_id']} not in s1.csv"
        if row["source"] == "s2":
            assert row["other_id"] in s2_ids, f"Link other_id {row['other_id']} not in s2.csv"
        elif row["source"] == "s3":
            assert row["other_id"] in s3_ids, f"Link other_id {row['other_id']} not in s3.csv"
        else:
            pytest.fail(f"Invalid source {row['source']}")

    # Check demo IDs
    s1_001_links = links_df[links_df["s1_id"] == "S1_001"]
    assert len(s1_001_links) == 2, "S1_001 should have 2 true match links"

    s1_002_links = links_df[links_df["s1_id"] == "S1_002"]
    assert len(s1_002_links) == 0, "S1_002 collision demo should have 0 true match links"

    s1_003_links = links_df[links_df["s1_id"] == "S1_003"]
    assert len(s1_003_links) == 0, "S1_003 singleton demo should have 0 true match links"
