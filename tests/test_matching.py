import json

import pandas as pd
import pytest

import app.matching as matching
from app.matching import list_entities, get_matches


TRUE_MATCH_ID = "S1_001"
COLLISION_ID = "S1_002"
SINGLETON_ID = "S1_003"


# ---------------------------------------------------------
# Basic contract
# ---------------------------------------------------------

def test_list_entities():

    entities = list_entities()

    assert len(entities) == 100

    assert all(
        "entity_id" in entity
        and "business_name" in entity
        for entity in entities
    )


# ---------------------------------------------------------
# True match
# ---------------------------------------------------------

def test_true_match():

    result = get_matches(TRUE_MATCH_ID)

    assert result["s1"]["entity_id"] == TRUE_MATCH_ID

    assert result["accepted_ids"] == [
        "S2_001",
        "S3_001",
    ]

    candidate_ids = {
        candidate["candidate_id"]
        for candidate in result["candidates"]
    }

    assert "S2_001" in candidate_ids
    assert "S3_001" in candidate_ids


# ---------------------------------------------------------
# Candidate contract
# ---------------------------------------------------------

def test_candidate_contract():

    result = get_matches(TRUE_MATCH_ID)

    required_candidate_fields = {
        "candidate_id",
        "source",
        "business_name",
        "business_address",
        "country",
        "score",
        "accepted",
        "features",
    }

    required_features = {
        "name_sim",
        "addr_sim",
        "exact_name",
        "country_match",
        "len_diff",
    }

    assert result["candidates"]

    for candidate in result["candidates"]:

        assert required_candidate_fields.issubset(
            candidate.keys()
        )

        assert required_features.issubset(
            candidate["features"].keys()
        )

        assert isinstance(
            candidate["score"],
            (int, float)
        )

        assert candidate["source"] in {
            "s2",
            "s3",
        }


# ---------------------------------------------------------
# Score sorting
# ---------------------------------------------------------

def test_candidates_sorted_descending():

    result = get_matches(TRUE_MATCH_ID)

    scores = [
        candidate["score"]
        for candidate in result["candidates"]
    ]

    assert scores == sorted(
        scores,
        reverse=True
    )


# ---------------------------------------------------------
# Threshold
# ---------------------------------------------------------

def test_threshold_matches_metrics():

    with open(
        matching.METRICS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        metrics = json.load(file)

    result = get_matches(TRUE_MATCH_ID)

    assert result["threshold"] == metrics["threshold"]


# ---------------------------------------------------------
# Collision
# ---------------------------------------------------------

def test_collision_case():

    result = get_matches(COLLISION_ID)

    assert result["s1"]["entity_id"] == COLLISION_ID

    candidate_ids = {
        candidate["candidate_id"]
        for candidate in result["candidates"]
    }

    # Both same-name collision records should survive blocking.
    assert "S2_002" in candidate_ids
    assert "S3_002" in candidate_ids

    # Neither should be accepted.
    assert "S2_002" not in result["accepted_ids"]
    assert "S3_002" not in result["accepted_ids"]


# ---------------------------------------------------------
# Singleton
# ---------------------------------------------------------

def test_singleton_case():

    result = get_matches(SINGLETON_ID)

    assert result["s1"]["entity_id"] == SINGLETON_ID

    assert isinstance(
        result["candidates"],
        list
    )

    assert result["accepted_ids"] == []


# ---------------------------------------------------------
# Unknown ID
# ---------------------------------------------------------

def test_unknown_id():

    with pytest.raises(
        ValueError,
        match="Unknown S1 ID"
    ):
        get_matches("S1_DOES_NOT_EXIST")


# ---------------------------------------------------------
# Zero candidate behavior
# ---------------------------------------------------------

def test_zero_candidate_behavior(monkeypatch):

    def fake_blocker(
        s1_row,
        candidates_df,
        max_candidates=20
    ):
        return pd.DataFrame()

    monkeypatch.setattr(
        matching,
        "block_candidates_for_s1",
        fake_blocker
    )

    result = get_matches(TRUE_MATCH_ID)

    assert result["candidates"] == []
    assert result["accepted_ids"] == []
    assert isinstance(
        result["threshold"],
        (int, float)
    )


# ---------------------------------------------------------
# Model-backed behavior
# ---------------------------------------------------------

def test_model_is_used():

    _, _, model, _, _ = matching._load_runtime()

    assert hasattr(
        model,
        "predict_proba"
    )


# ---------------------------------------------------------
# Feature order
# ---------------------------------------------------------

def test_saved_feature_order():

    _, _, _, feature_names, _ = (
        matching._load_runtime()
    )

    assert feature_names == [
        "name_sim",
        "addr_sim",
        "exact_name",
        "country_match",
        "len_diff",
    ]