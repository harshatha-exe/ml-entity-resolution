from app.matching import list_entities, get_matches


# Demo IDs selected from the current dataset.
TRUE_MATCH_ID = "S1_001"
COLLISION_ID = "S1_006"
SINGLETON_ID = "S1_003"


def test_list_entities():
    entities = list_entities()

    assert len(entities) == 100

    assert all(
        "entity_id" in entity and
        "business_name" in entity
        for entity in entities
    )


def test_true_match_contract():
    result = get_matches(TRUE_MATCH_ID)

    assert result["s1"]["entity_id"] == TRUE_MATCH_ID
    assert isinstance(result["threshold"], (int, float))
    assert isinstance(result["candidates"], list)
    assert isinstance(result["accepted_ids"], list)

    # S1_001 has known links to S2_001 and S3_001.
    candidate_ids = {
        candidate["candidate_id"]
        for candidate in result["candidates"]
    }

    assert "S2_001" in candidate_ids
    assert "S3_001" in candidate_ids


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
        assert required_candidate_fields.issubset(candidate.keys())
        assert required_features.issubset(
            candidate["features"].keys()
        )

        assert isinstance(candidate["score"], (int, float))
        assert candidate["source"] in {"s2", "s3"}


def test_collision_case():
    result = get_matches(COLLISION_ID)

    assert result["s1"]["entity_id"] == COLLISION_ID

    # The linked candidate exists and should be returned.
    candidate_ids = {
        candidate["candidate_id"]
        for candidate in result["candidates"]
    }

    assert "S2_006" in candidate_ids


def test_singleton_case():
    result = get_matches(SINGLETON_ID)

    assert result["s1"]["entity_id"] == SINGLETON_ID

    # The matcher should still return candidates for a singleton.
    assert isinstance(result["candidates"], list)

    # No known ground-truth links exist for S1_003.
    assert result["accepted_ids"] == []