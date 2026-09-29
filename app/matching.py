"""
Entity matching interface.

Issue #1 implementation:
- Reads the project CSV files.
- Provides list_entities().
- Provides get_matches(s1_id).
- Uses a temporary rule-based scoring stub.

The scoring implementation will be replaced/refined by the ML
implementation in the later matching issue.
"""

from pathlib import Path

import pandas as pd
from rapidfuzz.fuzz import ratio


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

S1_FILE = DATA_DIR / "s1.csv"
S2_FILE = DATA_DIR / "s2.csv"
S3_FILE = DATA_DIR / "s3.csv"


# ---------------------------------------------------------
# Required schema
# ---------------------------------------------------------

ENTITY_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def _load_entity_file(path):
    """Load and validate an entity CSV."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)

    missing = [
        column
        for column in ENTITY_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {missing}"
        )

    return df[ENTITY_COLUMNS].copy()


def _load_data():
    """Load S1, S2 and S3."""

    s1 = _load_entity_file(S1_FILE)
    s2 = _load_entity_file(S2_FILE)
    s3 = _load_entity_file(S3_FILE)

    return s1, s2, s3


# ---------------------------------------------------------
# Text helpers
# ---------------------------------------------------------

def _clean_text(value):
    """Basic text normalization for comparison."""

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def _similarity(value1, value2):
    """Return normalized fuzzy similarity between 0 and 1."""

    value1 = _clean_text(value1)
    value2 = _clean_text(value2)

    if not value1 or not value2:
        return 0.0

    return ratio(value1, value2) / 100.0


# ---------------------------------------------------------
# Feature calculation
# ---------------------------------------------------------

def _calculate_features(s1_row, candidate_row):
    """Calculate the features required by the project contract."""

    name1 = _clean_text(s1_row["business_name"])
    name2 = _clean_text(candidate_row["business_name"])

    address1 = _clean_text(s1_row["business_address"])
    address2 = _clean_text(candidate_row["business_address"])

    country1 = _clean_text(s1_row["country"])
    country2 = _clean_text(candidate_row["country"])

    name_sim = _similarity(name1, name2)
    addr_sim = _similarity(address1, address2)

    exact_name = int(
        bool(name1) and name1 == name2
    )

    country_match = int(
        bool(country1) and country1 == country2
    )

    len_diff = abs(len(name1) - len(name2))

    return {
        "name_sim": round(name_sim, 4),
        "addr_sim": round(addr_sim, 4),
        "exact_name": exact_name,
        "country_match": country_match,
        "len_diff": len_diff,
    }


# ---------------------------------------------------------
# Temporary scoring
# ---------------------------------------------------------

def _calculate_score(features):
    """
    Temporary scoring logic.

    This is NOT the final ML model.

    Person 2 will replace/refine this with the actual
    model-based matching implementation.
    """

    score = (
        0.50 * features["name_sim"]
        + 0.30 * features["addr_sim"]
        + 0.10 * features["exact_name"]
        + 0.10 * features["country_match"]
    )

    return round(score, 4)


# ---------------------------------------------------------
# Public function 1
# ---------------------------------------------------------

def list_entities():
    """
    Return original S1 IDs and business names.

    Contract:
    [
        {
            "entity_id": "...",
            "business_name": "..."
        }
    ]
    """

    s1, _, _ = _load_data()

    return s1[
        ["entity_id", "business_name"]
    ].to_dict(orient="records")


# ---------------------------------------------------------
# Public function 2
# ---------------------------------------------------------

def get_matches(s1_id):
    """
    Return ranked candidate matches for an S1 entity.

    Contract:

    {
        "s1": {
            "entity_id": ...,
            "business_name": ...,
            "business_address": ...,
            "country": ...
        },

        "threshold": numeric,

        "candidates": [
            {
                "candidate_id": ...,
                "source": ...,
                "business_name": ...,
                "business_address": ...,
                "country": ...,
                "score": ...,
                "accepted": ...,
                "features": {
                    "name_sim": ...,
                    "addr_sim": ...,
                    "exact_name": ...,
                    "country_match": ...,
                    "len_diff": ...
                }
            }
        ],

        "accepted_ids": [...]
    }

    NOTE:
    The threshold here is temporary and must NOT be treated
    as the final model threshold.
    """

    s1, s2, s3 = _load_data()

    matches = s1[
        s1["entity_id"].astype(str) == str(s1_id)
    ]

    if matches.empty:
        raise ValueError(f"Unknown S1 ID: {s1_id}")

    s1_row = matches.iloc[0]

    candidates = []

    # Process S2 and S3 separately so the source is preserved.
    for source, df in [("s2", s2), ("s3", s3)]:

        for _, candidate_row in df.iterrows():

            features = _calculate_features(
                s1_row,
                candidate_row
            )

            score = _calculate_score(features)

            candidate = {
                "candidate_id": candidate_row["entity_id"],
                "source": source,
                "business_name": candidate_row["business_name"],
                "business_address": candidate_row["business_address"],
                "country": candidate_row["country"],
                "score": score,
                "accepted": False,
                "features": features,
            }

            candidates.append(candidate)

    # Temporary threshold.
    # DO NOT treat 0.62 as the final selected threshold.
    threshold = 0.62

    # Rank highest score first.
    candidates.sort(
        key=lambda candidate: candidate["score"],
        reverse=True
    )

    # Temporary acceptance rule.
    for candidate in candidates:
        candidate["accepted"] = (
            candidate["score"] >= threshold
        )

    accepted_ids = [
        candidate["candidate_id"]
        for candidate in candidates
        if candidate["accepted"]
    ]

    return {
        "s1": {
            "entity_id": s1_row["entity_id"],
            "business_name": s1_row["business_name"],
            "business_address": s1_row["business_address"],
            "country": s1_row["country"],
        },
        "threshold": threshold,
        "candidates": candidates,
        "accepted_ids": accepted_ids,
    }