"""
ML-backed entity matching interface.

Issue #5 implementation:
- Loads S1/S2/S3 data.
- Normalizes entity data.
- Uses blocking to generate candidate pairs.
- Builds pair features using the saved feature order.
- Loads the trained model once.
- Loads the saved validation threshold once.
- Uses model.predict_proba() for scoring.
- Returns ranked candidates and accepted IDs.
- Preserves the Issue #1 public response structure.
"""

from functools import lru_cache
from pathlib import Path
import json

import joblib
import pandas as pd

from ml.normalize import normalize_dataframe
from ml.blocking import block_candidates_for_s1
from ml.features import pair_features


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ML_DIR = PROJECT_ROOT / "ml"

S1_FILE = DATA_DIR / "s1.csv"
S2_FILE = DATA_DIR / "s2.csv"
S3_FILE = DATA_DIR / "s3.csv"

MODEL_FILE = ML_DIR / "model.joblib"
METRICS_FILE = ML_DIR / "metrics.json"


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
# Data loading helpers
# ---------------------------------------------------------

def _load_entity_file(path):
    """Load and validate one entity CSV."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

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


def _load_normalized_data():
    """
    Load S1/S2/S3 and add normalized fields.

    The original columns are retained so that the API can return
    user-visible original text.
    """

    s1 = _load_entity_file(S1_FILE)
    s2 = _load_entity_file(S2_FILE)
    s3 = _load_entity_file(S3_FILE)

    s1 = normalize_dataframe(s1)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)

    s2["source"] = "s2"
    s3["source"] = "s3"

    candidates = pd.concat(
        [s2, s3],
        ignore_index=True
    )

    return s1, candidates


# ---------------------------------------------------------
# Model + configuration loading
# ---------------------------------------------------------

def _load_model():
    """
    Load the saved model artifact.

    The model artifact created by ml.train contains:
        model
        feature_names
        model_type
        val_f05
    """

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}. "
            f"Run 'python -m ml.train' first."
        )

    artifact = joblib.load(MODEL_FILE)

    if isinstance(artifact, dict):
        if "model" not in artifact:
            raise ValueError(
                "model.joblib does not contain a 'model' entry."
            )

        model = artifact["model"]
        feature_names = artifact.get("feature_names")

    else:
        model = artifact
        feature_names = None

    return model, feature_names


def _load_metrics():
    """
    Load evaluation metrics and the selected threshold.
    """

    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_FILE}. "
            f"Run 'python -m ml.evaluate' first."
        )

    with open(METRICS_FILE, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    if "threshold" not in metrics:
        raise ValueError(
            "metrics.json does not contain a threshold."
        )

    if "feature_names" not in metrics:
        raise ValueError(
            "metrics.json does not contain feature_names."
        )

    threshold = float(metrics["threshold"])
    feature_names = metrics["feature_names"]

    return threshold, feature_names


# ---------------------------------------------------------
# Cached runtime state
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def _load_runtime():
    """
    Load all runtime resources once per Python process.

    Returns:
        s1_df
        candidates_df
        model
        feature_names
        threshold
    """

    s1_df, candidates_df = _load_normalized_data()

    model, model_feature_names = _load_model()

    threshold, metrics_feature_names = _load_metrics()

    # Prefer the feature order saved with the model artifact.
    # Fall back to metrics.json if the artifact does not contain it.
    if model_feature_names:
        feature_names = list(model_feature_names)

        # Ensure the evaluation artifact agrees with the model artifact.
        if list(metrics_feature_names) != feature_names:
            raise ValueError(
                "Feature order mismatch between model.joblib "
                "and metrics.json."
            )
    else:
        feature_names = list(metrics_feature_names)

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The saved model does not support predict_proba()."
        )

    return (
        s1_df,
        candidates_df,
        model,
        feature_names,
        threshold,
    )


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

    s1_df, _, _, _, _ = _load_runtime()

    return s1_df[
        ["entity_id", "business_name"]
    ].to_dict(orient="records")


# ---------------------------------------------------------
# Public function 2
# ---------------------------------------------------------

def get_matches(s1_id):
    """
    Return ranked ML-based candidate matches for an S1 entity.

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
    """

    (
        s1_df,
        candidates_df,
        model,
        feature_names,
        threshold,
    ) = _load_runtime()

    # -----------------------------------------------------
    # Find S1 entity
    # -----------------------------------------------------

    matches = s1_df[
        s1_df["entity_id"].astype(str) == str(s1_id)
    ]

    if matches.empty:
        raise ValueError(
            f"Unknown S1 ID: {s1_id}"
        )

    s1_row = matches.iloc[0]

    # -----------------------------------------------------
    # Blocking
    # -----------------------------------------------------

    blocked_candidates = block_candidates_for_s1(
        s1_row,
        candidates_df,
        max_candidates=20
    )

    # -----------------------------------------------------
    # Zero-candidate behavior
    # -----------------------------------------------------

    if blocked_candidates.empty:
        return {
            "s1": {
                "entity_id": s1_row["entity_id"],
                "business_name": s1_row["business_name"],
                "business_address": s1_row["business_address"],
                "country": s1_row["country"],
            },
            "threshold": threshold,
            "candidates": [],
            "accepted_ids": [],
        }

    # -----------------------------------------------------
    # Build pair features
    # -----------------------------------------------------

    feature_rows = []
    candidate_records = []

    for _, candidate_row in blocked_candidates.iterrows():

        features = pair_features(
            s1_row,
            candidate_row
        )

        feature_rows.append(features)

        candidate_records.append(
            {
                "candidate_id": candidate_row["candidate_id"],
                "source": candidate_row["source"],
                "business_name": candidate_row["business_name"],
                "business_address": candidate_row["business_address"],
                "country": candidate_row["country"],
                "features": features,
            }
        )

    # -----------------------------------------------------
    # DataFrame in saved feature order
    # -----------------------------------------------------

    feature_df = pd.DataFrame(feature_rows)

    # Ensure the exact order expected by the trained model.
    feature_df = feature_df[feature_names]

    # -----------------------------------------------------
    # ML prediction
    # -----------------------------------------------------

    probabilities = model.predict_proba(
        feature_df
    )[:, 1]

    # -----------------------------------------------------
    # Attach predictions
    # -----------------------------------------------------

    candidates = []

    for candidate, score in zip(
        candidate_records,
        probabilities
    ):

        score = float(score)

        candidate_output = {
            "candidate_id": candidate["candidate_id"],
            "source": candidate["source"],
            "business_name": candidate["business_name"],
            "business_address": candidate["business_address"],
            "country": candidate["country"],
            "score": round(score, 4),
            "accepted": bool(score >= threshold),
            "features": candidate["features"],
        }

        candidates.append(candidate_output)

    # -----------------------------------------------------
    # Rank highest score first
    # -----------------------------------------------------

    candidates.sort(
        key=lambda candidate: candidate["score"],
        reverse=True
    )

    # -----------------------------------------------------
    # Accepted IDs
    # -----------------------------------------------------

    accepted_ids = [
        candidate["candidate_id"]
        for candidate in candidates
        if candidate["accepted"]
    ]

    # -----------------------------------------------------
    # Final Issue #1-compatible response
    # -----------------------------------------------------

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