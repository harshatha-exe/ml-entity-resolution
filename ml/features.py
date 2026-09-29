from typing import Dict, Any, List
import pandas as pd
from rapidfuzz import fuzz

FEATURE_NAMES = ["name_sim", "addr_sim", "exact_name", "country_match", "len_diff"]

def extract_pair_features(s1_name: str, s1_addr: str, s1_country: str,
                          cand_name: str, cand_addr: str, cand_country: str) -> Dict[str, float]:
    """
    Extracts numerical features for a pair of S1 and candidate entities.
    Returns dict with keys: FEATURE_NAMES
    """
    s1_name_str = str(s1_name) if pd.notna(s1_name) else ""
    cand_name_str = str(cand_name) if pd.notna(cand_name) else ""
    s1_addr_str = str(s1_addr) if pd.notna(s1_addr) else ""
    cand_addr_str = str(cand_addr) if pd.notna(cand_addr) else ""
    s1_country_str = str(s1_country) if pd.notna(s1_country) else ""
    cand_country_str = str(cand_country) if pd.notna(cand_country) else ""

    # RapidFuzz similarities (0 to 1 scale)
    name_sim = fuzz.token_sort_ratio(s1_name_str.lower(), cand_name_str.lower()) / 100.0
    addr_sim = fuzz.token_sort_ratio(s1_addr_str.lower(), cand_addr_str.lower()) / 100.0

    # Exact name match
    exact_name = 1.0 if s1_name_str.strip().lower() == cand_name_str.strip().lower() else 0.0

    # Country match
    country_match = 1.0 if (s1_country_str and cand_country_str and
                            s1_country_str.strip().lower() == cand_country_str.strip().lower()) else 0.0

    # Name length difference
    len_diff = float(abs(len(s1_name_str) - len(cand_name_str)))

    return {
        "name_sim": name_sim,
        "addr_sim": addr_sim,
        "exact_name": exact_name,
        "country_match": country_match,
        "len_diff": len_diff
    }

def pair_features(s1_row: pd.Series, candidate_row: pd.Series) -> Dict[str, float]:
    """Wrapper function matching issue contract."""
    return extract_pair_features(
        s1_row.get("norm_name", s1_row.get("business_name", "")),
        s1_row.get("norm_address", s1_row.get("business_address", "")),
        s1_row.get("norm_country", s1_row.get("country", "")),
        candidate_row.get("norm_name", candidate_row.get("business_name", "")),
        candidate_row.get("norm_address", candidate_row.get("business_address", "")),
        candidate_row.get("norm_country", candidate_row.get("country", ""))
    )

def prepare_featured_dataset(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    links_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges candidate pairs with S1 and S2/S3 entity details, computes features,
    and labels ground truth target (1 if in links_df, else 0).
    """
    s1_dict = s1_df.drop_duplicates(subset=["entity_id"]).set_index("entity_id").to_dict(orient="index")
    s2_dict = s2_df.drop_duplicates(subset=["entity_id"]).set_index("entity_id").to_dict(orient="index")
    s3_dict = s3_df.drop_duplicates(subset=["entity_id"]).set_index("entity_id").to_dict(orient="index")

    # Set of true links
    link_pairs = set(zip(links_df["s1_id"], links_df["other_id"]))

    rows = []
    for _, row in candidates_df.iterrows():
        s1_id = row["s1_id"]
        cand_id = row["candidate_id"]
        source = row.get("source", "s2")

        s1_info = s1_dict.get(s1_id, {})
        cand_info = s2_dict.get(cand_id, {}) if source == "s2" else s3_dict.get(cand_id, {})

        feats = pair_features(pd.Series(s1_info), pd.Series(cand_info))

        target = 1 if (s1_id, cand_id) in link_pairs else 0
        feats["s1_id"] = s1_id
        feats["candidate_id"] = cand_id
        feats["source"] = source
        feats["target"] = target

        rows.append(feats)

    return pd.DataFrame(rows)
