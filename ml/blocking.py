import os
from typing import Dict, List, Tuple
import pandas as pd
from rapidfuzz import fuzz

from ml.normalize import normalize_dataframe, clean_text
from ml.split import split_s1_ids
from ml.seed_data import generate_seed_data

def block_candidates_for_s1(
    s1_row: pd.Series,
    candidates_df: pd.DataFrame,
    max_candidates: int = 20
) -> pd.DataFrame:
    """
    Generates blocked candidate pairs for a single S1 entity against candidates_df.
    Keys used:
    1. Exact norm_name
    2. Exact suffix_free_name
    3. Exact norm_address
    Fallback (if candidates < 3):
    4. First token of name or high fuzzy token match
    Caps candidates at max_candidates per S1 by ranking fuzzy similarity.
    """
    s1_norm_name = s1_row["norm_name"]
    s1_sf_name = s1_row["suffix_free_name"]
    s1_norm_addr = s1_row["norm_address"]
    s1_first_word = s1_norm_name.split()[0] if s1_norm_name else ""

    # Matching conditions
    cond_name = candidates_df["norm_name"] == s1_norm_name
    cond_sf = (candidates_df["suffix_free_name"] == s1_sf_name) & (s1_sf_name != "")
    cond_addr = (candidates_df["norm_address"] == s1_norm_addr) & (s1_norm_addr != "")

    primary_matches = candidates_df[cond_name | cond_sf | cond_addr]

    if len(primary_matches) < 3 and s1_first_word and len(s1_first_word) >= 3:
        # Fallback: candidate names starting with or containing first word
        fallback_cond = candidates_df["norm_name"].str.contains(s1_first_word, regex=False, na=False)
        fallback_matches = candidates_df[fallback_cond]
        matched = pd.concat([primary_matches, fallback_matches]).drop_duplicates(subset=["entity_id"])
    else:
        matched = primary_matches

    if matched.empty:
        # If still empty, return top 3 fuzzy name matches to avoid 0 candidate hard drop
        scores = candidates_df["norm_name"].apply(lambda n: fuzz.token_sort_ratio(s1_norm_name, n))
        matched = candidates_df.iloc[scores.nlargest(3).index]

    if len(matched) > max_candidates:
        # Rank by fuzzy name similarity and cap
        scores = matched["norm_name"].apply(lambda n: fuzz.token_sort_ratio(s1_norm_name, n))
        matched = matched.iloc[scores.nlargest(max_candidates).index]

    matched_out = matched.copy()
    matched_out["s1_id"] = s1_row["entity_id"]
    matched_out = matched_out.rename(columns={"entity_id": "candidate_id"})
    return matched_out

def run_blocking(data_dir: str = "data") -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Ensure seed data exists
    if not os.path.exists(os.path.join(data_dir, "s1.csv")):
        generate_seed_data(data_dir)

    s1_df = pd.read_csv(os.path.join(data_dir, "s1.csv"))
    s2_df = pd.read_csv(os.path.join(data_dir, "s2.csv"))
    s3_df = pd.read_csv(os.path.join(data_dir, "s3.csv"))
    links_df = pd.read_csv(os.path.join(data_dir, "links.csv"))

    # Normalize datasets
    s1_norm = normalize_dataframe(s1_df)
    s2_norm = normalize_dataframe(s2_df)
    s3_norm = normalize_dataframe(s3_df)

    s2_norm["source"] = "s2"
    s3_norm["source"] = "s3"

    all_candidates = pd.concat([s2_norm, s3_norm], ignore_index=True)

    # Perform S1 split
    split_map, train_ids, val_ids, test_ids = split_s1_ids(s1_norm["entity_id"].tolist(), seed=42)
    s1_norm["split"] = s1_norm["entity_id"].map(split_map)

    candidates_by_split = {}
    recall_results = {}

    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_s1 = s1_norm[s1_norm["entity_id"].isin(split_ids)]
        blocked_list = []
        for _, s1_row in split_s1.iterrows():
            cand_df = block_candidates_for_s1(s1_row, all_candidates, max_candidates=20)
            blocked_list.append(cand_df)

        if blocked_list:
            split_candidates = pd.concat(blocked_list, ignore_index=True)
        else:
            split_candidates = pd.DataFrame()

        # Save to data/candidates_{split_name}.csv
        split_candidates.to_csv(os.path.join(data_dir, f"candidates_{split_name}.csv"), index=False)
        candidates_by_split[split_name] = split_candidates

        # Compute recall
        # True links belonging to this split
        split_links = links_df[links_df["s1_id"].isin(split_ids)]
        total_true_links = len(split_links)

        if total_true_links > 0:
            # Check how many (s1_id, other_id) pairs are in split_candidates
            merged = pd.merge(
                split_links,
                split_candidates,
                left_on=["s1_id", "other_id"],
                right_on=["s1_id", "candidate_id"],
                how="inner"
            )
            surviving_links = len(merged)
            recall = surviving_links / total_true_links
        else:
            recall = 1.0

        recall_results[split_name] = recall
        print(f"Split {split_name:5s}: Total True Links = {total_true_links}, Surviving Links = {surviving_links if total_true_links > 0 else 0}, Recall = {recall:.2%}")

    return recall_results, candidates_by_split["train"], candidates_by_split["val"], candidates_by_split["test"]

if __name__ == "__main__":
    run_blocking()
