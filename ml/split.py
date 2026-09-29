import random
from typing import Dict, List, Tuple
import pandas as pd

def split_s1_ids(
    s1_ids: List[str],
    test_ratio: float = 0.20,
    val_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[Dict[str, str], List[str], List[str], List[str]]:
    """
    Splits S1 IDs deterministically into train, val, and test splits.
    Returns:
        split_map: dict mapping s1_id -> split_name ('train', 'val', or 'test')
        train_ids: list of S1 IDs in train
        val_ids: list of S1 IDs in val
        test_ids: list of S1 IDs in test
    """
    sorted_ids = sorted(list(set(s1_ids)))
    rng = random.Random(seed)
    shuffled_ids = sorted_ids.copy()
    rng.shuffle(shuffled_ids)

    n_total = len(shuffled_ids)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_test - n_val

    test_ids = sorted(shuffled_ids[:n_test])
    val_ids = sorted(shuffled_ids[n_test:n_test + n_val])
    train_ids = sorted(shuffled_ids[n_test + n_val:])

    split_map = {}
    for sid in train_ids:
        split_map[sid] = "train"
    for sid in val_ids:
        split_map[sid] = "val"
    for sid in test_ids:
        split_map[sid] = "test"

    return split_map, train_ids, val_ids, test_ids
