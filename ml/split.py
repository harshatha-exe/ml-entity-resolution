import os
import random
from typing import Dict, List, Tuple
import pandas as pd

def split_s1_ids(
    s1_ids: List[str],
    test_ratio: float = 0.20,
    val_ratio: float = 0.20,
    seed: int = 42,
    data_dir: str = "data"
) -> Tuple[Dict[str, str], List[str], List[str], List[str]]:
    """
    Splits S1 IDs deterministically into train, val, and test splits with stratification
    on whether S1 entities have true links, ensuring all splits receive positive and negative pairs.
    """
    sorted_ids = sorted(list(set(s1_ids)))
    links_file = os.path.join(data_dir, "links.csv")

    if os.path.exists(links_file):
        links_df = pd.read_csv(links_file)
        linked_set = set(links_df["s1_id"])
    else:
        linked_set = set()

    linked_ids = [sid for sid in sorted_ids if sid in linked_set]
    unlinked_ids = [sid for sid in sorted_ids if sid not in linked_set]

    rng = random.Random(seed)
    rng.shuffle(linked_ids)
    rng.shuffle(unlinked_ids)

    def split_group(group: List[str]):
        n_tot = len(group)
        n_te = int(round(n_tot * test_ratio))
        n_va = int(round(n_tot * val_ratio))
        te = group[:n_te]
        va = group[n_te:n_te + n_va]
        tr = group[n_te + n_va:]
        return tr, va, te

    tr_linked, va_linked, te_linked = split_group(linked_ids)
    tr_unlinked, va_unlinked, te_unlinked = split_group(unlinked_ids)

    train_ids = sorted(tr_linked + tr_unlinked)
    val_ids = sorted(va_linked + va_unlinked)
    test_ids = sorted(te_linked + te_unlinked)

    split_map = {}
    for sid in train_ids:
        split_map[sid] = "train"
    for sid in val_ids:
        split_map[sid] = "val"
    for sid in test_ids:
        split_map[sid] = "test"

    return split_map, train_ids, val_ids, test_ids
