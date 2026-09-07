"""
partition_data.py

Phase 1 deliverable: splits the dataset into N simulated "node" partitions
(as if each were a separate network/device sending only model updates, never
raw data, once federated learning starts in Phase 3).

Produces TWO versions so later phases can compare:
  - IID split:      data/nodes/iid/node_<i>.csv       (random, evenly mixed)
  - non-IID split:  data/nodes/noniid/node_<i>.csv     (skewed by attack type)

Also holds out a global test set (data/nodes/global_test.csv) that no node
ever trains on -- used only for final, apples-to-apples evaluation in later
phases.

Run: python data/partition_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # noqa: E402
import pandas as pd
from common import (
    load_dataset, get_global_test_split,
    partition_iid, partition_non_iid,
    LABEL_COLUMN, RANDOM_STATE,
)

NODES_DIR = Path(__file__).resolve().parent / "nodes"
N_NODES = 4


def report(nodes, label):
    print(f"\n--- {label} partition ({len(nodes)} nodes) ---")
    rows = []
    for i, node_df in enumerate(nodes):
        n = len(node_df)
        attack_rate = node_df[LABEL_COLUMN].mean()
        top_attack_types = (
            node_df.loc[node_df[LABEL_COLUMN] == 1, "attack_type"]
            .value_counts(normalize=True)
            .head(2)
        )
        top_str = ", ".join(f"{t}:{p:.0%}" for t, p in top_attack_types.items())
        rows.append((i, n, attack_rate, top_str))
        print(f"  node_{i}: {n:>6,} rows | attack rate {attack_rate:.1%} | dominant attack types: {top_str or 'n/a'}")
    return rows


def main():
    NODES_DIR.mkdir(parents=True, exist_ok=True)
    (NODES_DIR / "iid").mkdir(exist_ok=True)
    (NODES_DIR / "noniid").mkdir(exist_ok=True)

    # 1. Global held-out test set (never touched by any node/client)
    train_df, test_df = get_global_test_split()
    test_df.to_csv(NODES_DIR / "global_test.csv", index=False)
    print(f"Global held-out test set: {len(test_df):,} rows -> {NODES_DIR / 'global_test.csv'}")
    print(f"Training pool for partitioning: {len(train_df):,} rows")

    # 2. IID partition
    iid_nodes = partition_iid(train_df, N_NODES, random_state=RANDOM_STATE)
    for i, node_df in enumerate(iid_nodes):
        node_df.to_csv(NODES_DIR / "iid" / f"node_{i}.csv", index=False)
    report(iid_nodes, "IID")

    # 3. Non-IID partition
    noniid_nodes = partition_non_iid(train_df, N_NODES, random_state=RANDOM_STATE)
    for i, node_df in enumerate(noniid_nodes):
        node_df.to_csv(NODES_DIR / "noniid" / f"node_{i}.csv", index=False)
    report(noniid_nodes, "non-IID")

    print(f"\nDone. Node CSVs written under {NODES_DIR}/iid/ and {NODES_DIR}/noniid/")


if __name__ == "__main__":
    main()
