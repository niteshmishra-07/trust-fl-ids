"""
common.py

Shared utilities: dataset loading, preprocessing, and node partitioning
(IID and non-IID) used by the baseline model, the federated clients, and the
attack simulation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data" / "raw" / "synthetic_iot_traffic.csv"
NODES_DIR = ROOT / "data" / "nodes"
NODES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
    "src_ip_bytes", "dst_ip_bytes", "missed_bytes", "conn_state_enc",
    "proto_enc", "service_enc", "http_request_body_len",
    "http_response_body_len", "dns_qclass", "ssl_version_enc", "weird_notice_enc",
]
LABEL_COLUMN = "label"
RANDOM_STATE = 42


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the dataset. Swap `path` to point at a real TON_IoT/CIC-IoT2023
    CSV once available -- as long as it exposes the same `label` column and
    numeric feature columns (see FEATURE_COLUMNS), nothing else needs to change.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/generate_synthetic_data.py` first "
            "(or point RAW_DATA_PATH at your real dataset)."
        )
    return pd.read_csv(path)


def get_global_test_split(test_size=0.15, random_state=RANDOM_STATE):
    """Held-out global test set, never seen by any federated client.
    Used only for final, apples-to-apples evaluation of the global model.
    """
    df = load_dataset()
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[LABEL_COLUMN]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])
    return scaler


def to_xy(df: pd.DataFrame, scaler: StandardScaler):
    X = scaler.transform(df[FEATURE_COLUMNS])
    y = df[LABEL_COLUMN].values.astype(np.int64)
    return X, y


def _split_dataframe(df: pd.DataFrame, n_chunks: int):
    """np.array_split-equivalent that reliably returns DataFrames (not ndarrays)."""
    idx_chunks = np.array_split(np.arange(len(df)), n_chunks)
    return [df.iloc[idx].reset_index(drop=True) for idx in idx_chunks]


def partition_iid(df: pd.DataFrame, n_nodes: int, random_state=RANDOM_STATE):
    """Randomly shuffle and split into n_nodes roughly-equal, class-balanced chunks."""
    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return _split_dataframe(shuffled, n_nodes)


def partition_non_iid(df: pd.DataFrame, n_nodes: int, primary_share=0.7, random_state=RANDOM_STATE):
    """Each node gets a disproportionate share of one or two attack_types,
    simulating networks that see different kinds of traffic/attacks.
    `primary_share` controls how skewed each node's data is toward its
    assigned attack type(s) (higher = more non-IID).
    """
    rng = np.random.RandomState(random_state)
    attack_types = [t for t in df["attack_type"].unique() if t != "normal"]
    rng.shuffle(attack_types)
    assignments = np.array_split(attack_types, n_nodes)

    normal_df = df[df["attack_type"] == "normal"]
    normal_chunks = _split_dataframe(
        normal_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True), n_nodes
    )

    nodes = []
    for i in range(n_nodes):
        primary_types = list(assignments[i])
        primary_df = df[df["attack_type"].isin(primary_types)]
        # Sample primary_share of this node's "primary" attack rows,
        # plus a small amount of everything else to avoid zero-shot classes.
        n_primary = int(len(primary_df) * primary_share / max(n_nodes, 1) * n_nodes)
        primary_sample = primary_df.sample(
            n=min(n_primary, len(primary_df)), random_state=random_state
        )
        other_df = df[~df["attack_type"].isin(primary_types) & (df["attack_type"] != "normal")]
        other_sample = other_df.sample(
            frac=(1 - primary_share) * 0.3, random_state=random_state
        )
        node_df = pd.concat([primary_sample, other_sample, normal_chunks[i]])
        node_df = node_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        nodes.append(node_df)
    return nodes


def save_node_partitions(nodes, prefix="node"):
    paths = []
    for i, node_df in enumerate(nodes):
        p = NODES_DIR / f"{prefix}_{i}.csv"
        node_df.to_csv(p, index=False)
        paths.append(p)
    return paths


def poison_node(node_df: pd.DataFrame, flip_fraction: float, random_state=RANDOM_STATE) -> pd.DataFrame:
    """Simulate a label-flipping poisoning attack: flips `flip_fraction` of
    this node's labels (0<->1). Returns a NEW dataframe; does not mutate input.
    """
    poisoned = node_df.copy()
    rng = np.random.RandomState(random_state)
    n_flip = int(len(poisoned) * flip_fraction)
    flip_idx = rng.choice(poisoned.index, size=n_flip, replace=False)
    poisoned.loc[flip_idx, LABEL_COLUMN] = 1 - poisoned.loc[flip_idx, LABEL_COLUMN]
    return poisoned
