"""
generate_synthetic_data.py

Generates a synthetic IoT network-traffic dataset that mirrors the *structure*
of TON_IoT / CIC-IoT2023 (numeric flow features + binary attack/normal label +
a multi-class attack-type column) so the rest of the pipeline (partitioning,
baseline model, federated learning, trust-weighting, attack simulation) can be
built and tested end-to-end without needing the real dataset downloaded.

TO SWITCH TO THE REAL DATASET LATER:
  1. Download TON_IoT (e.g. the "Train_Test_Network.csv" file) or a
     CIC-IoT2023 CSV.
  2. Place it at data/raw/real_dataset.csv
  3. Update `load_dataset()` in src/common.py to point at that file and map
     its label/feature columns to the same schema used here:
       - a binary column named `label` (0 = normal, 1 = attack)
       - a categorical column named `attack_type`
       - numeric feature columns
     No other file needs to change.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification

RANDOM_STATE = 42
N_SAMPLES = 60_000
N_FEATURES = 16
ATTACK_TYPES = ["scanning", "dos", "ddos", "backdoor", "injection", "password", "xss"]

OUT_DIR = Path(__file__).parent / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate():
    rng = np.random.RandomState(RANDOM_STATE)

    # Core numeric feature matrix + binary label via sklearn's classification
    # generator -- gives us a "learnable" decision boundary rather than pure
    # noise, similar to how real network flows separate attack vs normal.
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=10,
        n_redundant=3,
        n_clusters_per_class=3,
        weights=[0.65, 0.35],  # imbalanced, like real IDS datasets
        flip_y=0.02,           # a bit of label noise for realism
        class_sep=1.2,
        random_state=RANDOM_STATE,
    )

    feature_names = [
        "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
        "src_ip_bytes", "dst_ip_bytes", "missed_bytes", "conn_state_enc",
        "proto_enc", "service_enc", "http_request_body_len",
        "http_response_body_len", "dns_qclass", "ssl_version_enc", "weird_notice_enc",
    ]
    df = pd.DataFrame(X, columns=feature_names)

    # Rescale a few columns to look like plausible flow statistics rather
    # than the standard-normal output of make_classification.
    df["duration"] = np.abs(df["duration"]) * 5
    df["src_bytes"] = np.abs(df["src_bytes"]) * 500
    df["dst_bytes"] = np.abs(df["dst_bytes"]) * 500
    df["src_pkts"] = np.abs(df["src_pkts"]).astype(int) * 3
    df["dst_pkts"] = np.abs(df["dst_pkts"]).astype(int) * 3
    df["src_ip_bytes"] = np.abs(df["src_ip_bytes"]) * 400
    df["dst_ip_bytes"] = np.abs(df["dst_ip_bytes"]) * 400
    df["proto_enc"] = (np.abs(df["proto_enc"]) % 3).astype(int)       # tcp/udp/icmp
    df["service_enc"] = (np.abs(df["service_enc"]) % 8).astype(int)   # http/dns/ssl/...
    df["conn_state_enc"] = (np.abs(df["conn_state_enc"]) % 6).astype(int)

    df["label"] = y.astype(int)

    # Assign an attack_type only to attack rows (label == 1); normal rows get "normal"
    attack_type = np.array(["normal"] * len(df), dtype=object)
    attack_idx = df.index[df["label"] == 1]
    attack_type[attack_idx] = rng.choice(ATTACK_TYPES, size=len(attack_idx))
    df["attack_type"] = attack_type

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    out_path = OUT_DIR / "synthetic_iot_traffic.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} cols to {out_path}")
    print(df["label"].value_counts(normalize=True).rename("class_balance"))
    return out_path


if __name__ == "__main__":
    generate()
