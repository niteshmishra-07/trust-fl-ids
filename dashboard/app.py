"""
dashboard/app.py

Phase 6 deliverable: a demo/viva dashboard for the project.

Run:
    streamlit run dashboard/app.py

Shows:
  1. Centralized baseline vs. federated results (Phase 2 vs 3)
  2. The key Phase 5 result: FedAvg vs. trust-weighted aggregation, clean vs. under attack
  3. Per-node trust scores over rounds (from the trust-weighted run) -- shows the
     poisoned node's trust score getting driven down
  4. A live "feed a traffic sample through the model" classifier demo

This is a presentation aid, not the research contribution -- kept intentionally simple.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "results"

st.set_page_config(page_title="Trust-Weighted FL for IoT IDS", layout="wide")
st.title("Trust-Weighted Federated Learning for IoT Intrusion Detection")
st.caption("Live dashboard — Phase 6 demo")


@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


baseline = load_json(RESULTS_DIR / "baseline_results.json")
comparison = load_json(RESULTS_DIR / "comparison.json")

# ---------------------------------------------------------------------------
# 1. Headline metrics
# ---------------------------------------------------------------------------
st.header("1. Headline results")

if baseline is None and comparison is None:
    st.warning(
        "No results found yet. Run these first:\n\n"
        "```\npython src/baseline_model.py\npython src/run_experiment.py --suite\n```"
    )
else:
    cols = st.columns(4)
    if baseline:
        cols[0].metric("Centralized baseline (Phase 2)", f"{baseline['accuracy']:.1%}")
    if comparison and "clean_fedavg" in comparison:
        cols[1].metric("Federated, clean (FedAvg)", f"{comparison['clean_fedavg']['final_metrics']['accuracy']:.1%}")
    if comparison and "fedavg_under_attack" in comparison:
        cols[2].metric(
            "FedAvg UNDER ATTACK",
            f"{comparison['fedavg_under_attack']['final_metrics']['accuracy']:.1%}",
            delta=f"{(comparison['fedavg_under_attack']['final_metrics']['accuracy'] - comparison['clean_fedavg']['final_metrics']['accuracy']):.1%}",
            delta_color="inverse",
        )
    if comparison and "trust_under_attack" in comparison:
        cols[3].metric(
            "Trust-weighted UNDER ATTACK",
            f"{comparison['trust_under_attack']['final_metrics']['accuracy']:.1%}",
            delta=f"{(comparison['trust_under_attack']['final_metrics']['accuracy'] - comparison['clean_trust']['final_metrics']['accuracy']):.1%}",
        )

# ---------------------------------------------------------------------------
# 2. Phase 5 comparison chart
# ---------------------------------------------------------------------------
st.header("2. FedAvg vs. Trust-Weighted — clean vs. under a poisoning attack")

if comparison:
    rows = []
    for tag, res in comparison.items():
        fm = res["final_metrics"]
        rows.append({
            "run": tag,
            "accuracy": fm.get("accuracy"),
            "f1": fm.get("f1"),
            "precision": fm.get("precision"),
            "recall": fm.get("recall"),
        })
    df = pd.DataFrame(rows).set_index("run")
    st.bar_chart(df[["accuracy", "f1"]])
    st.dataframe(df.style.format("{:.3f}"), use_container_width=True)

    st.markdown(
        "**Reading this chart:** `clean_*` runs have no attacker. `*_under_attack` runs have one "
        "node poisoning its labels *and* scaling up its update to dominate the average "
        "(a model-replacement style attack). A large accuracy drop from clean → attack for "
        "`fedavg` but not for `trust` is the core result of this project."
    )
else:
    st.info("Run `python src/run_experiment.py --suite` to populate this section.")

# ---------------------------------------------------------------------------
# 3. Per-round convergence
# ---------------------------------------------------------------------------
st.header("3. Accuracy over communication rounds")

if comparison:
    run_choice = st.selectbox("Select a run", list(comparison.keys()))
    by_round = comparison[run_choice].get("metrics_by_round", {})
    if "accuracy" in by_round:
        acc_series = pd.Series(by_round["accuracy"], name="accuracy")
        acc_series.index = range(1, len(acc_series) + 1)
        st.line_chart(acc_series)
    else:
        st.write("No per-round data available for this run.")

# ---------------------------------------------------------------------------
# 4. Per-node trust scores (only present for trust-weighted runs)
# ---------------------------------------------------------------------------
st.header("4. Per-node trust scores over rounds")

trust_run = None
for tag in ["trust_under_attack", "clean_trust"]:
    if comparison and tag in comparison and "trust_log" in comparison[tag]:
        trust_run = tag
        break

if trust_run:
    st.caption(f"Showing trust scores from run: `{trust_run}`")
    trust_log = comparison[trust_run]["trust_log"]
    trust_df = pd.DataFrame(trust_log)
    pivot = trust_df.pivot(index="round", columns="node_id", values="trust")
    st.line_chart(pivot)
    if trust_run == "trust_under_attack":
        st.markdown(
            "Node `0` is the poisoned node in this run — watch its trust score fall relative "
            "to the honest nodes as rounds progress."
        )
else:
    st.info("Run the trust-weighted strategy (e.g. via `--suite`) to populate this section.")

# ---------------------------------------------------------------------------
# 5. Live classification demo
# ---------------------------------------------------------------------------
st.header("5. Try the model on a traffic sample")

try:
    from common import load_dataset, FEATURE_COLUMNS, get_global_test_split, fit_scaler, to_xy
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    @st.cache_resource
    def get_demo_model():
        train_df, test_df = get_global_test_split()
        scaler = fit_scaler(train_df)
        X_train, y_train = to_xy(train_df, scaler)
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        return clf, scaler, test_df

    clf, scaler, test_df = get_demo_model()

    st.write("Pick a random row from the held-out test set and classify it:")
    if st.button("Sample a random traffic flow"):
        row = test_df.sample(1)
        X_row, y_row = to_xy(row, scaler)
        pred = clf.predict(X_row)[0]
        proba = clf.predict_proba(X_row)[0]

        st.write(row[FEATURE_COLUMNS + ["attack_type"]].T.rename(columns={row.index[0]: "value"}))
        label_str = "🚨 ATTACK" if pred == 1 else "✅ NORMAL"
        true_str = "ATTACK" if y_row[0] == 1 else "NORMAL"
        st.subheader(f"Prediction: {label_str}  (confidence: {max(proba):.1%})")
        st.caption(f"Ground truth: {true_str}" + ("  — correct ✔️" if pred == y_row[0] else "  — wrong ✗"))

except FileNotFoundError:
    st.info("Run `python data/generate_synthetic_data.py` and `python data/partition_data.py` first.")
