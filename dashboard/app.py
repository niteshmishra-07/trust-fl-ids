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
# 0. Upload & Analyze a Device's Traffic -- THE main interactive feature
# ---------------------------------------------------------------------------
st.header("🔍 Upload & Analyze a Device's Traffic")
st.write(
    "Upload a CSV of a device's network traffic. The system classifies each traffic flow as "
    "**attack or normal**, and computes an overall **Trust Score** for the device -- the same "
    "mechanism (Phase 4) used to detect and down-weight malicious nodes during federated training. "
    "A low trust score means: if this device were part of the federated system, its contribution "
    "would be heavily discounted because its behavior doesn't match what a healthy device's does."
)

sample_cols = st.columns(2)
with sample_cols[0]:
    try:
        with open(ROOT / "data" / "sample_device_clean.csv", "rb") as f:
            st.download_button("⬇️ Download a sample CLEAN device", f, file_name="sample_device_clean.csv")
    except FileNotFoundError:
        pass
with sample_cols[1]:
    try:
        with open(ROOT / "data" / "sample_device_poisoned.csv", "rb") as f:
            st.download_button("⬇️ Download a sample POISONED device", f, file_name="sample_device_poisoned.csv")
    except FileNotFoundError:
        pass

st.caption(
    "For a live demo: download one of the samples above, then upload it below. Try both, one "
    "after another, to see the trust score contrast."
)

uploaded_file = st.file_uploader("Upload device traffic CSV", type="csv")

if uploaded_file is not None:
    try:
        import joblib
        from common import FEATURE_COLUMNS, LABEL_COLUMN, to_xy
        import model_utils as mu

        upload_df = pd.read_csv(uploaded_file)
        missing_cols = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in upload_df.columns]
        if missing_cols:
            st.error(f"Uploaded file is missing required columns: {missing_cols}")
        else:
            scaler = joblib.load(RESULTS_DIR / "scaler.joblib")
            npz = np.load(RESULTS_DIR / "global_model_weights.npz")
            global_weights = [npz[k] for k in npz.files]
            reference_consensus = np.load(RESULTS_DIR / "reference_consensus.npy")

            X, y = to_xy(upload_df, scaler)

            # Classify every row using the CURRENT trained global model
            clf_model = mu.build_model()
            init_X, init_y = X[:4], np.array([0, 1, 0, 1])[: min(4, len(X))]
            mu.init_architecture(clf_model, init_X, init_y)
            mu.set_weights(clf_model, global_weights)
            preds = clf_model.predict(X)
            probs = clf_model.predict_proba(X)

            # Trust score: locally train from the global model on this device's
            # data, then compare the resulting update direction to the
            # reference "healthy" direction learned during clean training.
            local_model = mu.build_model()
            mu.init_architecture(local_model, init_X, init_y)
            mu.set_weights(local_model, global_weights)
            mu.local_train(local_model, X, y, epochs=2)
            new_weights = mu.get_weights(local_model)
            delta = mu.flatten_weights(new_weights) - mu.flatten_weights(global_weights)
            norm = np.linalg.norm(delta)
            unit_delta = delta / norm if norm > 0 else delta
            cos_sim = float(
                np.dot(unit_delta, reference_consensus)
                / (np.linalg.norm(unit_delta) * np.linalg.norm(reference_consensus))
            )
            trust_score = max(cos_sim, 0.0)

            attack_rate = preds.mean()

            res_cols = st.columns(3)
            res_cols[0].metric("Flows analyzed", f"{len(upload_df):,}")
            res_cols[1].metric("Flagged as ATTACK", f"{attack_rate:.1%}")

            if trust_score > 0.6:
                verdict, color = "✅ TRUSTED", "green"
            elif trust_score > 0.2:
                verdict, color = "⚠️ SUSPICIOUS", "orange"
            else:
                verdict, color = "🚨 UNTRUSTED", "red"
            res_cols[2].metric("Trust Score", f"{trust_score:.2f}")
            st.markdown(f":{color}[**{verdict}**] -- this device's behavior pattern " +
                        ("closely matches" if trust_score > 0.6 else
                         "partially resembles" if trust_score > 0.2 else
                         "sharply diverges from") +
                        " what honest, trustworthy devices look like in this system.")

            preview = upload_df[FEATURE_COLUMNS].copy()
            preview["prediction"] = np.where(preds == 1, "ATTACK", "NORMAL")
            preview["confidence"] = probs.max(axis=1).round(3)
            st.write("Per-flow results (first 50 rows):")
            st.dataframe(preview.head(50), use_container_width=True)

    except FileNotFoundError:
        st.warning(
            "Model artifacts not found. Run `python src/run_experiment.py --suite` first -- "
            "it trains the global model and saves what this tool needs."
        )

st.divider()

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

    # -----------------------------------------------------------------------
    # 6. Live simulated IoT traffic feed
    # -----------------------------------------------------------------------
    st.header("6. Live simulated IoT traffic feed")
    st.caption(
        "Simulates an IoT device continuously sending traffic to the trained model, one flow "
        "at a time, in real time. Each row below is drawn from held-out, unseen traffic -- "
        "the same as a real device's traffic would look once the trained model is deployed."
    )

    import time
    import datetime

    if "live_feed_log" not in st.session_state:
        st.session_state.live_feed_log = []
    if "live_feed_running" not in st.session_state:
        st.session_state.live_feed_running = False
    if "force_attack_ticks" not in st.session_state:
        st.session_state.force_attack_ticks = 0

    normal_rows = test_df[test_df["label"] == 0]
    attack_rows = test_df[test_df["label"] == 1]

    ctrl_cols = st.columns([1, 1, 2])
    with ctrl_cols[0]:
        if st.button("▶ Start live feed" if not st.session_state.live_feed_running else "⏸ Pause live feed"):
            st.session_state.live_feed_running = not st.session_state.live_feed_running
    with ctrl_cols[1]:
        if st.button("🚨 Simulate attack burst"):
            st.session_state.force_attack_ticks = 5  # next 5 flows are drawn from real attack traffic
    with ctrl_cols[2]:
        status = "🟢 LIVE" if st.session_state.live_feed_running else "⏸️ paused"
        st.write(f"Status: **{status}**  |  Total flows classified: **{len(st.session_state.live_feed_log)}**")

    @st.fragment(run_every=2 if st.session_state.live_feed_running else None)
    def live_feed_fragment():
        if st.session_state.live_feed_running:
            if st.session_state.force_attack_ticks > 0 and len(attack_rows) > 0:
                row = attack_rows.sample(1)
                st.session_state.force_attack_ticks -= 1
            else:
                row = normal_rows.sample(1) if len(normal_rows) > 0 else test_df.sample(1)

            X_row, y_row = to_xy(row, scaler)
            pred = clf.predict(X_row)[0]
            proba = clf.predict_proba(X_row)[0]

            st.session_state.live_feed_log.insert(0, {
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "prediction": "🚨 ATTACK" if pred == 1 else "✅ NORMAL",
                "confidence": f"{max(proba):.1%}",
                "attack_type": row["attack_type"].iloc[0],
                "ground_truth": "ATTACK" if y_row[0] == 1 else "NORMAL",
            })
            st.session_state.live_feed_log = st.session_state.live_feed_log[:20]  # keep last 20

        if st.session_state.live_feed_log:
            feed_df = pd.DataFrame(st.session_state.live_feed_log)
            st.dataframe(feed_df, use_container_width=True, hide_index=True)
        else:
            st.info("Click **Start live feed** to begin streaming simulated traffic through the model.")

    live_feed_fragment()

except FileNotFoundError:
    st.info("Run `python data/generate_synthetic_data.py` and `python data/partition_data.py` first.")
