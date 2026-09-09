"""
dashboard/app.py

Phase 6 deliverable -- the project's interactive demo application.

Run:
    streamlit run dashboard/app.py

Tabs:
  1. Analyze Device  -- upload a CSV of a device's traffic, get per-flow
     attack/normal classification AND a Trust Score, computed the same way
     a malicious node gets down-weighted during federated training.
  2. Live Feed        -- a simulated device streaming traffic in real time.
  3. Model Performance -- Phase 2/3/5 results: baseline, FedAvg vs.
     trust-weighted, clean vs. under a poisoning attack.
  4. How It Works      -- plain-language explanation of the system, for
     anyone unfamiliar with the project (e.g. an evaluator).
"""

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
RESULTS_DIR = ROOT / "results"

st.set_page_config(page_title="Trust-Weighted FL for IoT IDS", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# Theme / styling
# ---------------------------------------------------------------------------
COLOR_TRUST = "#2DD4BF"     # teal  -- trusted / normal
COLOR_WARN = "#F59E0B"      # amber -- suspicious
COLOR_ALERT = "#EF4444"     # red   -- untrusted / attack
COLOR_MUTED = "#7B8794"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Space Grotesk', sans-serif;
}}
h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
}}
.mono, .stMetric, code, .stDataFrame, [data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
}}

.hero {{
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.5rem;
}}
h1.hero-title {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    line-height: 1.15 !important;
    margin: 0 !important;
    color: #E5E9F0 !important;
}}
p.hero-sub {{
    color: {COLOR_MUTED} !important;
    font-size: 1.08rem !important;
    margin-top: 0.7rem !important;
    max-width: 62ch;
    font-weight: 400 !important;
}}

.badge {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 0.35rem 0.9rem;
    border-radius: 4px;
    letter-spacing: 0.02em;
}}
.badge-trust {{ background: rgba(45,212,191,0.14); color: {COLOR_TRUST}; border: 1px solid rgba(45,212,191,0.4); }}
.badge-warn  {{ background: rgba(245,158,11,0.14); color: {COLOR_WARN}; border: 1px solid rgba(245,158,11,0.4); }}
.badge-alert {{ background: rgba(239,68,68,0.14); color: {COLOR_ALERT}; border: 1px solid rgba(239,68,68,0.4); }}

.section-note {{
    color: {COLOR_MUTED};
    font-size: 0.92rem;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1 class="hero-title">Trust-weighted federated learning for IoT intrusion detection</h1>
    <p class="hero-sub">Multiple networks jointly train one shared attack-detector without
    sharing raw traffic data -- and a trust score keeps a compromised device from
    poisoning that shared model.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


baseline = load_json(RESULTS_DIR / "baseline_results.json")
comparison = load_json(RESULTS_DIR / "comparison.json")

tab_analyze, tab_live, tab_perf, tab_about = st.tabs(
    ["🔍  Analyze Device", "📡  Live Feed", "📊  Model Performance", "ℹ️  How It Works"]
)

# ===========================================================================
# TAB 1 -- Analyze Device
# ===========================================================================
with tab_analyze:
    st.write(
        "Upload a CSV of a device's network traffic. Every flow gets classified as "
        "**attack** or **normal**, and the device as a whole gets a **Trust Score** -- the "
        "same mechanism used to detect and down-weight malicious nodes during training, "
        "applied here to new, previously unseen data."
    )

    sample_cols = st.columns(2)
    with sample_cols[0]:
        try:
            with open(ROOT / "data" / "sample_device_clean.csv", "rb") as f:
                st.download_button("⬇ Sample: clean device", f, file_name="sample_device_clean.csv", use_container_width=True)
        except FileNotFoundError:
            pass
    with sample_cols[1]:
        try:
            with open(ROOT / "data" / "sample_device_poisoned.csv", "rb") as f:
                st.download_button("⬇ Sample: poisoned device", f, file_name="sample_device_poisoned.csv", use_container_width=True)
        except FileNotFoundError:
            pass

    st.markdown(
        '<p class="section-note">For a live demo: download a sample above, then upload it below. '
        'Try both to see the trust score contrast.</p>', unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload device traffic (.csv)", type="csv", label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            import joblib
            import model_utils as mu
            from common import FEATURE_COLUMNS, LABEL_COLUMN, to_xy

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

                clf_model = mu.build_model()
                init_X, init_y = X[:4], np.array([0, 1, 0, 1])[: min(4, len(X))]
                mu.init_architecture(clf_model, init_X, init_y)
                mu.set_weights(clf_model, global_weights)
                preds = clf_model.predict(X)
                probs = clf_model.predict_proba(X)

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

                if trust_score > 0.6:
                    verdict, badge_class, gauge_color = "TRUSTED", "badge-trust", COLOR_TRUST
                elif trust_score > 0.2:
                    verdict, badge_class, gauge_color = "SUSPICIOUS", "badge-warn", COLOR_WARN
                else:
                    verdict, badge_class, gauge_color = "UNTRUSTED", "badge-alert", COLOR_ALERT

                result_cols = st.columns([1, 1.3])

                with result_cols[0]:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=trust_score,
                        number={"valueformat": ".2f", "font": {"family": "JetBrains Mono", "size": 46, "color": gauge_color}},
                        gauge={
                            "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": COLOR_MUTED},
                            "bar": {"color": gauge_color, "thickness": 0.3},
                            "bgcolor": "rgba(0,0,0,0)",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, 0.2], "color": "rgba(239,68,68,0.15)"},
                                {"range": [0.2, 0.6], "color": "rgba(245,158,11,0.15)"},
                                {"range": [0.6, 1.0], "color": "rgba(45,212,191,0.15)"},
                            ],
                        },
                    ))
                    fig.update_layout(
                        height=260, margin=dict(l=20, r=20, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E5E9F0"},
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f'<div style="text-align:center;"><span class="badge {badge_class}">{verdict}</span></div>', unsafe_allow_html=True)

                with result_cols[1]:
                    with st.container(border=True):
                        m1, m2 = st.columns(2)
                        m1.metric("Flows analyzed", f"{len(upload_df):,}")
                        m2.metric("Flagged as attack", f"{attack_rate:.1%}")
                        interpretation = {
                            "TRUSTED": "This device's behavior pattern closely matches what honest, "
                                       "trustworthy devices look like in this system.",
                            "SUSPICIOUS": "This device's behavior partially resembles trustworthy nodes, "
                                          "but shows some inconsistency worth monitoring.",
                            "UNTRUSTED": "This device's behavior sharply diverges from trustworthy nodes. "
                                         "If part of the federated system, its updates would be heavily "
                                         "down-weighted or excluded during aggregation.",
                        }[verdict]
                        st.write(interpretation)

                st.write("")
                st.write("**Per-flow results** (first 50 rows)")
                preview = upload_df[FEATURE_COLUMNS].copy()
                preview["prediction"] = np.where(preds == 1, "ATTACK", "NORMAL")
                preview["confidence"] = probs.max(axis=1).round(3)
                st.dataframe(preview.head(50), use_container_width=True, height=280)

        except FileNotFoundError:
            st.warning(
                "Model artifacts not found. Run `python src/run_experiment.py --suite` first -- "
                "it trains the global model and saves what this tool needs."
            )

# ===========================================================================
# TAB 2 -- Live Feed
# ===========================================================================
with tab_live:
    st.write(
        "Simulates a device continuously streaming traffic to the trained model in real time. "
        "Each row is drawn from held-out, unseen traffic -- the same as what a real deployed "
        "device's traffic would look like."
    )

    try:
        import joblib
        from common import FEATURE_COLUMNS, get_global_test_split, fit_scaler, to_xy
        from sklearn.ensemble import RandomForestClassifier

        @st.cache_resource
        def get_demo_model():
            train_df, test_df = get_global_test_split()
            scaler = fit_scaler(train_df)
            X_train, y_train = to_xy(train_df, scaler)
            clf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            return clf, scaler, test_df

        clf, scaler, test_df = get_demo_model()
        normal_rows = test_df[test_df["label"] == 0]
        attack_rows = test_df[test_df["label"] == 1]

        if "live_feed_log" not in st.session_state:
            st.session_state.live_feed_log = []
        if "live_feed_running" not in st.session_state:
            st.session_state.live_feed_running = False
        if "force_attack_ticks" not in st.session_state:
            st.session_state.force_attack_ticks = 0

        ctrl_cols = st.columns([1, 1, 2])
        with ctrl_cols[0]:
            label = "⏸ Pause" if st.session_state.live_feed_running else "▶ Start live feed"
            if st.button(label, use_container_width=True):
                st.session_state.live_feed_running = not st.session_state.live_feed_running
        with ctrl_cols[1]:
            if st.button("🚨 Simulate attack burst", use_container_width=True):
                st.session_state.force_attack_ticks = 5
        with ctrl_cols[2]:
            status = f'<span class="badge badge-trust">LIVE</span>' if st.session_state.live_feed_running else '<span class="badge" style="background:rgba(123,135,148,0.14);color:#7B8794;border:1px solid rgba(123,135,148,0.4);">PAUSED</span>'
            st.markdown(f'{status} &nbsp; Total flows: <span class="mono">{len(st.session_state.live_feed_log)}</span>', unsafe_allow_html=True)

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
                st.session_state.live_feed_log = st.session_state.live_feed_log[:20]

            if st.session_state.live_feed_log:
                feed_df = pd.DataFrame(st.session_state.live_feed_log)
                st.dataframe(feed_df, use_container_width=True, hide_index=True, height=460)
            else:
                st.info("Click **Start live feed** to begin streaming simulated traffic through the model.")

        live_feed_fragment()

    except FileNotFoundError:
        st.info("Run `python data/generate_synthetic_data.py` and `python data/partition_data.py` first.")

# ===========================================================================
# TAB 3 -- Model Performance
# ===========================================================================
with tab_perf:
    if baseline is None and comparison is None:
        st.warning(
            "No results found yet. Run these first:\n\n"
            "```\npython src/baseline_model.py\npython src/run_experiment.py --suite\n```"
        )
    else:
        cols = st.columns(4)
        if baseline:
            cols[0].metric("Centralized baseline", f"{baseline['accuracy']:.1%}")
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

        st.divider()
        st.subheader("FedAvg vs. Trust-Weighted -- clean vs. under a poisoning attack")

        if comparison:
            rows = []
            for tag, res in comparison.items():
                fm = res["final_metrics"]
                rows.append({"run": tag, "accuracy": fm.get("accuracy"), "f1": fm.get("f1"),
                             "precision": fm.get("precision"), "recall": fm.get("recall")})
            df = pd.DataFrame(rows).set_index("run")
            st.bar_chart(df[["accuracy", "f1"]])
            st.dataframe(df.style.format("{:.3f}"), use_container_width=True)

            st.markdown(
                '<p class="section-note"><b>Reading this chart:</b> <code>clean_*</code> runs have '
                'no attacker. <code>*_under_attack</code> runs have one node poisoning its labels '
                '<i>and</i> scaling up its update to dominate the average. A large accuracy drop from '
                'clean → attack for <code>fedavg</code> but not for <code>trust</code> is the core '
                "result of this project.</p>", unsafe_allow_html=True,
            )

            st.divider()
            st.subheader("Accuracy over communication rounds")
            run_choice = st.selectbox("Select a run", list(comparison.keys()))
            by_round = comparison[run_choice].get("metrics_by_round", {})
            if "accuracy" in by_round:
                acc_series = pd.Series(by_round["accuracy"], name="accuracy")
                acc_series.index = range(1, len(acc_series) + 1)
                st.line_chart(acc_series)

            st.divider()
            st.subheader("Per-node trust scores over rounds")
            trust_run = None
            for tag in ["trust_under_attack", "clean_trust"]:
                if tag in comparison and "trust_log" in comparison[tag]:
                    trust_run = tag
                    break
            if trust_run:
                st.caption(f"Showing trust scores from run: `{trust_run}`")
                trust_df = pd.DataFrame(comparison[trust_run]["trust_log"])
                pivot = trust_df.pivot(index="round", columns="node_id", values="trust")
                st.line_chart(pivot)
                if trust_run == "trust_under_attack":
                    st.markdown(
                        '<p class="section-note">Node <code>0</code> is the poisoned node in this run '
                        "-- watch its trust score fall relative to the honest nodes as rounds "
                        "progress.</p>", unsafe_allow_html=True,
                    )
        else:
            st.info("Run `python src/run_experiment.py --suite` to populate this section.")

# ===========================================================================
# TAB 4 -- How It Works
# ===========================================================================
with tab_about:
    st.markdown("""
### The problem

Several IoT networks each want an AI that can tell "normal traffic" from "attack traffic" --
but nobody wants to hand their raw network data to a central server. That's a privacy risk,
and it makes the central server itself a target.

### Federated learning

Instead of sharing raw traffic, each network trains its own local copy of the model on its
own data, and only shares the model's *learned weights* with a central aggregator. The
aggregator averages everyone's weights into one shared, smarter model and sends it back out.
Repeat over several rounds -- nobody's actual traffic ever leaves their own network.

### The problem with plain averaging

Standard federated averaging (FedAvg) treats every participant equally. If one network is
compromised and feeding the shared model corrupted updates, it gets the same vote as an
honest network -- and can drag the whole shared model off course.

### Trust-weighted aggregation (this project's contribution)

Each round, every node's update is scored by how closely it matches the *consensus
direction* the honest majority is moving in, tracked over time. A node whose updates
consistently look anomalous gets down-weighted in the aggregation -- automatically, without
needing to be told in advance which node is malicious.

### What the demo tabs show

- **Analyze Device** -- takes that same trust-scoring mechanism and applies it to a brand
  new file you upload, standing in for "a device's traffic."
- **Live Feed** -- shows the trained model classifying a continuous stream of traffic in
  real time.
- **Model Performance** -- the actual evidence: trained accuracy, and the head-to-head
  comparison proving trust-weighted aggregation survives a poisoning attack that collapses
  plain FedAvg.
""")
