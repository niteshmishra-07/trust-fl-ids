"""
baseline_model.py

Phase 2: a centralized (non-federated) classifier trained on the full
dataset. This is the reference ceiling that the federated setup (Phase 3)
and trust-weighted setup (Phase 4) are compared against.

Run: python src/baseline_model.py
"""

import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

from common import get_global_test_split, fit_scaler, to_xy, RANDOM_STATE

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    train_df, test_df = get_global_test_split()
    scaler = fit_scaler(train_df)
    X_train, y_train = to_xy(train_df, scaler)
    X_test, y_test = to_xy(test_df, scaler)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Centralized baseline — accuracy: {acc:.4f}  F1: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    results = {"accuracy": acc, "f1": f1, "report": report}
    with open(RESULTS_DIR / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {RESULTS_DIR / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
