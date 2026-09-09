"""
run_experiment.py

Phase 3 + 4 + 5 orchestrator. This is the script you actually run.

Examples
--------
Clean run, plain FedAvg, IID partition (control condition):
    python src/run_experiment.py --strategy fedavg --partition iid

Clean run, trust-weighted, IID (sanity check -- should be close to FedAvg):
    python src/run_experiment.py --strategy trust --partition iid

Poisoning attack, plain FedAvg (expect degraded accuracy):
    python src/run_experiment.py --strategy fedavg --partition iid --poison-node 0 --poison-fraction 0.8

Poisoning attack, trust-weighted (expect accuracy to hold up -- THE key result):
    python src/run_experiment.py --strategy trust --partition iid --poison-node 0 --poison-fraction 0.8

Run the full comparison suite in one go and save results/comparison.json:
    python src/run_experiment.py --suite
"""

import argparse
import json
from pathlib import Path

import numpy as np
import flwr as fl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss

from common import (
    get_global_test_split, fit_scaler, to_xy,
    partition_iid, partition_non_iid, poison_node, RANDOM_STATE,
)
import model_utils as mu
from fl_client import NodeClient
from trust_strategy import TrustWeightedFedAvg

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def build_node_partitions(train_df, n_nodes, partition_type, poison_node_idx, poison_fraction, seed=RANDOM_STATE):
    if partition_type == "iid":
        nodes = partition_iid(train_df, n_nodes, random_state=seed)
    else:
        nodes = partition_non_iid(train_df, n_nodes, random_state=seed)
    nodes = [n.reset_index(drop=True) for n in nodes]
    if poison_node_idx is not None:
        nodes[poison_node_idx] = poison_node(nodes[poison_node_idx], poison_fraction, random_state=seed)
    return nodes


def make_client_fn(node_dfs, scaler, poison_node_idx=None, boost_factor=1.0):
    def client_fn(context):
        pid = int(context.node_config["partition-id"])
        node_df = node_dfs[pid]
        boost = boost_factor if (poison_node_idx is not None and pid == poison_node_idx) else 1.0
        return NodeClient(node_id=str(pid), node_df=node_df, scaler=scaler, boost_factor=boost).to_client()
    return client_fn


def make_evaluate_fn(X_test, y_test, template_model):
    def evaluate_fn(server_round, parameters_ndarrays, config):
        mu.set_weights(template_model, parameters_ndarrays)
        probs = template_model.predict_proba(X_test)
        preds = template_model.predict(X_test)
        loss = log_loss(y_test, probs, labels=[0, 1])
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds, zero_division=0),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
        }
        return loss, metrics
    return evaluate_fn


def run_once(strategy_name, partition_type, n_nodes, n_rounds, local_epochs,
             poison_node_idx, poison_fraction, boost_factor=1.0, seed=RANDOM_STATE, verbose=True,
             return_strategy=False):
    train_df, test_df = get_global_test_split(random_state=seed)
    scaler = fit_scaler(train_df)
    X_test, y_test = to_xy(test_df, scaler)

    node_dfs = build_node_partitions(train_df, n_nodes, partition_type, poison_node_idx, poison_fraction, seed)

    # Build + initialize a template model to get architecture-correct initial weights
    template_model = mu.build_model(random_state=seed)
    init_X, init_y = np.zeros((4, X_test.shape[1])), np.array([0, 1, 0, 1])
    mu.init_architecture(template_model, init_X, init_y)
    initial_weights = mu.get_weights(template_model)
    initial_parameters = fl.common.ndarrays_to_parameters(initial_weights)

    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # we use centralized evaluate_fn instead of client-side eval
        min_fit_clients=n_nodes,
        min_available_clients=n_nodes,
        on_fit_config_fn=lambda rnd: {"local_epochs": local_epochs},
        evaluate_fn=make_evaluate_fn(X_test, y_test, template_model),
        initial_parameters=initial_parameters,
    )

    if strategy_name == "fedavg":
        strategy = fl.server.strategy.FedAvg(**common_kwargs)
    elif strategy_name == "trust":
        strategy = TrustWeightedFedAvg(**common_kwargs)
    else:
        raise ValueError(strategy_name)

    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(node_dfs, scaler, poison_node_idx=poison_node_idx, boost_factor=boost_factor),
        num_clients=n_nodes,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        ray_init_args={"include_dashboard": False, "logging_level": 40, "log_to_driver": False},
    )

    # history.metrics_centralized has per-round metrics dicts keyed by metric name
    rounds_metrics = {}
    for metric_name, values in history.metrics_centralized.items():
        rounds_metrics[metric_name] = [v for _, v in values]

    final = {k: v[-1] for k, v in rounds_metrics.items() if v}
    result = {
        "strategy": strategy_name,
        "partition": partition_type,
        "n_nodes": n_nodes,
        "n_rounds": n_rounds,
        "poison_node": poison_node_idx,
        "poison_fraction": poison_fraction if poison_node_idx is not None else 0.0,
        "final_metrics": final,
        "metrics_by_round": rounds_metrics,
    }
    trust_log = getattr(strategy, "trust_log", None)
    if trust_log:
        result["trust_log"] = trust_log

    if verbose:
        tag = f"[{strategy_name} | {partition_type} | poison={poison_node_idx}]"
        print(f"{tag} final accuracy={final.get('accuracy'):.4f}  f1={final.get('f1'):.4f}")

    if return_strategy:
        return result, strategy, scaler
    return result


def save_dashboard_artifacts(strategy, scaler):
    """Persists what the dashboard's 'upload & analyze a device' tool needs:
    the trained global model weights, the fitted scaler, and the 'healthy'
    consensus direction -- so any newly-uploaded traffic file can later be
    scored for trust without re-running the whole federated simulation.
    """
    import joblib
    weights = strategy.current_weights
    np.savez(RESULTS_DIR / "global_model_weights.npz", *weights)
    joblib.dump(scaler, RESULTS_DIR / "scaler.joblib")
    if strategy.last_consensus is not None:
        np.save(RESULTS_DIR / "reference_consensus.npy", strategy.last_consensus)
    print(f"Saved dashboard artifacts to {RESULTS_DIR} (global_model_weights.npz, scaler.joblib, reference_consensus.npy)")


def run_suite(n_nodes=4, n_rounds=10, local_epochs=2, poison_fraction=1.0, boost_factor=4.0, seed=RANDOM_STATE):
    """Runs the Phase 5 comparison: clean baseline, FedAvg+attack, Trust+attack."""
    configs = [
        dict(strategy_name="fedavg", poison_node_idx=None, boost_factor=1.0, tag="clean_fedavg"),
        dict(strategy_name="fedavg", poison_node_idx=0, boost_factor=boost_factor, tag="fedavg_under_attack"),
        dict(strategy_name="trust", poison_node_idx=None, boost_factor=1.0, tag="clean_trust"),
        dict(strategy_name="trust", poison_node_idx=0, boost_factor=boost_factor, tag="trust_under_attack"),
    ]
    all_results = {}
    for cfg in configs:
        tag = cfg.pop("tag")
        if tag == "clean_trust":
            # This run's final global model + consensus direction becomes the
            # reference used by the dashboard's "upload & analyze a device" tool
            # to score arbitrary new/uploaded traffic later.
            res, strategy, scaler = run_once(
                partition_type="iid",
                n_nodes=n_nodes,
                n_rounds=n_rounds,
                local_epochs=local_epochs,
                poison_fraction=poison_fraction,
                seed=seed,
                return_strategy=True,
                **cfg,
            )
            save_dashboard_artifacts(strategy, scaler)
        else:
            res = run_once(
                partition_type="iid",
                n_nodes=n_nodes,
                n_rounds=n_rounds,
                local_epochs=local_epochs,
                poison_fraction=poison_fraction,
                seed=seed,
                **cfg,
            )
        all_results[tag] = res

    out_path = RESULTS_DIR / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Phase 5 comparison summary ===")
    for tag, res in all_results.items():
        fm = res["final_metrics"]
        print(f"{tag:22s} acc={fm.get('accuracy', float('nan')):.4f}  f1={fm.get('f1', float('nan')):.4f}")
    print(f"\nSaved full results to {out_path}")
    return all_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["fedavg", "trust"], default="fedavg")
    p.add_argument("--partition", choices=["iid", "noniid"], default="iid")
    p.add_argument("--n-nodes", type=int, default=4)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--poison-node", type=int, default=None, help="index of node to poison (0-based)")
    p.add_argument("--poison-fraction", type=float, default=1.0)
    p.add_argument("--boost-factor", type=float, default=4.0, help="update-scaling factor for the poisoned node")
    p.add_argument("--suite", action="store_true", help="run the full Phase 5 comparison suite")
    args = p.parse_args()

    if args.suite:
        run_suite(n_nodes=args.n_nodes, n_rounds=args.rounds, local_epochs=args.local_epochs,
                  poison_fraction=args.poison_fraction, boost_factor=args.boost_factor)
    else:
        run_once(
            strategy_name=args.strategy,
            partition_type=args.partition,
            n_nodes=args.n_nodes,
            n_rounds=args.rounds,
            local_epochs=args.local_epochs,
            poison_node_idx=args.poison_node,
            poison_fraction=args.poison_fraction,
            boost_factor=args.boost_factor,
        )


if __name__ == "__main__":
    main()
