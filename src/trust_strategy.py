"""
trust_strategy.py

Phase 4 (core contribution): TrustWeightedFedAvg.

Standard FedAvg weights each client's contribution purely by how many
examples it has. That means a malicious or corrupted node with a lot of
(poisoned) data gets a big vote. TrustWeightedFedAvg additionally scores
each client's update by:

  1. Similarity  -- cosine similarity between this client's weight update
     and the (num-examples-weighted) mean update across all participating
     clients this round. An update that points in a very different
     direction from the consensus is suspicious.
  2. Historical consistency -- an exponential moving average (EMA) of a
     client's similarity score across rounds, so a node that is
     *consistently* a bit off is penalised more than one with a single bad
     round, and a previously-flagged node can slowly regain trust if it
     starts behaving normally again.

The final aggregation weight for each client combines its trust score with
its example count (so trust down-weights bad actors without ignoring
dataset size entirely), then does a weighted average of the raw weights.
"""

from typing import Optional
import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

import model_utils as mu


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class TrustWeightedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, trust_ema_alpha: float = 0.5, trust_floor: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.trust_ema_alpha = trust_ema_alpha
        self.trust_floor = trust_floor  # minimum weight so no node is ever fully zeroed out
        self.trust_history: dict[str, float] = {}
        self.trust_log: list[dict] = []  # for later plotting/analysis
        self.current_weights: Optional[list[np.ndarray]] = (
            parameters_to_ndarrays(kwargs["initial_parameters"])
            if kwargs.get("initial_parameters") is not None
            else None
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}

        node_ids, weight_list, n_examples, deltas = [], [], [], []
        prev_flat = (
            mu.flatten_weights(self.current_weights) if self.current_weights is not None else None
        )

        for _, fit_res in results:
            nid = str(fit_res.metrics.get("node_id", "unknown"))
            w = parameters_to_ndarrays(fit_res.parameters)
            node_ids.append(nid)
            weight_list.append(w)
            n_examples.append(fit_res.num_examples)
            flat = mu.flatten_weights(w)
            deltas.append(flat - prev_flat if prev_flat is not None else flat)

        # Consensus direction: mean of UNIT-NORMALIZED deltas (not raw deltas).
        # Averaging raw deltas would let a magnitude-boosted malicious update
        # (a "model replacement" style attack) drag the consensus itself
        # toward it, defeating the whole point of comparing against it.
        # Normalizing first means every client gets an equal-magnitude "vote"
        # on the consensus direction, so as long as honest clients are the
        # majority, the consensus stays anchored to their direction
        # regardless of how large the malicious update's magnitude is.
        unit_deltas = []
        for d in deltas:
            norm = np.linalg.norm(d)
            unit_deltas.append(d / norm if norm > 0 else d)
        consensus = np.mean(np.stack(unit_deltas), axis=0)

        round_scores = {}
        for nid, delta in zip(node_ids, deltas):
            sim = _cosine_sim(delta, consensus)
            sim_clipped = max(sim, 0.0)  # negative similarity -> treat as 0 trust this round
            prev = self.trust_history.get(nid, sim_clipped)
            ema = self.trust_ema_alpha * sim_clipped + (1 - self.trust_ema_alpha) * prev
            self.trust_history[nid] = ema
            round_scores[nid] = ema
            self.trust_log.append({"round": server_round, "node_id": nid, "trust": ema, "raw_similarity": sim})

        # Combine trust with example count, apply a floor so no client is fully zeroed
        raw_weights = np.array(
            [max(round_scores[nid], self.trust_floor) * n for nid, n in zip(node_ids, n_examples)]
        )
        norm_weights = raw_weights / raw_weights.sum()

        # Weighted average of the RAW client weights (not deltas)
        new_global = [np.zeros_like(layer) for layer in weight_list[0]]
        for w, alpha in zip(weight_list, norm_weights):
            for layer_idx, layer in enumerate(w):
                new_global[layer_idx] = new_global[layer_idx] + alpha * layer

        self.current_weights = new_global
        metrics = {f"trust/{nid}": round_scores[nid] for nid in node_ids}
        return ndarrays_to_parameters(new_global), metrics
