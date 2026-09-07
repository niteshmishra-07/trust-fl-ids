"""
fl_client.py

Phase 3: each simulated node (network) is a Flower client. It trains a local
copy of the model for a few epochs on its own data partition and returns
only the updated weights (never raw traffic data) to the server.
"""

import numpy as np
import pandas as pd
import flwr as fl
from sklearn.metrics import accuracy_score, f1_score, log_loss

from common import FEATURE_COLUMNS, LABEL_COLUMN
import model_utils as mu

LOCAL_EPOCHS = 2


class NodeClient(fl.client.NumPyClient):
    def __init__(self, node_id: str, node_df: pd.DataFrame, scaler, boost_factor: float = 1.0):
        """`boost_factor` > 1.0 simulates a malicious node scaling up its
        update ("model replacement" style attack) so that a naive averaging
        aggregator gives it outsized influence, on top of any label-flip
        poisoning already baked into node_df. 1.0 = honest node."""
        self.node_id = node_id
        self.boost_factor = boost_factor
        X = scaler.transform(node_df[FEATURE_COLUMNS])
        y = node_df[LABEL_COLUMN].values.astype(np.int64)

        # Small local held-out slice for this node's own evaluate() calls
        n = len(X)
        split = max(int(n * 0.85), 1)
        self.X_train, self.y_train = X[:split], y[:split]
        self.X_val, self.y_val = X[split:], y[split:]
        if len(self.X_val) == 0:  # tiny partitions edge case
            self.X_val, self.y_val = self.X_train, self.y_train

        self.model = mu.build_model()
        # Ensure both classes are present in the init batch or MLP init fails
        init_X = self.X_train[:4]
        init_y = np.array([0, 1, 0, 1])[: len(init_X)]
        mu.init_architecture(self.model, init_X, init_y)

    def get_parameters(self, config):
        return mu.get_weights(self.model)

    def fit(self, parameters, config):
        mu.set_weights(self.model, parameters)
        epochs = config.get("local_epochs", LOCAL_EPOCHS)
        mu.local_train(self.model, self.X_train, self.y_train, epochs=epochs)
        new_weights = mu.get_weights(self.model)

        if self.boost_factor != 1.0:
            # Scale the UPDATE (delta from the received global weights), not
            # the raw weights, so the attack amplifies this node's influence
            # on the direction it moved, mimicking a model-replacement attack.
            new_weights = [
                g + self.boost_factor * (w - g) for w, g in zip(new_weights, parameters)
            ]

        return new_weights, len(self.X_train), {"node_id": self.node_id}

    def evaluate(self, parameters, config):
        mu.set_weights(self.model, parameters)
        probs = self.model.predict_proba(self.X_val)
        preds = self.model.predict(self.X_val)
        loss = log_loss(self.y_val, probs, labels=[0, 1])
        acc = accuracy_score(self.y_val, preds)
        f1 = f1_score(self.y_val, preds, zero_division=0)
        return float(loss), len(self.X_val), {"accuracy": acc, "f1": f1}
