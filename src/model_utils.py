"""
model_utils.py

A small MLP (implemented via sklearn's MLPClassifier with partial_fit) used
as the local/global model in the federated setup. sklearn's MLP exposes its
weights as plain numpy arrays (coefs_, intercepts_), which is exactly what
Flower's NumPyClient needs to exchange -- no need for PyTorch/TensorFlow.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier

HIDDEN_LAYER_SIZES = (32, 16)
CLASSES = np.array([0, 1])


def build_model(random_state=42) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=1,       # we drive training manually via partial_fit per round
        warm_start=True,
        random_state=random_state,
    )


def init_architecture(model: MLPClassifier, X_sample: np.ndarray, y_sample: np.ndarray):
    """Must be called once before get/set_weights will work -- this is what
    allocates model.coefs_ / model.intercepts_ with the right shapes."""
    model.partial_fit(X_sample, y_sample, classes=CLASSES)
    return model


def get_weights(model: MLPClassifier) -> list[np.ndarray]:
    """Flatten coefs_ + intercepts_ into a single list of ndarrays (Flower's
    Parameters format). The split point (len(model.coefs_)) is fixed by the
    architecture, so set_weights can always unambiguously reverse this."""
    return [w.copy() for w in model.coefs_] + [b.copy() for b in model.intercepts_]


def set_weights(model: MLPClassifier, weights: list[np.ndarray]):
    n_layers = len(model.coefs_)  # architecture already initialized
    model.coefs_ = [w.copy() for w in weights[:n_layers]]
    model.intercepts_ = [b.copy() for b in weights[n_layers:]]


def local_train(model: MLPClassifier, X, y, epochs: int = 1):
    for _ in range(epochs):
        model.partial_fit(X, y)
    return model


def flatten_weights(weights: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([w.flatten() for w in weights])
