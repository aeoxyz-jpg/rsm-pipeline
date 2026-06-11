"""MLPClassifier factory. Note: MLP does not natively support class_weight."""

from __future__ import annotations

from sklearn.neural_network import MLPClassifier

from rsm_pipeline.models.schema import MLPConfig


def _make_mlp(cfg: MLPConfig, *, seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=tuple(cfg.hidden_layer_sizes),
        activation=cfg.activation,
        max_iter=cfg.max_iter,
        early_stopping=cfg.early_stopping,
        random_state=seed,
    )
