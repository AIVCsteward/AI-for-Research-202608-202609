"""Shared experiment configuration for AIVC stage 2-3.

The values here are deliberately plain Python data so Person B/C can import the
same contract without pulling in the feature implementation.
"""
from copy import deepcopy


EXPERIMENT_CONFIG = {
    "exp_name": "residual_v1",
    "encoder": {
        "hash_dim_chemical": 32,
        "hash_dim_plate": 16,
        "strain_prior_pca_dim": 32,
        "chem_anchor_pca_dim": 64,
        "cross_dim_strain_medium": 10,
        "cross_dim_chemical_temperature": 92,
        "time_period_minutes": 240.0,
        "use_strain_prior": True,
        "use_chem_anchor": True,
        "use_hash_features": True,
        "use_cross_features": True,
        "use_categorical_features": True,
        "use_temperature": True,
        "use_time_features": True,
        "d_emb": 256,
    },
    "decoder": {
        "hidden": 256,
        "dropout": 0.1,
        "residual_decompose": True,
    },
    "gnn": {
        "enabled": True,
        "k_neighbors": 5,
        "pearson_threshold": 0.7,
        "num_layers": 1,
    },
    "loss": {
        "mse_weight": 1.0,
        "fc_pearson_weight": 0.3,
        "residual_l2_weight": 0.01,
        "correlation_consistency_weight": 0.1,
    },
    "training": {
        "epochs": 100,
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "early_stopping_patience": 20,
    },
}


def get_experiment_config(overrides=None):
    """Return an isolated config copy with optional recursive overrides."""
    config = deepcopy(EXPERIMENT_CONFIG)
    if overrides:
        _deep_update(config, overrides)
    return config


def _deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target
