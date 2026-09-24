"""Model registry for the architecture spectrum in the proposal."""

from __future__ import annotations

from typing import Any

import flax.linen as nn

from interp_physics.models.attention_egnn import AttentionEGNN, AttentionEGNNConfig
from interp_physics.models.egnn import EGNN, EGNNConfig
from interp_physics.models.eq_transformer import EquivariantTransformer, EquivariantTransformerConfig
from interp_physics.models.hnn import HNN, HNNConfig
from interp_physics.models.mlp import MLPDynamics, MLPConfig
from interp_physics.models.non_eq_gnn import NonEquivariantGNN, NonEqGNNConfig

AVAILABLE_MODELS = (
    "egnn",
    "attention_egnn",
    "non_eq_gnn",
    "mlp",
    "hnn",
    "equivariant_transformer",
)


def build_model(name: str, **kwargs: Any) -> nn.Module:
    name = name.lower().replace("-", "_")

    def _flag(key: str, default: bool = False) -> bool:
        return bool(kwargs.get(key, default))

    if name in {"egnn", "full_egnn"}:
        return EGNN(
            config=EGNNConfig(
                d_hidden=int(kwargs.get("d_hidden", 64)),
                d_msg=int(kwargs.get("d_msg", 32)),
                n_layers=int(kwargs.get("n_layers", 4)),
                return_activations=_flag("return_activations"),
            )
        )

    if name in {"attention_egnn", "attn_egnn"}:
        return AttentionEGNN(
            config=AttentionEGNNConfig(
                d_hidden=int(kwargs.get("d_hidden", 64)),
                d_msg=int(kwargs.get("d_msg", 32)),
                n_layers=int(kwargs.get("n_layers", 4)),
                return_activations=_flag("return_activations"),
            )
        )

    if name in {"non_eq_gnn", "noneq_gnn", "non_equivariant_gnn"}:
        return NonEquivariantGNN(
            config=NonEqGNNConfig(
                d_hidden=int(kwargs.get("d_hidden", 64)),
                d_msg=int(kwargs.get("d_msg", 32)),
                n_layers=int(kwargs.get("n_layers", 4)),
                return_activations=_flag("return_activations"),
            )
        )

    if name in {"mlp", "mlp_dynamics"}:
        return MLPDynamics(
            config=MLPConfig(
                d_hidden=int(kwargs.get("d_hidden", 256)),
                n_layers=int(kwargs.get("n_layers", 4)),
                return_activations=_flag("return_activations"),
            )
        )

    if name in {"hnn", "hamiltonian"}:
        return HNN(
            config=HNNConfig(
                d_hidden=int(kwargs.get("d_hidden", 64)),
                n_layers_potential=int(kwargs.get("n_layers_potential", 2)),
                n_integrate=int(kwargs.get("n_integrate", 10)),
                dt=float(kwargs.get("dt", 0.1)),
                soft_cutoff_sq=float(kwargs.get("soft_cutoff_sq", 0.25)),
                return_activations=_flag("return_activations"),
            )
        )

    if name in {"equivariant_transformer", "eq_transformer", "equiformer"}:
        return EquivariantTransformer(
            config=EquivariantTransformerConfig(
                d_hidden=int(kwargs.get("d_hidden", 64)),
                n_heads=int(kwargs.get("n_heads", 4)),
                n_layers=int(kwargs.get("n_layers", 4)),
                return_activations=_flag("return_activations"),
            )
        )

    raise ValueError(f"Unknown model {name!r}. Available: {', '.join(AVAILABLE_MODELS)}")
