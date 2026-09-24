"""Model registry — add HNN / Attention-EGNN / Equiformer / MLP later."""

from __future__ import annotations

from typing import Any

import flax.linen as nn

from interp_physics.models.egnn import EGNN, EGNNConfig


def build_model(name: str, **kwargs: Any) -> nn.Module:
    name = name.lower().replace("-", "_")
    if name in {"egnn", "full_egnn"}:
        cfg = EGNNConfig(
            d_hidden=int(kwargs.get("d_hidden", 64)),
            d_msg=int(kwargs.get("d_msg", 32)),
            n_layers=int(kwargs.get("n_layers", 4)),
            return_activations=bool(kwargs.get("return_activations", False)),
        )
        return EGNN(config=cfg)
    raise ValueError(
        f"Unknown model {name!r}. Available now: egnn. "
        "Planned: hnn, attention_egnn, equiformer, non_eq_gnn, mlp."
    )
