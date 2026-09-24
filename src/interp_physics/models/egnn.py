"""E(3)-equivariant GNN with velocity integration (Assignment 2 baseline)."""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp

from interp_physics.models.base import ModelOutput
from interp_physics.models.message_gnn import MessageGNNConfig, MessagePassingGNN


@dataclass(frozen=True)
class EGNNConfig:
    d_hidden: int = 64
    d_msg: int = 32
    n_layers: int = 4
    return_activations: bool = False


class EGNN(nn.Module):
    """Full EGNN: distances in messages, tanh-bounded equivariant updates."""

    config: EGNNConfig = EGNNConfig()

    @nn.compact
    def __call__(
        self,
        node_feat: jnp.ndarray,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
        edges: jnp.ndarray,
        *,
        return_activations: bool | None = None,
    ) -> ModelOutput:
        cfg = self.config
        want = cfg.return_activations if return_activations is None else return_activations
        inner = MessagePassingGNN(
            config=MessageGNNConfig(
                d_hidden=cfg.d_hidden,
                d_msg=cfg.d_msg,
                n_layers=cfg.n_layers,
                return_activations=want,
            )
        )
        return inner(node_feat, pos, vel, edges, return_activations=want)
