"""Attention-weighted EGNN (Assignment 2 improvement)."""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp

from interp_physics.models.base import ModelOutput
from interp_physics.models.message_gnn import MessageGNNConfig, MessagePassingGNN


@dataclass(frozen=True)
class AttentionEGNNConfig:
    d_hidden: int = 64
    d_msg: int = 32
    n_layers: int = 4
    return_activations: bool = False


class AttentionEGNN(nn.Module):
    """EGNN with softmax attention over pairwise messages."""

    config: AttentionEGNNConfig = AttentionEGNNConfig()

    @nn.compact
    def __call__(
        self,
        node_feat: jnp.ndarray,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
        edges: jnp.ndarray,
        *,
        return_activations: bool | None = None,
        ablate_site: str | None = None,
        ablate_layer: int | None = None,
        ablate_basis: jnp.ndarray | None = None,
    ) -> ModelOutput:
        cfg = self.config
        want = cfg.return_activations if return_activations is None else return_activations
        inner = MessagePassingGNN(
            config=MessageGNNConfig(
                d_hidden=cfg.d_hidden,
                d_msg=cfg.d_msg,
                n_layers=cfg.n_layers,
                use_attention=True,
                return_activations=want,
            )
        )
        return inner(
            node_feat,
            pos,
            vel,
            edges,
            return_activations=want,
            ablate_site=ablate_site,
            ablate_layer=ablate_layer,
            ablate_basis=ablate_basis,
        )
