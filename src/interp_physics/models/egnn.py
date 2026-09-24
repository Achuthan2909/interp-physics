"""E(3)-equivariant GNN with velocity integration (Assignment 2 baseline)."""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp

from interp_physics.models.base import ModelOutput


@dataclass(frozen=True)
class EGNNConfig:
    d_hidden: int = 64
    d_msg: int = 32
    n_layers: int = 4
    return_activations: bool = False


class EGNN(nn.Module):
    """E(3) Equivariant GNN with velocity for dynamics prediction."""

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
        want_act = cfg.return_activations if return_activations is None else return_activations

        h = nn.Dense(cfg.d_hidden)(node_feat)
        x = pos
        v = vel
        acts: list[jnp.ndarray] = []

        for _ in range(cfg.n_layers):
            rel = x[:, :, None, :] - x[:, None, :, :]
            dist_sq = jnp.sum(rel**2, axis=-1, keepdims=True)

            m_ij = nn.relu(
                nn.Dense(cfg.d_msg)(h)[:, :, None, :]
                + nn.Dense(cfg.d_msg)(h)[:, None, :, :]
                + nn.Dense(cfg.d_msg)(dist_sq)
            )
            m_ij = m_ij * edges[:, :, :, None]

            x_weight = jnp.tanh(nn.Dense(1)(m_ij))
            coord_shift = (rel * x_weight * edges[:, :, :, None]).sum(axis=2)
            v = v + coord_shift

            agg = m_ij.sum(axis=2)
            h = h + nn.relu(nn.Dense(cfg.d_hidden)(jnp.concatenate([h, agg], axis=-1)))
            if want_act:
                acts.append(h)

        x = x + v
        return ModelOutput(pred_pos=x, activations=acts if want_act else None)


class EGNNFlax(EGNN):
    """Alias kept for clarity in configs."""

    pass
