"""Shared pairwise EGNN-style message passing with optional ablations."""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from interp_physics.models.base import ModelOutput


@dataclass(frozen=True)
class MessageGNNConfig:
    d_hidden: int = 64
    d_msg: int = 32
    n_layers: int = 4
    use_velocity: bool = True
    use_distance: bool = True
    use_tanh: bool = True
    use_residual: bool = True
    break_equivariance: bool = False
    use_attention: bool = False
    return_activations: bool = False


class MessagePassingGNN(nn.Module):
    """EGNN family used by full EGNN, Attention EGNN, and Non-equivariant GNN."""

    config: MessageGNNConfig = MessageGNNConfig()

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
        v = vel if cfg.use_velocity else jnp.zeros_like(vel)
        acts: list[jnp.ndarray] = []

        for _ in range(cfg.n_layers):
            rel = x[:, :, None, :] - x[:, None, :, :]
            dist_sq = jnp.sum(rel**2, axis=-1, keepdims=True)

            msg = nn.Dense(cfg.d_msg)(h)[:, :, None, :] + nn.Dense(cfg.d_msg)(h)[:, None, :, :]
            if cfg.use_distance:
                msg = msg + nn.Dense(cfg.d_msg)(dist_sq)
            if cfg.break_equivariance:
                abs_ij = jnp.concatenate(
                    [
                        jnp.broadcast_to(x[:, :, None, :], rel.shape),
                        jnp.broadcast_to(x[:, None, :, :], rel.shape),
                    ],
                    axis=-1,
                )
                msg = msg + nn.Dense(cfg.d_msg)(abs_ij)

            m_ij = nn.relu(msg) * edges[:, :, :, None]

            if cfg.use_attention:
                logits = nn.Dense(1)(m_ij).squeeze(-1)
                logits = jnp.where(edges > 0, logits, -1e9)
                attn = jax.nn.softmax(logits, axis=-1)
                m_ij = m_ij * attn[:, :, :, None]

            x_weight = nn.Dense(1)(m_ij)
            if cfg.use_tanh:
                x_weight = jnp.tanh(x_weight)
            v = v + (rel * x_weight * edges[:, :, :, None]).sum(axis=2)

            agg = m_ij.sum(axis=2)
            h_new = nn.relu(nn.Dense(cfg.d_hidden)(jnp.concatenate([h, agg], axis=-1)))
            h = h + h_new if cfg.use_residual else h_new
            if want_act:
                acts.append(h)

        return ModelOutput(pred_pos=x + v, activations=acts if want_act else None)
