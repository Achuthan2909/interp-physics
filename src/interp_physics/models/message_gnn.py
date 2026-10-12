"""Shared pairwise EGNN-style message passing with optional ablations."""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from interp_physics.models.base import ModelOutput
from interp_physics.models.project import project_out


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
        ablate_site: str | None = None,
        ablate_layer: int | None = None,
        ablate_basis: jnp.ndarray | None = None,
    ) -> ModelOutput:
        cfg = self.config
        want_act = cfg.return_activations if return_activations is None else return_activations

        h = nn.Dense(cfg.d_hidden)(node_feat)
        x = pos
        v = vel if cfg.use_velocity else jnp.zeros_like(vel)
        acts: list[jnp.ndarray] = []
        msgs: list[jnp.ndarray] = []
        edge_updates: list[jnp.ndarray] = []
        if ablate_basis is not None and ablate_site == "node" and ablate_layer == 0:
            h = project_out(h, ablate_basis)
        if want_act:
            acts.append(h)

        for layer_i in range(cfg.n_layers):
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
            if ablate_basis is not None and ablate_site == "message" and ablate_layer == layer_i:
                m_ij = project_out(m_ij, ablate_basis)

            if cfg.use_attention:
                logits = nn.Dense(1)(m_ij).squeeze(-1)
                logits = jnp.where(edges > 0, logits, -1e9)
                attn = jax.nn.softmax(logits, axis=-1)
                m_ij = m_ij * attn[:, :, :, None]

            x_weight = nn.Dense(1)(m_ij)
            if cfg.use_tanh:
                x_weight = jnp.tanh(x_weight)
            edge_vec = rel * x_weight * edges[:, :, :, None]
            v = v + edge_vec.sum(axis=2)

            agg = m_ij.sum(axis=2)
            h_new = nn.relu(nn.Dense(cfg.d_hidden)(jnp.concatenate([h, agg], axis=-1)))
            h = h + h_new if cfg.use_residual else h_new
            if ablate_basis is not None and ablate_site == "node" and ablate_layer == layer_i + 1:
                h = project_out(h, ablate_basis)
            if want_act:
                acts.append(h)
                msgs.append(m_ij)
                edge_updates.append(edge_vec)

        return ModelOutput(
            pred_pos=x + v,
            activations=acts if want_act else None,
            messages=msgs if want_act else None,
            edge_updates=edge_updates if want_act else None,
        )
