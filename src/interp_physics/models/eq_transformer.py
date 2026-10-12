"""Simplified equivariant transformer (Equiformer-inspired, not full irreps).

Uses invariant multi-head attention over (features, distances) to produce
scalar weights that scale relative vectors — preserving E(3) equivariance
without SO(3) irreducible representations.
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from interp_physics.models.base import ModelOutput
from interp_physics.models.project import project_out


@dataclass(frozen=True)
class EquivariantTransformerConfig:
    d_hidden: int = 64
    n_heads: int = 4
    n_layers: int = 4
    return_activations: bool = False


class EquivariantTransformer(nn.Module):
    """Lightweight equivariant attention dynamics model."""

    config: EquivariantTransformerConfig = EquivariantTransformerConfig()

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
        assert cfg.d_hidden % cfg.n_heads == 0
        d_head = cfg.d_hidden // cfg.n_heads

        h = nn.Dense(cfg.d_hidden)(node_feat)
        x = pos
        v = vel
        acts: list[jnp.ndarray] = []
        msgs: list[jnp.ndarray] = []
        edge_updates: list[jnp.ndarray] = []
        if ablate_basis is not None and ablate_site == "node" and ablate_layer == 0:
            h = project_out(h, ablate_basis)
        if want:
            acts.append(h)

        for layer_i in range(cfg.n_layers):
            rel = x[:, :, None, :] - x[:, None, :, :]
            dist_sq = jnp.sum(rel**2, axis=-1, keepdims=True)

            # Invariant content for attention: node feats + distance
            # ReLU so the content can depend on the two charges jointly.
            content = nn.relu(
                nn.Dense(cfg.d_hidden)(h)[:, :, None, :]
                + nn.Dense(cfg.d_hidden)(h)[:, None, :, :]
                + nn.Dense(cfg.d_hidden)(dist_sq)
            )
            if ablate_basis is not None and ablate_site == "message" and ablate_layer == layer_i:
                content = project_out(content, ablate_basis)

            q = nn.Dense(cfg.d_hidden)(content)
            k = nn.Dense(cfg.d_hidden)(content)
            # reshape to heads: [B, N, N, H, D_h]
            b, n, _, _ = content.shape
            q = q.reshape(b, n, n, cfg.n_heads, d_head)
            k = k.reshape(b, n, n, cfg.n_heads, d_head)
            logits = (q * k).sum(-1) / jnp.sqrt(d_head)  # [B,N,N,H]
            logits = jnp.where(edges[..., None] > 0, logits, -1e9)
            attn = jax.nn.softmax(logits, axis=2)  # attend over senders

            # Equivariant coordinate update: attention-weighted relative vectors
            # mean over heads → scalar weight per edge
            w = attn.mean(axis=-1, keepdims=True)  # [B,N,N,1]
            # Small initial readout keeps tanh out of saturation at the start.
            w = jnp.tanh(nn.Dense(1, kernel_init=nn.initializers.normal(1e-3))(content)) * w
            edge_vec = rel * w * edges[:, :, :, None]
            v = v + edge_vec.sum(axis=2)

            # Feature update from attended content
            agg = (content * attn.mean(axis=-1, keepdims=True) * edges[:, :, :, None]).sum(axis=2)
            h = h + nn.relu(nn.Dense(cfg.d_hidden)(jnp.concatenate([h, agg], axis=-1)))
            if ablate_basis is not None and ablate_site == "node" and ablate_layer == layer_i + 1:
                h = project_out(h, ablate_basis)
            if want:
                acts.append(h)
                msgs.append(content)
                edge_updates.append(edge_vec)

        return ModelOutput(
            pred_pos=x + v,
            activations=acts if want else None,
            messages=msgs if want else None,
            edge_updates=edge_updates if want else None,
        )
