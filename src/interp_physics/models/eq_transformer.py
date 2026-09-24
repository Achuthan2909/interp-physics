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
    ) -> ModelOutput:
        cfg = self.config
        want = cfg.return_activations if return_activations is None else return_activations
        assert cfg.d_hidden % cfg.n_heads == 0
        d_head = cfg.d_hidden // cfg.n_heads

        h = nn.Dense(cfg.d_hidden)(node_feat)
        x = pos
        v = vel
        acts: list[jnp.ndarray] = []

        for _ in range(cfg.n_layers):
            rel = x[:, :, None, :] - x[:, None, :, :]
            dist_sq = jnp.sum(rel**2, axis=-1, keepdims=True)

            # Invariant content for attention: node feats + distance
            content = (
                nn.Dense(cfg.d_hidden)(h)[:, :, None, :]
                + nn.Dense(cfg.d_hidden)(h)[:, None, :, :]
                + nn.Dense(cfg.d_hidden)(dist_sq)
            )

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
            w = jnp.tanh(nn.Dense(1)(content)) * w
            v = v + (rel * w * edges[:, :, :, None]).sum(axis=2)

            # Feature update from attended content
            agg = (content * attn.mean(axis=-1, keepdims=True) * edges[:, :, :, None]).sum(axis=2)
            h = h + nn.relu(nn.Dense(cfg.d_hidden)(jnp.concatenate([h, agg], axis=-1)))
            if want:
                acts.append(h)

        return ModelOutput(pred_pos=x + v, activations=acts if want else None)
