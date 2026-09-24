"""Unconstrained MLP baseline — no graph or geometric structure."""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp

from interp_physics.models.base import ModelOutput


@dataclass(frozen=True)
class MLPConfig:
    d_hidden: int = 256
    n_layers: int = 4
    return_activations: bool = False


class MLPDynamics(nn.Module):
    """Flatten (node_feat, pos, vel) → MLP → predicted positions."""

    config: MLPConfig = MLPConfig()

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
        del edges  # unused — unconstrained
        cfg = self.config
        want = cfg.return_activations if return_activations is None else return_activations
        b, n, _ = pos.shape

        x = jnp.concatenate(
            [node_feat.reshape(b, -1), pos.reshape(b, -1), vel.reshape(b, -1)],
            axis=-1,
        )
        acts: list[jnp.ndarray] = []
        for _ in range(cfg.n_layers):
            x = nn.relu(nn.Dense(cfg.d_hidden)(x))
            if want:
                # Broadcast MLP state to a pseudo node activation [B, N, D]
                acts.append(jnp.broadcast_to(x[:, None, :], (b, n, cfg.d_hidden)))

        delta = nn.Dense(n * 3)(x).reshape(b, n, 3)
        return ModelOutput(pred_pos=pos + delta, activations=acts if want else None)
