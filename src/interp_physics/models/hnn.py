"""Hamiltonian Neural Network (physics-explicit inductive bias).

Learns H = T(p) + V_θ(q, charges) and integrates Hamilton's equations
to predict final positions (FlatHGN-style pairwise potential).
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from interp_physics.models.base import ModelOutput


@dataclass(frozen=True)
class HNNConfig:
    d_hidden: int = 64
    n_layers_potential: int = 2
    n_integrate: int = 10
    dt: float = 0.1  # n_integrate * dt ≈ dataset total time (1.0)
    soft_cutoff_sq: float = 0.25
    return_activations: bool = False


class PairPotential(nn.Module):
    """Scalar pairwise potential V(q; charges), E(3)-invariant inputs only."""

    d_hidden: int = 64
    n_layers: int = 2
    soft_cutoff_sq: float = 0.25

    @nn.compact
    def __call__(
        self,
        pos: jnp.ndarray,
        node_feat: jnp.ndarray,
        edges: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = nn.Dense(self.d_hidden)(node_feat)
        dist_sq = jnp.sum((pos[:, :, None, :] - pos[:, None, :, :]) ** 2, axis=-1, keepdims=True)
        dist = jnp.sqrt(jnp.maximum(dist_sq, self.soft_cutoff_sq))

        pair = (
            nn.Dense(self.d_hidden)(h)[:, :, None, :]
            + nn.Dense(self.d_hidden)(h)[:, None, :, :]
            + nn.Dense(self.d_hidden)(dist)
        )
        for _ in range(self.n_layers):
            pair = nn.relu(nn.Dense(self.d_hidden)(pair))

        e_ij = nn.Dense(1)(pair).squeeze(-1) * edges
        v = 0.5 * e_ij.sum(axis=(-1, -2))
        return v, h


class HNN(nn.Module):
    """Integrate Hamilton dynamics from a learned pairwise potential."""

    config: HNNConfig = HNNConfig()

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
        potential = PairPotential(
            d_hidden=cfg.d_hidden,
            n_layers=cfg.n_layers_potential,
            soft_cutoff_sq=cfg.soft_cutoff_sq,
        )

        def hamiltonian(q, p):
            kinetic = 0.5 * jnp.sum(p**2, axis=(-1, -2))
            v, embed = potential(q, node_feat, edges)
            return kinetic + v, embed

        q, p = pos, vel
        acts: list[jnp.ndarray] = []

        def energy_sum(q_, p_):
            h, _ = hamiltonian(q_, p_)
            return h.sum()

        for _ in range(cfg.n_integrate):
            dH_dq = jax.grad(energy_sum, argnums=0)(q, p)
            p = p - cfg.dt * dH_dq
            dH_dp = jax.grad(energy_sum, argnums=1)(q, p)
            q = q + cfg.dt * dH_dp
            if want:
                _, embed = hamiltonian(q, p)
                acts.append(embed)

        extras = None
        if want:
            h_val, _ = hamiltonian(q, p)
            extras = {"hamiltonian": h_val}

        return ModelOutput(pred_pos=q, activations=acts if want else None, extras=extras)
