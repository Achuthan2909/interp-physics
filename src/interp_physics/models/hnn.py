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
from interp_physics.models.project import project_out


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
        *,
        capture: bool = False,
        ablate_layer: int | None = None,
        ablate_basis: jnp.ndarray | None = None,
        ablate_embed: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, list[jnp.ndarray]]:
        h = nn.Dense(self.d_hidden)(node_feat)
        if ablate_embed is not None:
            h = project_out(h, ablate_embed)
        dist_sq = jnp.sum((pos[:, :, None, :] - pos[:, None, :, :]) ** 2, axis=-1, keepdims=True)
        dist = jnp.sqrt(jnp.maximum(dist_sq, self.soft_cutoff_sq))

        pair = (
            nn.Dense(self.d_hidden)(h)[:, :, None, :]
            + nn.Dense(self.d_hidden)(h)[:, None, :, :]
            + nn.Dense(self.d_hidden)(dist)
        )
        layers: list[jnp.ndarray] = []
        for layer_i in range(self.n_layers):
            pair = nn.relu(nn.Dense(self.d_hidden)(pair))
            if ablate_basis is not None and ablate_layer == layer_i:
                pair = project_out(pair, ablate_basis)
            if capture:
                layers.append(pair)

        e_ij = nn.Dense(1)(pair).squeeze(-1) * edges
        v = 0.5 * e_ij.sum(axis=(-1, -2))
        if capture:
            return v, h, layers
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
        ablate_site: str | None = None,
        ablate_layer: int | None = None,
        ablate_basis: jnp.ndarray | None = None,
    ) -> ModelOutput:
        cfg = self.config
        want = cfg.return_activations if return_activations is None else return_activations
        potential = PairPotential(
            d_hidden=cfg.d_hidden,
            n_layers=cfg.n_layers_potential,
            soft_cutoff_sq=cfg.soft_cutoff_sq,
        )
        pair_basis = ablate_basis if ablate_site == "message" else None
        pair_layer = ablate_layer if ablate_site == "message" else None
        embed_basis = ablate_basis if ablate_site == "node" else None

        def hamiltonian(q, p):
            kinetic = 0.5 * jnp.sum(p**2, axis=(-1, -2))
            v, _embed = potential(
                q,
                node_feat,
                edges,
                ablate_layer=pair_layer,
                ablate_basis=pair_basis,
                ablate_embed=embed_basis,
            )
            return kinetic + v

        q, p = pos, vel

        def energy_sum(q_, p_):
            return hamiltonian(q_, p_).sum()

        for _ in range(cfg.n_integrate):
            dH_dq = jax.grad(energy_sum, argnums=0)(q, p)
            p = p - cfg.dt * dH_dq
            dH_dp = jax.grad(energy_sum, argnums=1)(q, p)
            q = q + cfg.dt * dH_dp

        acts = None
        msgs = None
        extras = None
        if want:
            _v, embed, layers = potential(pos, node_feat, edges, capture=True)
            acts = [embed]
            msgs = layers
            model_force = -jax.grad(energy_sum, argnums=0)(pos, vel)
            extras = {"model_force": model_force}

        return ModelOutput(
            pred_pos=q,
            activations=acts,
            messages=msgs,
            extras=extras,
        )
