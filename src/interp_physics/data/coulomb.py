"""Coulomb N-body simulator and analytic physics labels.

Ported from DS 595 Assignment 2, extended with ground-truth quantities
for probing (pairwise distance, charge product, force, potential energy).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


@dataclass(frozen=True)
class CoulombConfig:
    n_samples: int = 6400
    n_particles: int = 10
    dt: float = 0.001
    n_steps: int = 1000
    soft_cutoff_sq: float = 0.25  # min dist^2 = 0.25 → r_min = 0.5
    pos_scale: float = 1.0
    vel_scale: float = 0.1
    seed: int = 42


def pairwise_coulomb_force(
    pos: jnp.ndarray,
    charges: jnp.ndarray,
    soft_cutoff_sq: float = 0.25,
) -> jnp.ndarray:
    """Net Coulomb force on each particle. pos: [N, 3], charges: [N]."""
    rel = pos[:, None, :] - pos[None, :, :]
    dist_sq = jnp.sum(rel**2, axis=-1, keepdims=True)
    dist = jnp.sqrt(jnp.maximum(dist_sq, soft_cutoff_sq))
    charge_prod = charges[:, None] * charges[None, :]
    force = charge_prod[:, :, None] * rel / (dist**3)
    mask = 1.0 - jnp.eye(pos.shape[0])[:, :, None]
    return (force * mask).sum(axis=1)


def compute_physics_labels(
    pos: np.ndarray,
    charges: np.ndarray,
    soft_cutoff_sq: float = 0.25,
) -> dict[str, np.ndarray]:
    """Analytic quantities at *initial* positions for probing.

    Returns arrays shaped for pairwise (i,j) or system-level probes:
      pairwise_dist:     [B, N, N]
      charge_product:    [B, N, N]
      force_magnitude:   [B, N, N]
      force_direction:   [B, N, N, 3]  (unit vector of F_ij; 0 on diag)
      pairwise_force:    [B, N, N, 3]
      net_force:         [B, N, 3]
      potential_energy:  [B]
    """
    pos_j = jnp.asarray(pos)
    charges_j = jnp.asarray(charges)

    def one(p, q):
        rel = p[:, None, :] - p[None, :, :]
        dist_sq = jnp.sum(rel**2, axis=-1)
        dist = jnp.sqrt(jnp.maximum(dist_sq, soft_cutoff_sq))
        qp = q[:, None] * q[None, :]
        # F_ij on i due to j (same convention as simulator)
        force_ij = qp[:, :, None] * rel / (dist[:, :, None] ** 3)
        mask = 1.0 - jnp.eye(p.shape[0])
        force_ij = force_ij * mask[:, :, None]
        mag = jnp.linalg.norm(force_ij, axis=-1)
        direction = jnp.where(mag[:, :, None] > 1e-8, force_ij / mag[:, :, None], 0.0)
        net = force_ij.sum(axis=1)
        # U = sum_{i<j} q_i q_j / r  (soft cutoff in denominator)
        u = 0.5 * jnp.sum(mask * qp / dist)
        return dist, qp, mag, direction, force_ij, net, u

    dist, qp, mag, direction, force_ij, net, u = jax.vmap(one)(pos_j, charges_j)
    return {
        "pairwise_dist": np.asarray(dist),
        "charge_product": np.asarray(qp),
        "force_magnitude": np.asarray(mag),
        "force_direction": np.asarray(direction),
        "pairwise_force": np.asarray(force_ij),
        "net_force": np.asarray(net),
        "potential_energy": np.asarray(u),
    }


def generate_charged_particles(
    config: CoulombConfig | None = None,
    *,
    key: jax.Array | None = None,
) -> dict[str, Any]:
    """Simulate Coulomb dynamics; return train-ready batch dict + labels."""
    cfg = config or CoulombConfig()
    if key is None:
        key = jr.PRNGKey(cfg.seed)
    k1, k2, k3 = jr.split(key, 3)

    pos = jr.normal(k1, (cfg.n_samples, cfg.n_particles, 3)) * cfg.pos_scale
    vel = jr.normal(k2, (cfg.n_samples, cfg.n_particles, 3)) * cfg.vel_scale
    charges = jr.choice(
        k3, jnp.array([-1.0, 1.0]), shape=(cfg.n_samples, cfg.n_particles)
    )

    edges = jnp.ones((cfg.n_particles, cfg.n_particles)) - jnp.eye(cfg.n_particles)
    edges = jnp.broadcast_to(edges, (cfg.n_samples, cfg.n_particles, cfg.n_particles))

    soft = cfg.soft_cutoff_sq
    dt = cfg.dt
    n_steps = cfg.n_steps

    def simulate(pos_i, vel_i, charges_i):
        def one_step(carry, _):
            p, v = carry
            f = pairwise_coulomb_force(p, charges_i, soft)
            v_new = v + f * dt
            p_new = p + v_new * dt
            return (p_new, v_new), None

        (pos_final, _), _ = jax.lax.scan(one_step, (pos_i, vel_i), None, length=n_steps)
        return pos_final

    target_pos = jax.vmap(simulate)(pos, vel, charges)

    node_feat = jnp.stack(
        [
            (charges == -1.0).astype(jnp.float32),
            (charges == +1.0).astype(jnp.float32),
        ],
        axis=-1,
    )

    pos_np = np.asarray(pos)
    charges_np = np.asarray(charges)
    labels = compute_physics_labels(pos_np, charges_np, soft)

    disp = np.abs(np.asarray(target_pos) - pos_np)
    meta = {
        **asdict(cfg),
        "total_time": cfg.dt * cfg.n_steps,
        "mean_displacement": float(disp.mean()),
        "max_displacement": float(disp.max()),
    }

    return {
        "pos": pos_np.astype(np.float32),
        "vel": np.asarray(vel).astype(np.float32),
        "charges": charges_np.astype(np.float32),
        "node_feat": np.asarray(node_feat).astype(np.float32),
        "edges": np.asarray(edges).astype(np.float32),
        "target_pos": np.asarray(target_pos).astype(np.float32),
        **labels,
        "meta": meta,
    }


def split_dataset(
    data: dict[str, Any],
    n_train: int = 2400,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split along the sample axis. Meta is copied to both."""
    meta = data.get("meta", {})
    train = {k: v[:n_train] for k, v in data.items() if k != "meta" and hasattr(v, "shape")}
    val = {k: v[n_train:] for k, v in data.items() if k != "meta" and hasattr(v, "shape")}
    train["meta"] = {**meta, "split": "train", "n": n_train}
    val["meta"] = {**meta, "split": "val", "n": len(next(iter(val.values())))}
    return train, val


def save_dataset(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in data.items() if k != "meta"}
    arrays["meta"] = np.asarray(data.get("meta", {}), dtype=object)
    np.savez_compressed(path, **arrays)
    return path


def load_dataset(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        if "meta" in z.files:
            meta = z["meta"]
            out["meta"] = meta.item() if getattr(meta, "shape", ()) == () else {}
        else:
            out["meta"] = {}
    return out
