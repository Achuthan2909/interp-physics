"""Remove a high-scoring probe direction and measure the position error."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from interp_physics.probes.collect import collect_sites
from interp_physics.probes.ridge import readout_basis

GOOD_R2 = 0.8


def _offdiag_mask(n_particles: int) -> np.ndarray:
    return ~np.eye(n_particles, dtype=bool)


def _basis_for(
    sites: dict[str, np.ndarray | None],
    data: dict[str, Any],
    row: dict[str, Any],
    *,
    l2: float,
) -> tuple[str, int, np.ndarray] | None:
    """Return (site, layer index, basis) for one probe row, or None to skip."""
    target = row["target"]
    site = row["site"]
    layer_name = row["layer"]
    if target == "force_direction" or target == "net_force":
        return None

    if site == "node":
        layer = 0 if layer_name == "embed" else int(layer_name.split("_")[1])
        acts = sites["activations"][layer]
        x = acts.reshape(-1, acts.shape[-1])
        y = np.asarray(data["charges"]).reshape(-1)
        return "node", layer, readout_basis(x, y, l2=l2)

    if site in {"message", "mean_message"}:
        block = int(layer_name.split("_")[1]) - 1
        messages = sites["messages"][block]
        mask = _offdiag_mask(messages.shape[1])
        if site == "mean_message":
            x = messages[:, mask, :].mean(axis=1)
            y = np.asarray(data["potential_energy"])
        else:
            x = messages[:, mask, :].reshape(-1, messages.shape[-1])
            y = np.asarray(data[target])[:, mask].reshape(-1)
        return "message", block, readout_basis(x, y, l2=l2)

    return None


def _predict_mse(model, params: Any, data: dict[str, Any], *, batch_size: int, apply_kwargs: dict) -> float:
    @jax.jit
    def forward(p, node_feat, pos, vel, edges):
        out = model.apply(p, node_feat, pos, vel, edges, **apply_kwargs)
        return out.pred_pos

    chunks: list[np.ndarray] = []
    n = int(data["pos"].shape[0])
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        pred = forward(
            params,
            jnp.asarray(data["node_feat"][start:stop]),
            jnp.asarray(data["pos"][start:stop]),
            jnp.asarray(data["vel"][start:stop]),
            jnp.asarray(data["edges"][start:stop]),
        )
        chunks.append(np.asarray(pred))
    pred = np.concatenate(chunks, axis=0)
    return float(np.mean((pred - data["target_pos"][:n]) ** 2))


def ablate_good_probes(
    model,
    params: Any,
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    probe_rows: list[dict[str, Any]],
    *,
    batch_size: int = 64,
    l2: float = 1.0,
    r2_min: float = GOOD_R2,
) -> list[dict[str, Any]]:
    """Project out each probe with val R² at or above ``r2_min`` and remeasure MSE."""
    interesting = [
        row
        for row in probe_rows
        if row.get("r2") is not None and row["r2"] >= r2_min and row["target"] not in {"force_direction", "net_force"}
    ]
    if not interesting:
        return []

    sites = collect_sites(model, params, train_data, batch_size=batch_size)
    base_mse = _predict_mse(model, params, val_data, batch_size=batch_size, apply_kwargs={})
    results: list[dict[str, Any]] = []
    for row in interesting:
        spec = _basis_for(sites, train_data, row, l2=l2)
        if spec is None:
            continue
        site, layer, basis = spec
        if basis.shape[1] == 0:
            continue
        mse = _predict_mse(
            model,
            params,
            val_data,
            batch_size=batch_size,
            apply_kwargs={
                "ablate_site": site,
                "ablate_layer": layer,
                "ablate_basis": jnp.asarray(basis),
            },
        )
        results.append(
            {
                "target": row["target"],
                "layer": row["layer"],
                "site": row["site"],
                "r2": row["r2"],
                "mse": base_mse,
                "mse_ablated": mse,
                "mse_increase": mse - base_mse,
            }
        )
    return results
