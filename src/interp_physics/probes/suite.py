"""Fit one linear probe per layer for each physics label."""

from __future__ import annotations

from typing import Any

import numpy as np

from interp_physics.probes.collect import collect_sites
from interp_physics.probes.ridge import fit_ridge

PAIR_TARGETS = (
    ("pairwise_dist", "pairwise_dist"),
    ("charge_product", "charge_product"),
    ("force_magnitude", "force_magnitude"),
)


def _pair_index(n_particles: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n_particles, k=1)


def _unordered_pairs(messages: np.ndarray) -> np.ndarray:
    """Concatenate m_ij and m_ji for every unordered pair.

    messages: [batch, N, N, D] -> [batch, n_pairs, 2D]
    """
    ii, jj = _pair_index(messages.shape[1])
    return np.concatenate([messages[:, ii, jj, :], messages[:, jj, ii, :]], axis=-1)


def _rows(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a leading batch (and optional pair) axis into probe rows."""
    x = np.asarray(x)
    y = np.asarray(y)
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(x.shape[0], -1)
    if y.shape[1] == 1:
        y = y[:, 0]
    return x, y


def _score_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    l2: float,
    seed: int,
) -> dict[str, float | None]:
    x_train, y_train = _rows(x_train, y_train)
    x_val, y_val = _rows(x_val, y_val)
    fit = fit_ridge(x_train, y_train, x_val, y_val, l2=l2)
    rng = np.random.default_rng(seed)
    shuffled = y_train.copy()
    rng.shuffle(shuffled)
    control = fit_ridge(x_train, shuffled, x_val, y_val, l2=l2)
    return {
        "r2": fit["r2"],
        "r2_shuffle": control["r2"],
        "accuracy": fit["accuracy"],
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
    }


def _mean_cosine(edge: np.ndarray, force: np.ndarray) -> float:
    """Mean cosine over off-diagonal pairs. Diagonal forces are zero."""
    numer = np.sum(edge * force, axis=-1)
    denom = np.linalg.norm(edge, axis=-1) * np.linalg.norm(force, axis=-1)
    mask = denom > 1e-8
    if not np.any(mask):
        return float("nan")
    return float(np.mean(numer[mask] / denom[mask]))


def run_probe_suite(
    model,
    params: Any,
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    *,
    batch_size: int = 64,
    l2: float = 1.0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Probe train systems and score on val systems.

    Pairwise labels are read from the edge message, before messages are summed
    onto a node. Charge is read from the node state. Potential energy is read
    from the mean message. Force direction is the cosine between the model's
    edge update ``w_ij (x_i - x_j)`` and the true pairwise force.
    """
    train = collect_sites(model, params, train_data, batch_size=batch_size)
    val = collect_sites(model, params, val_data, batch_size=batch_size)
    results: list[dict[str, Any]] = []

    n_act = int(train["activations"].shape[0])
    has_messages = train["messages"] is not None
    if not has_messages:
        return _probe_global_state(train, val, train_data, val_data, n_act, l2=l2, seed=seed)

    for layer in range(n_act):
        if has_messages and layer == 0:
            site_name = "embed"
        elif has_messages:
            site_name = f"block_{layer}"
        else:
            site_name = f"block_{layer + 1}"
        scored = _score_split(
            train["activations"][layer],
            train_data["charges"],
            val["activations"][layer],
            val_data["charges"],
            l2=l2,
            seed=seed + layer,
        )
        results.append(
            {
                "target": "charge",
                "site": "node",
                "layer": site_name,
                **scored,
            }
        )

    if train["messages"] is None:
        return results

    n_blocks = int(train["messages"].shape[0])
    for block in range(n_blocks):
        layer_name = f"block_{block + 1}"
        x_train = _unordered_pairs(train["messages"][block])
        x_val = _unordered_pairs(val["messages"][block])
        for target_name, label_key in PAIR_TARGETS:
            ii, jj = _pair_index(train_data[label_key].shape[1])
            scored = _score_split(
                x_train,
                train_data[label_key][:, ii, jj],
                x_val,
                val_data[label_key][:, ii, jj],
                l2=l2,
                seed=seed + 100 + block,
            )
            results.append(
                {
                    "target": target_name,
                    "site": "message",
                    "layer": layer_name,
                    **scored,
                }
            )

        u_train = x_train.mean(axis=1)
        u_val = x_val.mean(axis=1)
        scored = _score_split(
            u_train,
            train_data["potential_energy"],
            u_val,
            val_data["potential_energy"],
            l2=l2,
            seed=seed + 200 + block,
        )
        results.append(
            {
                "target": "potential_energy",
                "site": "mean_message",
                "layer": layer_name,
                **scored,
            }
        )

        if val["edge_updates"] is not None:
            cosine = _mean_cosine(val["edge_updates"][block], val_data["pairwise_force"])
            shifted = np.roll(val_data["pairwise_force"], 1, axis=0)
            cosine_shift = _mean_cosine(val["edge_updates"][block], shifted)
            results.append(
                {
                    "target": "force_direction",
                    "site": "edge_update",
                    "layer": layer_name,
                    "cosine": cosine,
                    "cosine_shifted": cosine_shift,
                    "r2": None,
                    "r2_shuffle": None,
                    "accuracy": None,
                    "n_train": None,
                    "n_val": int(val_data["pos"].shape[0]),
                }
            )

    if val["model_force"] is not None:
        cosine = _mean_cosine(val["model_force"][:, None, :, :], val_data["net_force"][:, None, :, :])
        shifted = np.roll(val_data["net_force"], 1, axis=0)
        cosine_shift = _mean_cosine(val["model_force"][:, None, :, :], shifted[:, None, :, :])
        results.append(
            {
                "target": "net_force",
                "site": "hamiltonian_force",
                "layer": "initial",
                "cosine": cosine,
                "cosine_shifted": cosine_shift,
                "r2": None,
                "r2_shuffle": None,
                "accuracy": None,
                "n_train": None,
                "n_val": int(val_data["pos"].shape[0]),
            }
        )

    return results


def _probe_global_state(
    train: dict[str, Any],
    val: dict[str, Any],
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    n_act: int,
    *,
    l2: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Probe a model whose hidden state is one vector per system (the MLP)."""
    results: list[dict[str, Any]] = []
    n_particles = int(train_data["pos"].shape[1])
    ii, jj = _pair_index(n_particles)
    for layer in range(n_act):
        layer_name = f"block_{layer + 1}"
        x_train = train["activations"][layer][:, 0, :]
        x_val = val["activations"][layer][:, 0, :]
        scored = _score_split(
            x_train,
            train_data["charges"],
            x_val,
            val_data["charges"],
            l2=l2,
            seed=seed + layer,
        )
        results.append({"target": "charge", "site": "system", "layer": layer_name, **scored})
        for target_name, label_key in PAIR_TARGETS:
            scored = _score_split(
                x_train,
                train_data[label_key][:, ii, jj],
                x_val,
                val_data[label_key][:, ii, jj],
                l2=l2,
                seed=seed + 100 + layer,
            )
            results.append(
                {"target": target_name, "site": "system", "layer": layer_name, **scored}
            )
        scored = _score_split(
            x_train,
            train_data["potential_energy"],
            x_val,
            val_data["potential_energy"],
            l2=l2,
            seed=seed + 200 + layer,
        )
        results.append(
            {"target": "potential_energy", "site": "system", "layer": layer_name, **scored}
        )
    return results


def format_probe_table(results: list[dict[str, Any]]) -> str:
    """Plain-text table of probe scores."""
    lines = [
        f"{'target':<18} {'layer':<10} {'val R2':>8} {'shuffle':>8} {'acc':>8} {'cosine':>8} {'shifted':>8}",
        "-" * 74,
    ]
    for row in results:
        r2 = _fmt(row.get("r2"))
        shuf = _fmt(row.get("r2_shuffle"))
        acc = _fmt(row.get("accuracy"))
        cosine = _fmt(row.get("cosine"))
        shifted = _fmt(row.get("cosine_shifted"))
        lines.append(
            f"{row['target']:<18} {row['layer']:<10} {r2:>8} {shuf:>8} {acc:>8} {cosine:>8} {shifted:>8}"
        )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    return f"{value:.3f}"
