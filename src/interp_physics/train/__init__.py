"""Shared training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax import serialization

EVAL_CHUNK = 500
TRAIN_EVAL_SYSTEMS = 4000


@dataclass
class TrainConfig:
    n_epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 0
    log_every: int = 20
    grad_clip: float = 1.0


def mse_pos(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((pred - target) ** 2)


def train_model(
    model,
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    config: TrainConfig | None = None,
) -> tuple[Any, list[float], list[float]]:
    """Train a Flax model with signature → ModelOutput (or raw array)."""
    cfg = config or TrainConfig()
    key = jr.PRNGKey(cfg.seed)
    params = model.init(
        key,
        train_data["node_feat"][:1],
        train_data["pos"][:1],
        train_data["vel"][:1],
        train_data["edges"][:1],
    )

    optimizer = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.lr))
    opt_state = optimizer.init(params)

    def _pred(p, nf, pos, vel, edges):
        out = model.apply(p, nf, pos, vel, edges)
        return out.pred_pos if hasattr(out, "pred_pos") else out

    @jax.jit
    def step(params, opt_state, node_feat, pos, vel, edges, target_pos):
        def loss_fn(p):
            pred = _pred(p, node_feat, pos, vel, edges)
            return mse_pos(pred, target_pos)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt, loss

    @jax.jit
    def eval_sq_err(params, node_feat, pos, vel, edges, target_pos):
        pred = _pred(params, node_feat, pos, vel, edges)
        return jnp.sum((pred - target_pos) ** 2)

    def eval_mse(params, data, n_systems):
        """MSE over the first ``n_systems`` systems, in chunks to bound memory."""
        total = 0.0
        for start in range(0, n_systems, EVAL_CHUNK):
            stop = min(start + EVAL_CHUNK, n_systems)
            total += float(
                eval_sq_err(
                    params,
                    data["node_feat"][start:stop],
                    data["pos"][start:stop],
                    data["vel"][start:stop],
                    data["edges"][start:stop],
                    data["target_pos"][start:stop],
                )
            )
        return total / data["target_pos"][:n_systems].size

    n_train = len(train_data["pos"])
    rng = np.random.default_rng(cfg.seed)
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(cfg.n_epochs):
        perm = rng.permutation(n_train)
        for i in range(0, n_train - cfg.batch_size + 1, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            params, opt_state, _ = step(
                params,
                opt_state,
                train_data["node_feat"][idx],
                train_data["pos"][idx],
                train_data["vel"][idx],
                train_data["edges"][idx],
                train_data["target_pos"][idx],
            )

        # Train error is tracked on a fixed subset so each epoch stays cheap.
        tr = eval_mse(params, train_data, min(n_train, TRAIN_EVAL_SYSTEMS))
        va = eval_mse(params, val_data, len(val_data["pos"]))
        train_losses.append(tr)
        val_losses.append(va)

        if (epoch + 1) % max(1, cfg.log_every) == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d} | Train MSE: {tr:.6f} | Val MSE: {va:.6f}")

    return params, train_losses, val_losses


def save_checkpoint(
    path: str | Path,
    *,
    params: Any,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "params.msgpack", "wb") as f:
        f.write(serialization.to_bytes(params))
    import json

    (path / "config.json").write_text(json.dumps(config, indent=2))
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return path


def load_params(path: str | Path, target_params: Any) -> Any:
    path = Path(path)
    with open(path / "params.msgpack", "rb") as f:
        return serialization.from_bytes(target_params, f.read())
