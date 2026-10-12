"""Run a trained model and stack the tensors a probe reads."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def collect_sites(
    model,
    params: Any,
    data: dict[str, Any],
    *,
    batch_size: int = 64,
) -> dict[str, np.ndarray | None]:
    """Return activations, messages, and edge updates for every system.

    Stacked axes are ``[layer, batch, ...]``. ``layer 0`` of activations is the
    charge embedding; later layers line up with messages and edge updates.
    """

    @jax.jit
    def forward(p, node_feat, pos, vel, edges):
        out = model.apply(p, node_feat, pos, vel, edges, return_activations=True)
        acts = jnp.stack(tuple(out.activations))
        msgs = jnp.stack(tuple(out.messages)) if out.messages else None
        updates = jnp.stack(tuple(out.edge_updates)) if out.edge_updates else None
        force = None
        if out.extras is not None and "model_force" in out.extras:
            force = out.extras["model_force"]
        return acts, msgs, updates, force

    n = int(data["pos"].shape[0])
    act_parts: list[np.ndarray] = []
    msg_parts: list[np.ndarray] = []
    upd_parts: list[np.ndarray] = []
    force_parts: list[np.ndarray] = []
    has_messages: bool | None = None
    has_updates: bool | None = None
    has_force: bool | None = None

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        acts, msgs, updates, force = forward(
            params,
            jnp.asarray(data["node_feat"][start:stop]),
            jnp.asarray(data["pos"][start:stop]),
            jnp.asarray(data["vel"][start:stop]),
            jnp.asarray(data["edges"][start:stop]),
        )
        act_parts.append(np.asarray(acts))
        if has_messages is None:
            has_messages = msgs is not None
            has_updates = updates is not None
            has_force = force is not None
        if has_messages:
            msg_parts.append(np.asarray(msgs))
        if has_updates:
            upd_parts.append(np.asarray(updates))
        if has_force:
            force_parts.append(np.asarray(force))

    activations = np.concatenate(act_parts, axis=1)
    messages = np.concatenate(msg_parts, axis=1) if has_messages else None
    edge_updates = np.concatenate(upd_parts, axis=1) if has_updates else None
    model_force = np.concatenate(force_parts, axis=0) if has_force else None
    return {
        "activations": activations,
        "messages": messages,
        "edge_updates": edge_updates,
        "model_force": model_force,
    }
