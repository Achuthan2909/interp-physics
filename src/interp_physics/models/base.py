"""Shared model output types for training and interpretability hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


@dataclass
class ModelOutput:
    """Prediction plus optional per-layer tensors for probing / CKA.

    When activations are requested from a message-passing model:
      activations[0] is the charge embedding, before any message passing.
      activations[k] for k >= 1 is the node state h after block k.
      messages[k] is the pairwise message m_ij of block k+1, shape [B, N, N, D].
      edge_updates[k] is w_ij * (x_i - x_j) for that block, shape [B, N, N, 3].
    """

    pred_pos: jnp.ndarray
    activations: list[jnp.ndarray] | None = None
    messages: list[jnp.ndarray] | None = None
    edge_updates: list[jnp.ndarray] | None = None
    extras: dict[str, Any] | None = None
