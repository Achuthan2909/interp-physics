"""Shared model output types for training and interpretability hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


@dataclass
class ModelOutput:
    """Prediction plus optional per-layer activations for probing / CKA."""

    pred_pos: jnp.ndarray
    activations: list[jnp.ndarray] | None = None  # node features h after each layer
    extras: dict[str, Any] | None = None
