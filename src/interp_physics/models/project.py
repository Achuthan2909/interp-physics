"""Remove a linear readout subspace from an activation."""

from __future__ import annotations

import jax.numpy as jnp


def project_out(x: jnp.ndarray, basis: jnp.ndarray) -> jnp.ndarray:
    """Subtract the components of ``x`` that lie in the columns of ``basis``.

    ``basis`` has shape ``[D, K]``. Zero columns are ignored, so a padded
    basis removes nothing along that column.
    """
    basis = jnp.asarray(basis)
    if basis.ndim == 1:
        basis = basis[:, None]
    scale = jnp.linalg.norm(basis, axis=0)
    unit = basis / (scale + 1e-12)
    keep = scale > 1e-8
    coef = jnp.einsum("...d,dk->...k", x, unit) * keep
    return x - jnp.einsum("...k,dk->...d", coef, unit)
