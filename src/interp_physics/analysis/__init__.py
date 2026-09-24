"""Plotting helpers."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(
    train_losses: list[float],
    val_losses: list[float],
    title: str = "",
    path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title(title or "Learning curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_trajectories(
    model,
    params: Any,
    data: dict[str, Any],
    n_examples: int = 4,
    title: str = "",
    path: str | None = None,
):
    out = model.apply(
        params,
        data["node_feat"][:n_examples],
        data["pos"][:n_examples],
        data["vel"][:n_examples],
        data["edges"][:n_examples],
    )
    pred = np.asarray(out.pred_pos if hasattr(out, "pred_pos") else out)

    fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 4))
    if n_examples == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        target = data["target_pos"][idx]
        predicted = pred[idx]
        initial = data["pos"][idx]
        charges = data["node_feat"][idx][:, 1] - data["node_feat"][idx][:, 0]
        colors = ["#4477AA" if q < 0 else "#CC6677" for q in charges]

        ax.scatter(initial[:, 0], initial[:, 1], c=colors, s=50, alpha=0.3, label="initial")
        ax.scatter(
            target[:, 0],
            target[:, 1],
            c=colors,
            s=80,
            edgecolors="black",
            linewidths=1.2,
            label="truth",
        )
        ax.scatter(predicted[:, 0], predicted[:, 1], c=colors, s=80, marker="x", linewidths=2, label="pred")
        ax.set_aspect("equal")
        ax.set_title(f"System {idx}")

    if title:
        fig.suptitle(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
