"""Train a registered model on cached Coulomb data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from interp_physics.data.coulomb import load_dataset
from interp_physics.models.registry import build_model
from interp_physics.train import TrainConfig, save_checkpoint, train_model


def _load_yaml(path: Path | None) -> dict:
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train physics dynamics model")
    p.add_argument("--config", type=Path, default=Path("configs/egnn_baseline.yaml"))
    p.add_argument("--data", type=Path, default=Path("artifacts/data"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--smoke", action="store_true", help="2-epoch sanity train")
    args = p.parse_args(argv)

    cfg = _load_yaml(args.config)
    model_name = cfg.get("model", "egnn")
    model_kw = cfg.get("model_kwargs", {})
    train_kw = cfg.get("train", {})

    if args.smoke:
        train_kw = {**train_kw, "n_epochs": 2, "batch_size": 8, "log_every": 1}

    train_data = load_dataset(args.data / "train.npz")
    val_data = load_dataset(args.data / "val.npz")
    print(
        f"Loaded train={train_data['pos'].shape[0]} val={val_data['pos'].shape[0]} "
        f"N={train_data['pos'].shape[1]}"
    )

    model = build_model(model_name, **model_kw)
    tcfg = TrainConfig(
        n_epochs=int(train_kw.get("n_epochs", 100)),
        batch_size=int(train_kw.get("batch_size", 64)),
        lr=float(train_kw.get("lr", 1e-3)),
        seed=int(train_kw.get("seed", 0)),
        log_every=int(train_kw.get("log_every", 20)),
    )

    print(f"Training {model_name} for {tcfg.n_epochs} epochs …")
    params, train_losses, val_losses = train_model(model, train_data, val_data, tcfg)

    out = args.out or Path(cfg.get("out", f"artifacts/runs/{model_name}"))
    metrics = {
        "train_mse_final": train_losses[-1],
        "val_mse_final": val_losses[-1],
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    save_checkpoint(
        out,
        params=params,
        config={"model": model_name, "model_kwargs": model_kw, "train": train_kw},
        metrics=metrics,
    )
    print(f"Saved checkpoint → {out}")
    print(f"Final Val MSE: {val_losses[-1]:.6f}")
    print(json.dumps({k: metrics[k] for k in ("train_mse_final", "val_mse_final")}, indent=2))


if __name__ == "__main__":
    main()
