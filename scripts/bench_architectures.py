"""Train all registered architectures and write a scoreboard JSON."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from interp_physics.data.coulomb import load_dataset
from interp_physics.models.registry import AVAILABLE_MODELS, build_model
from interp_physics.train import TrainConfig, save_checkpoint, train_model

CONFIGS = {
    "egnn": "configs/egnn_baseline.yaml",
    "attention_egnn": "configs/attention_egnn.yaml",
    "equivariant_transformer": "configs/equivariant_transformer.yaml",
    "hnn": "configs/hnn.yaml",
    "non_eq_gnn": "configs/non_eq_gnn.yaml",
    "mlp": "configs/mlp.yaml",
}


def main() -> None:
    data_dir = Path("artifacts/data_bench")
    out_root = Path("artifacts/runs/bench")
    train_data = load_dataset(data_dir / "train.npz")
    val_data = load_dataset(data_dir / "val.npz")

    scoreboard = {
        "task": "Predict final particle positions after Coulomb dynamics (T=1.0, 1000 steps)",
        "data": {
            "n_train": int(train_data["pos"].shape[0]),
            "n_val": int(val_data["pos"].shape[0]),
            "n_particles": int(train_data["pos"].shape[1]),
            "n_steps": 1000,
            "dt": 0.001,
        },
        "train": {"n_epochs": 40, "batch_size": 64, "lr": 1e-3},
        "models": {},
    }

    for name in AVAILABLE_MODELS:
        cfg_path = Path(CONFIGS[name])
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        model_kw = dict(cfg.get("model_kwargs", {}))
        # Keep HNN cheaper but still meaningful
        if name == "hnn":
            model_kw.setdefault("n_integrate", 10)
            batch_size = 32
        else:
            batch_size = 64

        print(f"\n===== {name} =====")
        model = build_model(name, **model_kw)
        tcfg = TrainConfig(n_epochs=40, batch_size=batch_size, lr=1e-3, seed=0, log_every=10)
        params, train_losses, val_losses = train_model(model, train_data, val_data, tcfg)
        run_dir = out_root / name
        save_checkpoint(
            run_dir,
            params=params,
            config={"model": name, "model_kwargs": model_kw, "train": scoreboard["train"]},
            metrics={
                "train_mse_final": train_losses[-1],
                "val_mse_final": val_losses[-1],
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
        )
        scoreboard["models"][name] = {
            "val_mse": val_losses[-1],
            "train_mse": train_losses[-1],
            "best_val_mse": min(val_losses),
            "best_epoch": int(min(range(len(val_losses)), key=lambda i: val_losses[i]) + 1),
            "run_dir": str(run_dir),
        }
        print(f"{name}: val={val_losses[-1]:.6f} best={min(val_losses):.6f}")

    out_json = out_root / "scoreboard.json"
    out_root.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(scoreboard, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
