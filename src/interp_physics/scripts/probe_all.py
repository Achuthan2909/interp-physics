"""Train every architecture, probe it against its untrained copy, ablate good scores."""

from __future__ import annotations

import json
from pathlib import Path

import jax.random as jr
import yaml

from interp_physics.data.coulomb import load_dataset
from interp_physics.models.registry import build_model
from interp_physics.probes.ablate import ablate_good_probes
from interp_physics.probes.suite import format_probe_table, run_probe_suite
from interp_physics.train import TrainConfig, load_params, save_checkpoint, train_model

CONFIGS = {
    "egnn": "configs/egnn_baseline.yaml",
    "attention_egnn": "configs/attention_egnn.yaml",
    "non_eq_gnn": "configs/non_eq_gnn.yaml",
    "mlp": "configs/mlp.yaml",
    "hnn": "configs/hnn.yaml",
    "equivariant_transformer": "configs/equivariant_transformer.yaml",
}


def _index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["target"], row["layer"]): row for row in rows}


def _ensure_trained(name: str, cfg: dict, train_data: dict, val_data: dict) -> tuple[object, object, Path]:
    model_kw = dict(cfg.get("model_kwargs", {}))
    model_kw["return_activations"] = False
    model = build_model(name, **model_kw)
    init = model.init(
        jr.PRNGKey(0),
        train_data["node_feat"][:1],
        train_data["pos"][:1],
        train_data["vel"][:1],
        train_data["edges"][:1],
    )
    out = Path(cfg.get("out", f"artifacts/runs/{name}"))
    ckpt = out / "params.msgpack"
    if ckpt.exists():
        print(f"Loading {name} from {out}")
        params = load_params(out, init)
    else:
        train_kw = cfg.get("train", {})
        tcfg = TrainConfig(
            n_epochs=int(train_kw.get("n_epochs", 100)),
            batch_size=int(train_kw.get("batch_size", 64)),
            lr=float(train_kw.get("lr", 1e-3)),
            seed=int(train_kw.get("seed", 0)),
            log_every=int(train_kw.get("log_every", 20)),
        )
        print(f"Training {name} for {tcfg.n_epochs} epochs")
        params, train_losses, val_losses = train_model(model, train_data, val_data, tcfg)
        save_checkpoint(
            out,
            params=params,
            config={"model": name, "model_kwargs": model_kw, "train": train_kw},
            metrics={
                "train_mse_final": train_losses[-1],
                "val_mse_final": val_losses[-1],
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
        )
        print(f"  val MSE {val_losses[-1]:.6f}")
    return model, params, out


def main() -> None:
    data_dir = Path("artifacts/data")
    train_data = load_dataset(data_dir / "train.npz")
    val_data = load_dataset(data_dir / "val.npz")
    summary: dict = {"models": {}}

    for name, config_path in CONFIGS.items():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model, params, out = _ensure_trained(name, cfg, train_data, val_data)
        init = model.init(
            jr.PRNGKey(0),
            train_data["node_feat"][:1],
            train_data["pos"][:1],
            train_data["vel"][:1],
            train_data["edges"][:1],
        )
        print(f"Probing {name}")
        trained = run_probe_suite(model, params, train_data, val_data)
        fresh = run_probe_suite(model, init, train_data, val_data)
        fresh_idx = _index(fresh)
        for row in trained:
            match = fresh_idx.get((row["target"], row["layer"]))
            if match is not None and row.get("r2") is not None:
                row["r2_init"] = match.get("r2")
            if match is not None and row.get("cosine") is not None:
                row["cosine_init"] = match.get("cosine")
        print(format_probe_table(trained))
        print(f"Ablating {name} readouts with R² >= 0.8")
        ablations = ablate_good_probes(model, params, train_data, val_data, trained)
        for row in ablations:
            print(
                f"  {row['target']:<18} {row['layer']:<10} "
                f"R2 {row['r2']:.3f}  MSE {row['mse']:.4f} -> {row['mse_ablated']:.4f}  "
                f"delta {row['mse_increase']:+.4f}"
            )
        summary["models"][name] = {"probes": trained, "ablations": ablations}
        (out / "probes.json").write_text(json.dumps({"model": name, "results": trained, "ablations": ablations}, indent=2))

    dest = Path("artifacts/probes/all.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
