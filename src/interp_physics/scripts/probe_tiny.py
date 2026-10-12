"""Same probe and deletion test on a much smaller copy of each network."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.random as jr

from interp_physics.data.coulomb import load_dataset
from interp_physics.models.registry import build_model
from interp_physics.probes.ablate import ablate_good_probes
from interp_physics.probes.suite import format_probe_table, run_probe_suite
from interp_physics.scripts.probe_all import _index
from interp_physics.train import TrainConfig, load_params, save_checkpoint, train_model

# One block, and about 8x narrower than the trained suite.
TINY = {
    "egnn": {"d_hidden": 8, "d_msg": 4, "n_layers": 1},
    "attention_egnn": {"d_hidden": 8, "d_msg": 4, "n_layers": 1},
    "non_eq_gnn": {"d_hidden": 8, "d_msg": 4, "n_layers": 1},
    "mlp": {"d_hidden": 16, "n_layers": 1},
    "hnn": {"d_hidden": 8, "n_layers_potential": 1, "n_integrate": 10, "dt": 0.1},
    "equivariant_transformer": {"d_hidden": 8, "n_heads": 2, "n_layers": 1},
}


def _n_params(params) -> int:
    return int(sum(x.size for x in jax.tree.leaves(params)))


def main() -> None:
    train_data = load_dataset(Path("artifacts/data/train.npz"))
    val_data = load_dataset(Path("artifacts/data/val.npz"))
    summary: dict = {"size": TINY, "models": {}}
    tcfg = TrainConfig(n_epochs=100, batch_size=64, lr=1e-3, seed=0, log_every=20)

    for name, model_kw in TINY.items():
        kw = {**model_kw, "return_activations": False}
        model = build_model(name, **kw)
        init = model.init(
            jr.PRNGKey(0),
            train_data["node_feat"][:1],
            train_data["pos"][:1],
            train_data["vel"][:1],
            train_data["edges"][:1],
        )
        out = Path("artifacts/runs/tiny") / name
        print(f"{name}: { _n_params(init) } parameters, {model_kw}")
        if (out / "params.msgpack").exists():
            params = load_params(out, init)
            metrics = json.loads((out / "metrics.json").read_text())
            print(f"  loaded, val MSE {metrics['val_mse_final']:.6f}")
        else:
            params, train_losses, val_losses = train_model(model, train_data, val_data, tcfg)
            save_checkpoint(
                out,
                params=params,
                config={"model": name, "model_kwargs": kw, "train": {"n_epochs": 100}},
                metrics={
                    "train_mse_final": train_losses[-1],
                    "val_mse_final": val_losses[-1],
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "n_params": _n_params(params),
                },
            )
            print(f"  val MSE {val_losses[-1]:.6f}")

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
        ablations = ablate_good_probes(model, params, train_data, val_data, trained)
        for row in ablations:
            print(
                f"  {row['target']:<18} {row['layer']:<10} "
                f"R2 {row['r2']:.3f}  MSE {row['mse']:.4f} -> {row['mse_ablated']:.4f}  "
                f"delta {row['mse_increase']:+.4f}"
            )
        summary["models"][name] = {
            "n_params": _n_params(params),
            "probes": trained,
            "ablations": ablations,
        }
        (out / "probes.json").write_text(json.dumps(summary["models"][name], indent=2))

    dest = Path("artifacts/probes/tiny.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
