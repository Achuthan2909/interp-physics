"""Fit linear probes on a trained checkpoint and print val scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.random as jr
import yaml

from interp_physics.data.coulomb import load_dataset
from interp_physics.models.registry import build_model
from interp_physics.probes import format_probe_table, run_probe_suite
from interp_physics.train import load_params


def _load_yaml(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Linear probes on a trained dynamics model")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("artifacts/data"))
    p.add_argument("--config", type=Path, default=None, help="Used when the checkpoint has no config.json")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    ckpt_cfg_path = args.checkpoint / "config.json"
    if ckpt_cfg_path.exists():
        cfg = json.loads(ckpt_cfg_path.read_text())
    else:
        cfg = _load_yaml(args.config)
    model_name = cfg.get("model", "egnn")
    model_kw = dict(cfg.get("model_kwargs", {}))
    model_kw["return_activations"] = False

    train_data = load_dataset(args.data / "train.npz")
    val_data = load_dataset(args.data / "val.npz")
    model = build_model(model_name, **model_kw)
    init = model.init(
        jr.PRNGKey(0),
        train_data["node_feat"][:1],
        train_data["pos"][:1],
        train_data["vel"][:1],
        train_data["edges"][:1],
    )
    params = load_params(args.checkpoint, init)

    print(
        f"Probing {model_name} | train={train_data['pos'].shape[0]} "
        f"val={val_data['pos'].shape[0]} N={train_data['pos'].shape[1]}"
    )
    results = run_probe_suite(
        model,
        params,
        train_data,
        val_data,
        batch_size=args.batch_size,
        l2=args.l2,
        seed=args.seed,
    )
    print(format_probe_table(results))

    out = args.out or (args.checkpoint / "probes.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_name, "results": results}
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
