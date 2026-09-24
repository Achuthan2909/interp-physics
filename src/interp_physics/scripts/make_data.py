"""Generate and cache the Coulomb dynamics dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from interp_physics.data.coulomb import (
    CoulombConfig,
    generate_charged_particles,
    save_dataset,
    split_dataset,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate Coulomb particle dataset")
    p.add_argument("--out", type=Path, default=Path("artifacts/data"))
    p.add_argument("--n-samples", type=int, default=6400)
    p.add_argument("--n-particles", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--n-train", type=int, default=2400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="Tiny dataset for a quick check")
    args = p.parse_args(argv)

    if args.smoke:
        cfg = CoulombConfig(
            n_samples=32,
            n_particles=5,
            n_steps=50,
            dt=0.001,
            seed=args.seed,
        )
        n_train = 16
    else:
        cfg = CoulombConfig(
            n_samples=args.n_samples,
            n_particles=args.n_particles,
            n_steps=args.n_steps,
            dt=args.dt,
            seed=args.seed,
        )
        n_train = args.n_train

    print(f"Generating Coulomb dataset: {cfg}")
    data = generate_charged_particles(cfg)
    meta = data["meta"]
    print(
        f"Done. mean|Δx|={meta['mean_displacement']:.4f}, "
        f"max|Δx|={meta['max_displacement']:.4f}, T={meta['total_time']:.2f}"
    )

    train, val = split_dataset(data, n_train=n_train)
    args.out.mkdir(parents=True, exist_ok=True)
    save_dataset(data, args.out / "full.npz")
    save_dataset(train, args.out / "train.npz")
    save_dataset(val, args.out / "val.npz")
    print(f"Wrote {args.out / 'full.npz'}, train.npz, val.npz")
    print(f"Probe labels available: pairwise_dist, charge_product, force_*, potential_energy")


if __name__ == "__main__":
    main()
