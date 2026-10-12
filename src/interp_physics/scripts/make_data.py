"""Generate and cache the Coulomb dynamics dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from interp_physics.data.coulomb import (
    CoulombConfig,
    generate_charged_particles,
    save_dataset,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate Coulomb particle dataset")
    p.add_argument("--out", type=Path, default=Path("artifacts/data"))
    p.add_argument("--n-train", type=int, default=38400)
    p.add_argument("--n-val", type=int, default=4000)
    p.add_argument("--n-particles", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42, help="Val uses this seed, train uses seed + 1")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    # Train and val are separate simulations, so the val set does not change
    # when the train set is made larger.
    for split, n_samples, seed in (("val", args.n_val, args.seed), ("train", args.n_train, args.seed + 1)):
        cfg = CoulombConfig(
            n_samples=n_samples,
            n_particles=args.n_particles,
            n_steps=args.n_steps,
            dt=args.dt,
            seed=seed,
        )
        print(f"Generating {split}: {cfg}")
        data = generate_charged_particles(cfg)
        meta = data["meta"]
        print(
            f"Done. mean|Δx|={meta['mean_displacement']:.4f}, "
            f"max|Δx|={meta['max_displacement']:.4f}, T={meta['total_time']:.2f}"
        )
        data["meta"] = {**meta, "split": split, "n": n_samples}
        save_dataset(data, args.out / f"{split}.npz")
        print(f"Wrote {args.out / f'{split}.npz'}")
    print("Probe labels available: pairwise_dist, charge_product, force_*, potential_energy")


if __name__ == "__main__":
    main()
