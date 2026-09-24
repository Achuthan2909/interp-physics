# Interp-Physics

Mechanistic interpretability across a spectrum of physics-informed neural architectures on **Coulomb particle dynamics**.

See `docs/proposal/proposal.tex` for the thesis claim.

## Setup

```bash
cd /Users/achu/Projects/Interp-Physics
uv sync
```

Always run through `uv` (project `.venv`), not global Python.

## Scaffolding layout

```
src/interp_physics/
  data/         Coulomb simulator + analytic probe labels
  models/       EGNN baseline (+ registry for HNN / Equiformer later)
  train/        shared train / checkpoint helpers
  probes/       Phase 1 stubs
  causal/       Phase 2 stubs
  geometry/     Phase 3 CKA stubs
  analysis/     plots
  scripts/      CLI: make-data, train
configs/        YAML run configs
artifacts/      datasets + checkpoints (gitignored)
docs/           proposal + thesis
```

## Quickstart (smoke)

```bash
# 1. Tiny dataset
uv run interp-make-data --smoke --out artifacts/data_smoke

# 2. Short train
uv run interp-train --config configs/egnn_smoke.yaml --data artifacts/data_smoke --smoke
```

## Full Assignment-2-scale data

```bash
uv run interp-make-data --out artifacts/data
uv run interp-train --config configs/egnn_baseline.yaml --data artifacts/data
```

## Build order (proposal phases)

1. **Data + EGNN baseline** ← you are here  
2. Activation hooks + linear probes (distance, charge product, force, energy)  
3. More architectures under the same `build_model` registry  
4. Causal ablation / activation patching  
5. CKA across models  
6. Grokking / data-size sweeps  

## Notes

- Simulator is ported from DS 595 Assignment 2 (JAX semi-implicit Euler, soft distance cutoff).
- Dataset includes analytic labels for probing at initial positions.
- EGNN `ModelOutput` can return per-layer node activations when `return_activations=True`.
