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

## Architecture suite

| Name | Config | Inductive bias |
|------|--------|----------------|
| `egnn` | `configs/egnn_baseline.yaml` | E(3) equivariant GNN |
| `attention_egnn` | `configs/attention_egnn.yaml` | EGNN + message attention |
| `equivariant_transformer` | `configs/equivariant_transformer.yaml` | Equivariant multi-head attention (lite) |
| `hnn` | `configs/hnn.yaml` | Physics-explicit Hamiltonian |
| `non_eq_gnn` | `configs/non_eq_gnn.yaml` | Graph net without rotation equivariance |
| `mlp` | `configs/mlp.yaml` | Unconstrained MLP |

```bash
uv run python -c "from interp_physics.models import AVAILABLE_MODELS, build_model; print(AVAILABLE_MODELS)"
```

## Build order (proposal phases)

1. Data + EGNN baseline  
2. **Architecture registry** ← you are here  
3. Activation hooks + linear probes  
4. Causal ablation / activation patching  
5. CKA across models  
6. Grokking / data-size sweeps  

## Notes

- Simulator is ported from DS 595 Assignment 2 (JAX semi-implicit Euler, soft distance cutoff).
- Dataset includes analytic labels for probing at initial positions.
- Models return `ModelOutput`; set `return_activations=True` for probing hooks.
- `equivariant_transformer` is a lightweight equivariant-attention stand-in, not full Equiformer irreps.
