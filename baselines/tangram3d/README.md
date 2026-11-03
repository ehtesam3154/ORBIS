# Tangram-3D

Tangram (Biancalani et al.) maps single cells onto spatial measurements. Here the "spatial" side is a seeded uniform subsample of
the training cells with their 3-D positions, and each query cell takes the position of its most probable reference cell. The
official package is used unmodified.

- Official repository: https://github.com/broadinstitute/Tangram, commit `4c68995a418f41dc8caef567598c4d9b47781a13` (BSD 3-Clause).
- No patch: the configs put the clean checkout on `PYTHONPATH`.
- Driver: `tangram_3d.py`. Configs: `baselines/configs/tangram3d_<test>.sh`. The layout is written by `baselines/remap/to_layout.py`.

## Settings

- `tg.pp_adatas` + `tg.map_cells_to_space(mode="cells", density_prior="uniform")` (the authors' choice for single-cell-resolution
  spatial data), learning rate 0.1 (default). Query expression `normalize_total` (tutorial), reference raw counts.
- Position = 3-D coordinate of the most probable reference cell (the repository's argmax rule); the top-5 reference cells and
  probabilities are saved too. Allen: the saved Harmony input as reconstructed genes (as for REMAP).
- Query in 10,000-cell chunks. Reference size = the largest seeded subsample that fits one 20 GB GPU with those chunks:
  55,000 cells (blocks), 48,000 (embryos), 1,000 epochs (default). Brains and Allen: a 15,000-cell reference and 500 epochs (the value
  in Tangram's own tests), set by the compute budget.

## Run

```bash
DATA_DIR=... OUT_DIR=... PY_TANGRAM=/path/to/python GPU=0 bash baselines/configs/tangram3d_blockA.sh
```

Environment of the reported runs (Python 3.11): torch 2.6.0+cu124, numpy 1.26.4, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.9.1,
scanpy 1.10.2, anndata 0.12.19.
