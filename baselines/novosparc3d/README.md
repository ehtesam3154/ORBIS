# novoSpaRc-3D

novoSpaRc (Nitzan, Karaiskos et al.) reconstructs positions by optimal transport between cells and target locations. Here it runs
in its atlas-assisted mode with a seeded subsample of training cells (3-D positions and expression) as the target. The official
package is used unmodified.

- Official repository: https://github.com/rajewsky-lab/novosparc, commit `a3c9f3d76e59c7a601cfcd0e211630e2458d79a3` (BSD 3-Clause).
- No patch: the configs put the clean checkout on `PYTHONPATH`.
- Driver: `novosparc_3d.py`. Configs: `baselines/configs/novosparc3d_<test>.sh`. The layout is written by `baselines/remap/to_layout.py`.

## Settings

- The authors' Drosophila tutorial (atlas-assisted): `setup_reconstruction(atlas, markers = all shared genes, num_neighbors_s = 5,
  num_neighbors_t = 5)`, `reconstruct(alpha_linear=0.8, epsilon=5e-3)`.
- Position = the transport-plan-weighted mean of the target positions (the tutorial gives no per-cell position); the most probable
  target and its probability are saved too. Expression: `normalize_total` 1e4 + log1p. Allen: the saved Harmony input as reconstructed
  genes (as for REMAP).
- Target = seeded uniform subsample of the training cells; query in seeded chunks. Sizes set by the compute budget (the dense
  target-target matrices of all training cells do not fit): 10,000 targets and 10,000-cell chunks for the blocks and embryos,
  5,000 / 5,000 for the brains and Allen. `--part i/n` runs every n-th chunk; the chunks do not depend on n; `--merge n` joins the parts.

## Run

```bash
DATA_DIR=... OUT_DIR=... PY_NOVOSPARC=/path/to/python THREADS=4 bash baselines/configs/novosparc3d_embryo_d.sh
```

Environment of the reported runs (Python 3.11): numpy 1.26.4, scipy 1.17.1, pandas 2.3.3, POT 0.9.5, scanpy 1.10.2, anndata 0.12.19.
