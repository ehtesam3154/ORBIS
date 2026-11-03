# REMAP (3-D)

REMAP (Jiang et al.) predicts each query cell's coordinates from its expression and its predicted neighbourhood covariance. It
predicts any number of coordinates, so it needs no code change for 3-D; the official package is used unmodified.

- Official repository: https://github.com/ShunzhouJiang/REMAP, commit `b663e2ba21d93a8451e088336fd3fdc32633ae5b`.
  License: Penn Software REMAP permission notice, non-profit research use only (full text: `LICENSE_REMAP.txt`). Files here that
  adapt REMAP code carry its copyright notice.
- No patch: the configs put the clean checkout on `PYTHONPATH`.
- Configs: `baselines/configs/remap_<test>.sh`.

## Scripts

| file | role |
|---|---|
| `remap_3d.py` | brain folds: reference = cell stores of the training mice, query = every cell of the test mouse's store |
| `remap_3d_stores.py` | any store layout (Allen, Deep-STARmap blocks, embryos); gene subset, zero-count handling, external corrected input, fit path |
| `remap_lowmem.py` | memory-only, same-math versions of REMAP's covariance initialisation, niche-model training, covariance PCA, query decoding and preprocessing |
| `remap_lowmem_fit2.py` | REMAP's fit loop with three memory-only releases of arrays it no longer reads (`--fit lowmem2`) |
| `to_layout.py` | final coordinates (store row order, mm) -> our layout format; also used by CeLEry-3D, Tangram-3D and novoSpaRc-3D |

## Settings

Official everywhere: k = 100 neighbours, 100 covariance genes, niche model 10,000 steps, PCA pve 0.98, `n_iter` 3, hidden layers
400 / 200 / 100, batch 256, REMAP's own learning rate and stopping rule, `log_input` 0.1 (the function default). Every training cell
is in the reference, except mouse 2 (see below).

- **Batch correction.** Same lab and platform (Zhuang mice, the Deep-STARmap blocks, the embryos): `harmony=False`, REMAP's own option,
  so REMAP gets the same raw input as the other methods. Allen (another lab): the saved Harmony-corrected input of the other methods, fed
  as the authors' tutorials feed external correction (`harmony=False` + `obsm['corrected']`). The tutorials' corrected input is
  gene-level count-scale expression, so the 477 measured genes are reconstructed from the 200 corrected PCs
  (z = PC x components + mean; x = z x sd + mean; expression = max(expm1(x), 0)) and REMAP normalises them as usual.
- **Mouse 2** reuses the earlier run with a seeded uniform 400,000-cell reference (proportional per animal) and Harmony on (the
  authors' advice for cross-subject use).
- **Zero-count cells.** REMAP's library normalisation drops cells without counts, and the released code would then misalign the
  reference coordinates; such training cells are removed first (`--drop_zero_ref 1`: 83 Allen-fold, 34 embryo cells). One zero-count
  embryo-c test cell gets the mean predicted position (`--query_zero fill_mean`).

## Memory-only adaptations (same math)

The released code holds the N x 100 x 100 covariance matrices and their square roots in RAM (and on the GPU for training) and runs a
float32 full SVD of N x 5,050 values: about 650 GB of RAM and 175 GB of GPU memory at 4 million reference cells. `remap_lowmem.py`
changes only where arrays live:

| step | change | equivalence (identical inputs) |
|---|---|---|
| covariance + square roots | computed in 32,768-cell chunks into float32 row files; the global mean shift reproduced bit for bit | max abs diff 0 |
| niche-model training | each step's rows read from the file; the official random index vectors drawn up front in the same order | weights after training: max abs diff 0 |
| covariance PCA | the same PCA from a float64 covariance accumulated over chunks; sklearn's component-count and sign rules | same component count and signs; differs from the official float32 SVD only by that SVD's own float32 error |
| query decoding | the official 16 decode chunks, each projected before the next | max abs diff 0 |
| counts, preprocessing | no duplicate copies; library normalisation, log1p and z-score in place, in row chunks | max abs diff 0 |
| fit (`remap_lowmem_fit2.py`) | arrays the loop no longer reads are released | final coordinates max abs diff 0 |

End to end (20,000 reference / 20,000 query cells) the official and memory-only runs agree to within the official code's own
run-to-run float noise (the thread count alone moves official predictions by a median 0.18 mm).

## Run

```bash
DATA_DIR=... OUT_DIR=... PY_REMAP=/path/to/python GPU=0 bash baselines/configs/remap_mouse3.sh
```

Writes `OUT_DIR/remap/<run>/final_coords.npz` and `OUT_DIR/layouts/remap_<test>.npz`. Threads: 8 (OpenMP / MKL / OpenBLAS / Numba);
the thread count changes only float rounding. Environment of the reported runs (Python 3.11): torch 2.11.0+cu128, numpy 1.26.4,
scipy 1.13.1, pandas 2.2.2, scikit-learn 1.5.1, scanpy 1.10.2, anndata 0.10.8, harmonypy 0.0.10, numba 0.68.0;
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` lets REMAP's own `torch.load` read its pickled networks.
