# LUNA-3D

LUNA (Yu et al.) adapted from 2-D tissue sections to 3-D specimens. Everything not listed below runs as released.

- Official repository: https://github.com/mlbio-epfl/LUNA, commit `64f10eb55a4cdc862c0535051a1fe042ab2ceadc` (MIT License).
- Our change to LUNA's code: `luna_3d.patch` (apply to a clean checkout of that commit with `git apply luna_3d.patch`).
- Our scripts (this folder): input builders, train / test wrappers, protocol R collector.
- Configs: `baselines/configs/luna3d_<test>.sh`, one per test (`mouse1 mouse2 mouse3 allen blockA blockB embryo_d embryo_c`).

## What `luna_3d.patch` changes, and why

1. **2-D to 3-D coordinates (required by the experiment).** Every place where LUNA fixes the coordinate dimension at 2 is set to 3:
   - `datasets/data_module.py`: reads and requires `coord_Z`;
   - `utils/data/load.py`: `position_normalize` also normalises `coord_Z`; `to_dataframe` writes it;
   - `models/self_attention.py`: position input `Linear(2, 64)` -> `Linear(3, 64)`; position head output 2 -> 3;
   - `models/model.py`: input size of `mlp_out_pos_norm` (features + positions + norm) `+3` -> `+4`;
   - `utils/diffusion_model/diffusion/noise_model.py`: initial test-time noise `(..., 2)` -> `(..., 3)`;
   - `metrics/evaluation_statistics.py` (LUNA's own logged metrics): uses `coord_Z` instead of zero padding.
   The norm-based position MLP and the training noise (`randn(positions.shape)`) are already dimension-free.
2. **No figures.** LUNA's test writes one scatter-plot PDF per test graph; switched off (`metrics/evaluation_plot.py`).
3. **Memory-mapped training table (required by memory).** LUNA's CSV loader keeps ~30 KB per cell and one copy per DDP rank, so
   3-6 million training cells do not fit. With `+dataset.train_memmap_dir=DIR` the training table comes from binary files written by
   `make_luna_bin.py` / `make_luna_flysta.py` (rows already in the order LUNA's `Dataset.process_data` sorts them, genes in
   `filter_genes` order); features are read through a memory map. `+train.num_workers` sets the DataLoader workers (default 32 =
   LUNA's hard-coded value). Without these keys LUNA runs its original code path.

The same model, loss, graphs and values are used either way: on a 4-graph subset the binary path gave node features, positions,
classes, ids, slices and statistics bit-identical to the original CSV path.

## Data side

- Expression: the stores' `log2(1 + count)` (LUNA's README asks for log2-normalised counts); Allen: the 200 saved Harmony-corrected PCs,
  the same input as our model. LUNA applies `nn.Linear` to its node features first, so real-valued PCs are valid input.
- Graphs: in 2-D a LUNA graph is one section. A 3-D specimen (0.03-4.6 million cells) is cut into random ~5,000-cell subsets (seed 0;
  5,000 = the median section size of LUNA's own training set), so depth varies inside every graph and gets a gradient under LUNA's
  pairwise-distance loss. Every training cell is used once; test graphs are formed without positions; every test cell is predicted once.
- Training uses every cell of the training specimens (LUNA's released config does not downsample).

## Recipe

As released (`configs/experiment/MERFISH_mouse_cortex.yaml`): LR 5e-4 constant (AdamW, amsgrad), seed 0, 1,000-step sampler. Changes:
300 training epochs (400 on the Deep-STARmap blocks) instead of 1,000, checkpoint every 100 epochs, the last checkpoint is tested;
global batch 8 (the per-GPU batch is 8 / number of GPUs, so the recipe does not depend on the GPU count).

## Scripts

| file | role |
|---|---|
| `make_luna_csv.py` | brain folds: test CSV parts (`test_partNN.csv`, 50 graphs each), `train_infos.csv`, a training CSV (its seeded draw fixes the test partition) |
| `make_luna_bin.py` | brain folds: binary training table with every training cell |
| `make_luna_flysta.py` | one-ruler specimens (embryos, tissue blocks): binary training table, test CSVs, `test_tiny.csv`, `train_infos.csv` |
| `make_allen_store_views.py` | Allen fold: store views with the Harmony PCs as features |
| `run_luna3d_train_ddp.sh` | training with LUNA's `main.py` |
| `run_luna3d_test.sh`, `luna_test_nometrics.py` | LUNA's `test_only` sampler on every test part; LUNA's per-graph metric step (which needs SciPy 1.9.1) is replaced by a stub, predictions are written before that step and are unchanged |
| `make_row_map.py` | mouse 1: truth row -> store row map (exact position match) |
| `luna_unstretched.py` | protocol R: per-graph predictions -> one layout in our format |

## Protocol R (how LUNA's output becomes a layout)

LUNA has no physical units: it min-max normalises every graph per axis and predicts normalised, centred coordinates in an arbitrary
orientation (its loss compares pairwise distances only). Protocol R uses only information our model also gets: (1) the unit is the
mean per-axis extent of LUNA's training graphs; (2) all test graphs are pooled as LUNA outputs them; (3) ONE rotation / reflection,
no scale, for the whole specimen, fitted against the truth in the training ruler (Allen: the truth folded at the midline, because the
training mice are hemibrains). Output: `P` (layout), `G` (truth), `cc` (cell classes), `iso` - the format of our samples.

## Run

```bash
DATA_DIR=... OUT_DIR=... PY_LUNA=/path/to/python bash baselines/configs/luna3d_mouse3.sh            # all stages
DATA_DIR=... OUT_DIR=... STAGES="test collect" GPU=0 bash baselines/configs/luna3d_mouse3.sh       # some stages
```

The config fetches the official commit, applies the patch, builds the inputs, trains, samples every test part and writes
`OUT_DIR/layouts/luna3d_<test>.npz`. Environment of the reported runs (Python 3.11): torch 2.6.0+cu124, pytorch-lightning 2.6.4,
torch-geometric 2.7.0, linear-attention-transformer 0.19.1, hydra-core 1.3.2, omegaconf 2.3.0, numpy 2.4.5, scipy 1.17.1,
pandas 2.3.3, scikit-learn 1.8.0.

## Dissociated-cell (10x) application

Config: `baselines/configs/luna3d_allen10x.sh` (stages `prep train sample`).

- Training (`make_luna_bin_a10x.py`, `make_a10x_tiny_csvs.py`, `run_luna3d_train_ddp.sh`): every cell of the five spatial brains in
  `$DATA_DIR/allen10x/stores` (Zhuang mice 1-4 and the Allen hemisphere, 200 reference-space features), random ~5,000-cell graphs, seed 0,
  300 epochs, global batch 8, LR 5e-4; the last checkpoint `epoch=299.ckpt` is used.
- Sampling (`luna_sample_features.py`, `luna_sample_common.py`): dissociated cells have no positions, so LUNA's own sampler runs on
  graphs built from the features alone (positions are inert zeros used only for their shape). Every application cohort (primary / strat
  1-5, seeds 1000-1002) is cut into 20 random graphs of 5,000 cells (partition seed 25000 + cohort); 1,000 sampler steps.
- Output: `$APP/raw/baseline_<kind><c>_s<seed>.npz` (`P_norm`, `P_raw`, `graph_id`, `cell_ID`) and a sidecar `.json` with `train_unit_EXT`
  (the mean per-axis extent of the training graphs), read by `scripts/allen10x/application.py`, which applies the label-free pose readout.
