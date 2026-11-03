# CeLEry-3D

CeLEry (Zhang et al.) trains a neural network that regresses a cell's coordinates from its expression. Here it predicts three
coordinates instead of two; everything else runs as released.

- Official repository: https://github.com/QihuangZhang/CeLEry, commit `b0b73764bb517b0b7b360d4b18bed5d6b77a4615` (MIT License).
- Our change to CeLEry's code: `celery_3d.patch` (apply to a clean checkout with `git apply celery_3d.patch`):
  - `CeLEry/DNN.py`: last layer 3 outputs instead of 2;
  - `CeLEry/datasetgenemap.py` (`wrap_gene_location`): z is normalised exactly as x and y (min - 1, max + 1 per axis);
  - `CeLEry/fit_functions.py` (`Predict_cord`, `report_prop_method_sc`): 3 output columns.
- Driver: `celery_3d.py`. Configs: `baselines/configs/celery3d_<test>.sh`. The layout is written by `baselines/remap/to_layout.py`.

## Settings

- `Fit_cord(hidden_dims=[30, 25, 15])` with its defaults: batch 4, lr 1e-3, Adam weight decay 1e-5, LR halving after 15 worse epochs;
  CeLEry's z-score on reference and query separately; `Predict_cord` per cell. CPU (CeLEry's trainer), 1 BLAS thread.
- Reference = every training cell. Expression: counts -> library size 1e4 -> log1p (the tutorial input is log-normalised) ->
  CeLEry's z-score; Allen: the 200 saved Harmony-corrected PCs (the input of the other methods). Coordinates in micrometres, so
  CeLEry's +-1 padding before min-max stays negligible, as with its own pixel-unit data.
- Training budget: the default 500 epochs at batch 4 take about 25 days per brain. The budget is the authors' tutorial maximum in
  optimizer updates, 500 epochs x 1,456 steps = 728,000, rounded up to whole epochs: epochs = max(1, ceil(728,000 / steps per epoch)).
  Reported runs: blocks A / B 15 / 16 epochs, embryos 11, brains and Allen 1 epoch (0.74-1.46 million updates).

## Run

```bash
DATA_DIR=... OUT_DIR=... PY_CELERY=/path/to/python bash baselines/configs/celery3d_mouse3.sh
```

Environment of the reported runs (Python 3.11): torch 2.5.1+cpu, numpy 1.26.4, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.9.1,
scanpy 1.10.2, anndata 0.12.19, scikit-image 0.26.0.
