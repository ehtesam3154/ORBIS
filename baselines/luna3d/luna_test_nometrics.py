"""Run LUNA's own test pipeline (main.py general.mode=test_only) without its per-graph metric step.

perform_evaluation is replaced in memory by a stub; predictions are written by prepare_data before that step, so they are
unchanged. LUNA's source dir comes from LUNA_SRC (also on PYTHONPATH). usage: python luna_test_nometrics.py <Hydra overrides>
"""
import os, runpy, sys
LUNA = os.environ["LUNA_SRC"]
os.chdir(LUNA)
import utils.diffusion_model.test.test as _t
def _no_metrics(epoch, metadata_true, metadata_pred_normalized, unique_cell_classes, dir_path):
    return {"test/epoch": epoch, "num_cells": len(metadata_true)}
_t.perform_evaluation = _no_metrics
print("[luna_test_nometrics] LUNA's perform_evaluation -> stub (predictions saved by prepare_data; metrics not computed)", flush=True)
sys.argv = ["main.py"] + sys.argv[1:]
runpy.run_path(os.path.join(LUNA, "main.py"), run_name="__main__")
