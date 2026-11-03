# Portions adapted from Penn Software REMAP (https://github.com/ShunzhouJiang/REMAP, commit b663e2b).
# Copyright (C) 2025 The Trustees of the University of Pennsylvania. All rights reserved.
# Used and modified under the REMAP COPYRIGHT AND PERMISSION NOTICE (non-profit research purposes only);
# the full notice, conditions and disclaimer are in LICENSE_REMAP.txt in this folder.
"""REMAP's released fit loop (Fit_cord / Predict_cord / Fit_covet / Predict_covet over n_iter rounds) with three memory-only
changes: unused return values, read-once input copies and superseded covariance arrays are released early.
"""
import gc
import sys
import numpy as np
import torch
import remap_lowmem as LM
from remap.util import use_corrected_expression
from remap.fit_functions import Fit_cord, Predict_cord, Fit_covet, Predict_covet, pred_transform
def fit_cord_single_lowmem2(Rdata, location_data, Qdata, path_name, hidden_dims=[400, 200, 100], n_iter=3, batch_size=256):
    ff = sys.modules["remap.fit_functions"]
    orig = (ff.wrap_gene_location, ff.wrap_gene_covet)
    ff.wrap_gene_location, ff.wrap_gene_covet = LM._WrapLocationNoCopy, LM._WrapCovetNoCopy
    try:
        return _body(Rdata, location_data, Qdata, path_name, hidden_dims, n_iter, batch_size)
    finally:
        ff.wrap_gene_location, ff.wrap_gene_covet = orig
def _body(Rdata, location_data, Qdata, path_name, hidden_dims, n_iter, batch_size):
    print(f"Model is on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    norm = True
    Rdata = use_corrected_expression(Rdata)
    Qdata = use_corrected_expression(Qdata)
    for A in (Rdata, Qdata):
        A.obsm.pop("corrected", None)
    covet_train_pca = Rdata.obsm.pop("covet")
    covet_test_pca = Qdata.obsm.pop("covet")
    gc.collect()
    if norm:
        Rdata, Qdata = LM.preprocess_inplace(Rdata, Qdata)
    location_data = np.array(location_data)
    LM.mem("preprocess (in place)")
    for i in range(n_iter):
        file_name_loc = f"location_iter{i}"
        Fit_cord(data_train=Rdata, location_data=location_data, covet_train=covet_train_pca, hidden_dims=hidden_dims,
                 num_epochs_max=500, number_error_try=15, batch_size=batch_size, path=path_name, filename=file_name_loc)
        gc.collect(); LM.mem(f"Fit_cord {i}")
        pred_loc_test = Predict_cord(data_test=Qdata, covet_test=covet_test_pca, out_dim=location_data.shape[1], path=path_name,
                                     filename=file_name_loc)
        pred_loc_train = Predict_cord(data_test=Rdata, covet_test=covet_train_pca, out_dim=location_data.shape[1], path=path_name,
                                      filename=file_name_loc)
        LM.mem(f"Predict_cord {i} (query + reference)")
        pred_loc_test_transform = pred_transform(pred_cord=pred_loc_test, train_cord=np.array(location_data))
        pred_loc_train_transform = pred_transform(pred_cord=pred_loc_train, train_cord=np.array(location_data))
        if i != n_iter - 1:
            file_name_covet = f"covariance_iter{i}"
            Fit_covet(data_train=Rdata, location_data=pred_loc_train_transform, covet_train=covet_train_pca, hidden_dims=hidden_dims,
                      num_epochs_max=500, number_error_try=15, batch_size=batch_size, path=path_name, filename=file_name_covet)
            gc.collect(); LM.mem(f"Fit_covet {i}")
            out_dim = covet_train_pca.shape[1]
            del covet_train_pca, covet_test_pca; gc.collect()
            pred_covet_test = Predict_covet(data_test=Qdata, location_test=pred_loc_test_transform, out_dim=out_dim,
                                            location_train=pred_loc_train_transform, path=path_name, filename=file_name_covet)
            pred_covet_train = Predict_covet(data_test=Rdata, location_test=pred_loc_train_transform, out_dim=out_dim,
                                             path=path_name, filename=file_name_covet)
            LM.mem(f"Predict_covet {i} (query + reference)")
            covet_test_pca = pred_covet_test
            covet_train_pca = pred_covet_train
            del pred_covet_test, pred_covet_train
    np.save(f"{path_name}/remap_loc.npy", pred_loc_test_transform)
    return pred_loc_test_transform
