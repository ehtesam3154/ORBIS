"""Expression-only sampling with a trained LUNA-3D model (dissociated cells have no positions).

LUNA's own sampler (sample_graphs -> sample_noise, iterate_sampling) runs on a DataHolder built per graph from the per-cell features
and opaque ids; positions are inert zeros (shape only), cell_class 0, mask all True. Graphs = a seeded random partition of the rows
into training-size graphs; feature columns are permuted by name into the training table's order. torch.manual_seed calls are shifted
by --seed_offset and the initial draw is pinned per graph with --init_seed; K = 1,000 steps. Output npz: P_norm (LUNA's exported,
per-graph min-max normalised prediction), P_raw, graph_id, cell_ID, graph_order; sidecar <out>.json with train_unit_EXT (the mean
per-axis extent of the training graphs) and the run record. LUNA's source dir must be on PYTHONPATH.
"""
import os, sys
import luna_sample_common as C
C.install_audit_hook()
import argparse, json, time, types
import numpy as np
import torch
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--run_dir", required=True, help="the training run directory holding .hydra/config.yaml (model hyper-parameters)")
ap.add_argument("--train_dir", required=True, help="the training table directory (meta.json, pos.npy, section_id.npy)")
ap.add_argument("--features", required=True, help="N x G float32 .npy, columns named <feature_prefix>1 ... <feature_prefix>G")
ap.add_argument("--ids", required=True, help="N opaque integer cell IDs (.npy)")
ap.add_argument("--feature_prefix", default="", help="the provided columns are named <prefix>1 ... <prefix>G (old basis: hPC)")
ap.add_argument("--feature_names_json", default="", help="alternative to --feature_prefix: a JSON list with the G column names of --features")
ap.add_argument("--partition_seed", type=int, required=True)
ap.add_argument("--seed_offset", type=int, required=True)
ap.add_argument("--init_seed", type=int, required=True)
ap.add_argument("--graph_size", type=int, default=0, help="0 = the training table's graph size (meta.json 'graph')")
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--out", required=True)
ap.add_argument("--gpu", default="0")
ap.add_argument("--placeholder_fuzz", type=int, default=-1)
a = ap.parse_args()
t_start = time.time()
from omegaconf import OmegaConf
import utils.diffusion_model.sample.sample as S
from diffusion_model import FullDenoisingDiffusion
from utils.data.dataholder import DataHolder
from utils.data.load import to_dataframe, position_normalize
dev = torch.device(f"cuda:{a.gpu}")
F = np.load(a.features)
ids = np.load(a.ids)
assert F.ndim == 2 and F.dtype == np.float32, f"features must be N x G float32, got {F.shape} {F.dtype}"
N, G = F.shape
assert ids.shape == (N,) and np.issubdtype(ids.dtype, np.integer) and len(np.unique(ids)) == N, "ids: N unique integers"
assert np.isfinite(F).all(), "non-finite feature values"
meta = json.load(open(f"{a.train_dir}/meta.json"))
train_genes = list(meta["genes"])
assert len(train_genes) == G, f"the training table has {len(train_genes)} feature columns, got {G}"
assert bool(a.feature_prefix) != bool(a.feature_names_json), "give exactly one of --feature_prefix / --feature_names_json"
names = json.load(open(a.feature_names_json)) if a.feature_names_json else [f"{a.feature_prefix}{i}" for i in range(1, G + 1)]
assert len(names) == G and len(set(names)) == G, "feature names: G unique names"
assert sorted(names) == sorted(train_genes), "the provided column names are not the training table's"
col = np.array([names.index(g) for g in train_genes])
F_lu = np.ascontiguousarray(F[:, col])
size = a.graph_size or int(meta["graph"])
pos_tr = np.load(f"{a.train_dir}/pos.npy"); gid = np.load(f"{a.train_dir}/section_id.npy")
import pandas as pd
gid = pd.factorize(np.asarray(gid))[0]; o = np.argsort(gid, kind="stable"); cut = np.flatnonzero(np.diff(gid[o])) + 1
ext = np.array([p.max(0) - p.min(0) for p in np.split(np.asarray(pos_tr, dtype=np.float64)[o], cut)])
EXT = ext.mean(0); n_train_graphs = int(len(ext)); n_train_cells = int(len(pos_tr)); del pos_tr, gid, o
cfg = OmegaConf.load(f"{a.run_dir}/.hydra/config.yaml")
assert int(cfg.dataset.gene_columns_end) - int(cfg.dataset.gene_columns_start) == G, "config feature-column count"
infos = types.SimpleNamespace(input_dims={"node_features_dimensions": G, "diffusion_time_dimensions": 1},
                              output_dims={"node_features_dimensions": G, "diffusion_time_dimensions": 0})
model = FullDenoisingDiffusion.load_from_checkpoint(a.ckpt, dataset_infos=infos, cfg=cfg, map_location="cpu")
model.eval(); model.to(dev)
n_train_mode = sum(1 for m in model.modules() if m.training)
assert n_train_mode == 0, f"{n_train_mode} modules still in training mode (dropout would be active)"
n_dropout = sum(1 for m in model.modules() if isinstance(m, torch.nn.Dropout))
calls = [0]
model.model.register_forward_pre_hook(lambda m, i: calls.__setitem__(0, calls[0] + 1))
K_released = int(cfg.model.diffusion_steps)
assert K_released == model.max_diffusion_steps == 1000
_true_seed = torch.manual_seed
if a.steps != K_released:
    assert K_released % a.steps == 0, "--steps must divide 1000 (LUNA's sampler takes any s < t, noise_model.py:362-432)"
    INTERVAL = K_released // a.steps
    def _iterate_sampling(self, z_t, batch):
        for s_int in reversed(range(0, self.max_diffusion_steps, INTERVAL)):
            s_array = torch.full((1, 1), s_int, dtype=torch.long, device=batch.node_features.device)
            z_t = S.sample_zs_from_zt(self, z_t, s_array)
        return z_t
    S.iterate_sampling = _iterate_sampling
_orig_sample_noise = S.sample_noise
def _sample_noise(self, batch):
    _true_seed(int(a.init_seed))
    return _orig_sample_noise(self, batch)
S.sample_noise = _sample_noise
torch.manual_seed = lambda s: _true_seed(int(s) + a.seed_offset)
graphs = C.partition(N, size, a.partition_seed)
assert sum(len(g) for g in graphs) == N and len(np.unique(np.concatenate(graphs))) == N
order = list(range(len(graphs)))
P_raw = np.full((N, 3), np.nan, dtype=np.float32); P_norm = np.full((N, 3), np.nan, dtype=np.float32); graph_id = np.full(N, -1, dtype=np.int32)
per_graph = []
gen = torch.Generator().manual_seed(a.placeholder_fuzz if a.placeholder_fuzz >= 0 else 0)
print(f"[luna_sample] ckpt {a.ckpt} | N {N:,} cells x {G} features | {len(graphs)} graphs of {sorted(set(len(g) for g in graphs))} cells (size {size}, partition seed {a.partition_seed}) | "
      f"seed_offset {a.seed_offset} init_seed {a.init_seed} | steps {a.steps} | gene order: first training columns {train_genes[:4]} | model.training modules {n_train_mode}, Dropout modules {n_dropout} | "
      f"placeholders {'RANDOM seed ' + str(a.placeholder_fuzz) if a.placeholder_fuzz >= 0 else 'constants (positions 0, class 0)'} | training unit EXT {np.round(EXT, 4).tolist()} over {n_train_graphs} graphs / {n_train_cells:,} cells", flush=True)
try:
    with torch.inference_mode():
        for gi in order:
            rows = graphs[gi]; n = len(rows); tg = time.time(); c0 = calls[0]
            X = torch.from_numpy(F_lu[rows]).unsqueeze(0).to(dev)
            mask = torch.ones(1, n, dtype=torch.bool, device=dev)
            if a.placeholder_fuzz >= 0:
                pos_ph = (torch.randn(1, n, 3, generator=gen) * 100.0).to(dev)
                cls_ph = torch.randint(0, 40, (1, n, 1), generator=gen).to(dev)
            else:
                pos_ph = torch.zeros(1, n, 3, device=dev)
                cls_ph = torch.zeros(1, n, 1, dtype=torch.long, device=dev)
            cid = torch.from_numpy(ids[rows].astype(np.int64)).view(1, n, 1).to(dev)
            batch = DataHolder(node_features=X, positions=pos_ph, node_mask=mask, cell_ID=cid, cell_class=cls_ph, diffusion_time=None).mask()
            samples = S.sample_graphs(model, batch=batch, test=True)
            pred = samples[0].cpu().numpy()
            assert pred.shape == (n, 3) and pred.dtype == np.float32
            nan_raw = int(np.isnan(pred).sum())
            df = to_dataframe([0] * n, pred, index=ids[rows])
            dfn = position_normalize(df.copy())
            nan_norm = int(dfn[["coord_X", "coord_Y", "coord_Z"]].isna().sum().sum())
            assert nan_raw == 0 and nan_norm == 0, f"graph {gi}: NaN in the prediction ({nan_raw} raw, {nan_norm} normalised)"
            dfn.fillna(0, inplace=True)
            P_raw[rows] = pred; P_norm[rows] = dfn[["coord_X", "coord_Y", "coord_Z"]].to_numpy(); graph_id[rows] = gi
            per_graph.append(dict(graph=gi, cells=n, network_calls=calls[0] - c0, seconds=round(time.time() - tg, 1),
                                  raw_min=pred.min(0).round(4).tolist(), raw_max=pred.max(0).round(4).tolist()))
            print(f"[luna_sample] graph {gi}: {n} cells, {calls[0] - c0} network evaluations, {time.time() - tg:.1f} s", flush=True)
finally:
    torch.manual_seed = _true_seed
torch.cuda.synchronize()
assert (graph_id >= 0).all() and np.isfinite(P_raw).all() and np.isfinite(P_norm).all(), "row coverage / finiteness"
np.savez_compressed(a.out, P_norm=P_norm, P_raw=P_raw, graph_id=graph_id, cell_ID=ids.astype(np.int64), graph_order=np.asarray(order, dtype=np.int32))
side = dict(
    script=os.path.abspath(__file__), script_md5=C.md5_file(os.path.abspath(__file__)), common_md5=C.md5_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "luna_sample_common.py")),
    argv=sys.argv, ckpt=a.ckpt, ckpt_md5=C.md5_file(a.ckpt), run_dir=a.run_dir, config_md5=C.md5_file(f"{a.run_dir}/.hydra/config.yaml"),
    train_dir=a.train_dir, train_meta_md5=C.md5_file(f"{a.train_dir}/meta.json"), features=a.features, features_md5=C.md5_file(a.features), ids=a.ids, ids_md5=C.md5_file(a.ids),
    N_in=int(N), N_out=int(P_norm.shape[0]), G=int(G), graphs=len(graphs), graph_sizes=[len(g) for g in graphs], graph_size_requested=size, partition_seed=a.partition_seed,
    gene_order_first_training_columns=train_genes[:6], gene_column_permutation_first=col[:6].tolist(), seed_offset=a.seed_offset, init_seed=a.init_seed,
    seed_convention=f"LUNA chain unchanged (re-seeds manual_seed(0) after the initial draw; shifted by seed_offset={a.seed_offset}); initial draw pinned per graph with init_seed={a.init_seed} (fixed-step harness convention)",
    steps=a.steps, network_calls_per_graph=[g["network_calls"] for g in per_graph], model_training_modules=n_train_mode, dropout_modules=n_dropout,
    placeholders=("random seed %d" % a.placeholder_fuzz) if a.placeholder_fuzz >= 0 else "constants: positions 0, cell_class 0",
    train_unit_EXT=EXT.tolist(), train_graphs=n_train_graphs, train_cells=n_train_cells, per_graph=per_graph,
    P_norm_md5=C.md5_array(P_norm), P_raw_md5=C.md5_array(P_raw), dtype=str(P_norm.dtype), nan_rows=0,
    wall_seconds=round(time.time() - t_start, 1), peak_rss_gb=round(C.peak_rss_gb(), 2), peak_gpu_gb=round(torch.cuda.max_memory_allocated(dev) / 2**30, 2),
    device=torch.cuda.get_device_name(dev), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    files_opened=[list(x) for x in C.opened_files()])
C.write_json(a.out + ".json", side)
print(f"[luna_sample] DONE N_out {N:,} | graphs {len(graphs)} | evaluations per graph {side['network_calls_per_graph']} | wall {side['wall_seconds']} s | peak RSS {C.peak_rss_gb():.2f} GB, "
      f"peak GPU {side['peak_gpu_gb']} GB | P_norm md5 {side['P_norm_md5']} | wrote {a.out}", flush=True)
