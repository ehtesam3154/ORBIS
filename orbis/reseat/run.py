"""Reseating CLI: reassign the cells of a sampled layout to its sites with the niche engine (one GPU), keeping every site.

Fixed settings: scales 32 / 128 nearest sites, T = 1, 30-nearest-site proposals, 128 sweeps, swap seed 20261019, cap = 1 x tau_G
(tau_G = the engine's anchor scale, the RMS per-axis gap between the sample and its fitted best guess), block side = 2 x the
quantile of the per-proposal reach. Writes the reseated layout (the sample's npz keys, P permuted) and a run json with the run facts.

usage: python -m orbis.reseat.run --sample S.npz --guess G.npz --reader R.npz --test_dir PREBUILT --mm_per_unit F --out OUT.npz"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--sample", required=True, help="finished sample layout npz (P, G, cc, iso)")
ap.add_argument("--guess", required=True, help="one-call best guess npz of the same checkpoint and sampler seed")
ap.add_argument("--reader", required=True, help="reader npz (p, q32, q128, sigma32, sigma128; rows = the test set's)")
ap.add_argument("--reader_md5", default="", help="expected md5 of the reader file; empty = only recorded")
ap.add_argument("--test_dir", required=True, help="prebuilt test directory (pos.npy, cell_class.npy) for the identity checks")
ap.add_argument("--mm_per_unit", type=float, required=True, help="millimetres per model unit of the test set")
ap.add_argument("--out", required=True, help="output npz; the run json is written beside it (same name, .json)")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--block_um", type=float, default=0.0, help="block side in um (0 = the rule: 2 x quantile of the reach)")
ap.add_argument("--quantile", type=float, default=0.9, help="quantile of the block-side rule")
ap.add_argument("--nvcc", default="", help="nvcc executable (default: $CUDA_HOME/bin/nvcc, else nvcc on PATH)")
ap.add_argument("--ccbin", default="g++-11", help="host compiler passed to nvcc")
ap.add_argument("--cuda_build_dir", default="", help="where the compiled kernels are cached (default: <out dir>/cuda_build)")
A = ap.parse_args()
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(A.threads)
os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
import json
import time

import numpy as np

from orbis.reseat import engine_cpu as NA
from orbis.reseat import engine_gpu as NG
from orbis.reseat import inputs as IN

SWAP_SEED = 20261019
SWEEPS = 128
CAP_TAU = 1.0
SCALES = (32, 128)
T0 = time.time()


def P(*a):
    """Print with the elapsed time."""
    print(f"[{time.time() - T0:7.0f} s]", *a, flush=True)


def mlock_arrays(eng):
    """Pin the CPU engine arrays that the final from-scratch energy re-reads, so they are not swapped out during the GPU sweeps."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    arrs = [eng.p, eng.S, eng.G] + [a for sc in eng.scales for a in (sc["q"], sc["nbr"])]
    tot, fail = 0, 0
    for a in arrs:
        if libc.mlock(ctypes.c_void_p(a.ctypes.data), ctypes.c_size_t(a.nbytes)) == 0:
            tot += a.nbytes
        else:
            fail += 1
    P(f"  mlock: {tot / 1e9:.2f} GB of CPU engine arrays pinned ({fail} arrays not locked)")


def torch_peak():
    """Peak GPU memory allocated by this process (GB)."""
    import torch
    return torch.cuda.max_memory_allocated() / 1e9


def run():
    """Build the inputs and the engines, run the sweeps, check the final energy from scratch, write the layout and the run json."""
    out_npz = os.path.abspath(A.out)
    out_json = out_npz[:-4] + ".json" if out_npz.endswith(".npz") else out_npz + ".json"
    assert not os.path.exists(out_npz), f"{out_npz} exists"
    assert not os.path.exists(out_json), f"{out_json} exists"
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    nvcc = A.nvcc or (os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc") if os.environ.get("CUDA_HOME") else "nvcc")
    build_dir = A.cuda_build_dir or os.path.join(os.path.dirname(out_npz), "cuda_build")
    t_all = time.time()
    I = IN.build(A.sample, A.guess, A.reader, A.test_dir, A.mm_per_unit, A.reader_md5 or None)
    P(f"reseat: {len(I['S']):,} sites ({A.sample}, md5 {I['sample_md5']}); best guess {A.guess} md5 {I['guess_md5']}; reader {A.reader} md5 "
      f"{I['reader_md5']}; {A.mm_per_unit!r} mm per unit; swap seed {SWAP_SEED}; all sites active; {SWEEPS} sweeps")
    P(f"  best guess after the rigid fit onto the sample: median |G - S| {np.median(np.linalg.norm(I['G'] - I['S'], axis=1)):.1f} um")
    active = np.ones(len(I["S"]), bool)
    t0 = time.time()
    d0 = ((np.ascontiguousarray(I["S"], dtype=np.float64) - np.ascontiguousarray(I["G"], dtype=np.float64)) ** 2).sum(1)
    tau_pre = float(np.sqrt(float(d0.mean() / 3.0)))
    cap_req = CAP_TAU * tau_pre
    P(f"  cap rule: tau_G {tau_pre:.4f} um -> requested cap = 1 x tau_G = {cap_req:.4f} um")
    eng = NA.Engine(I["S"], I["p"], I["G"], SCALES, I["isig"], I["q"], active, T=1.0, cap_um=cap_req, use_niche=True, workers=A.threads, log=P,
                    tau_um=None)
    assert abs(float(np.sqrt(eng.tau2)) - tau_pre) <= 1e-9 * tau_pre, (np.sqrt(eng.tau2), tau_pre)
    assert eng.cap2 == float(cap_req) ** 2, (eng.cap2, cap_req)
    t_cpu_setup = time.time() - t0
    t0 = time.time()
    ge = NG.GPUEngine(eng, nvcc, A.ccbin, build_dir, block_um=(A.block_um or None), quantile=A.quantile, log=P, check="first")
    t_gpu_setup = time.time() - t0
    gpu_cap2 = float(ge.state().cap2)
    P(f"  cap check: requested {cap_req:.4f} um | engine cap2 {eng.cap2:.6f} (= {np.sqrt(eng.cap2):.4f} um) | GPU state cap2 {gpu_cap2:.6f} "
      f"(= {np.sqrt(gpu_cap2):.4f} um)")
    assert gpu_cap2 == eng.cap2, (gpu_cap2, eng.cap2)
    mlock_arrays(eng)
    traj = ge.run(SWEEPS, seed=SWAP_SEED)
    P(f"  start energy {traj[0]['total']:.6f} (niche {traj[0]['niche']})")
    t0 = time.time()
    Ef = NA.total_energy(eng.cell_at, eng.S, eng.p, eng.G, eng.inv2tau2, [(s["q"], s["isig"], s["nbr"]) for s in eng.scales], eng.delta, eng.mu0,
                         eng.s0, *eng.use)
    final_same = Ef["total"] == traj[-1]["total"]
    P(f"  final energy: tracked {traj[-1]['total']:.6f}, from scratch (CPU code) {Ef['total']:.6f} -> "
      f"{'BIT-IDENTICAL' if final_same else 'DIFFERENT'} ({time.time() - t0:.0f} s)")
    z0 = I["z0"]
    P_out = ge.output_layout(z0["P"])
    assert P_out.dtype == z0["P"].dtype and NA.same_multiset(P_out, z0["P"]), "output positions are not the sample's positions"
    tmp = out_npz[:-4] + ".tmp.npz"
    np.savez(tmp, P=P_out, **{k: z0[k] for k in z0.files if k != "P"})
    os.replace(tmp, out_npz)
    sec = np.array([t["seconds"] for t in traj[1:]])
    here = os.path.dirname(os.path.abspath(__file__))
    res = dict(sample=os.path.abspath(A.sample), sample_md5=I["sample_md5"], best_guess=os.path.abspath(A.guess), best_guess_md5=I["guess_md5"],
               reader=dict(path=os.path.abspath(A.reader), md5=I["reader_md5"]), test_dir=os.path.abspath(A.test_dir), mm_per_unit=A.mm_per_unit,
               swap_seed=SWAP_SEED, output=dict(path=out_npz, md5=IN.md5(out_npz)), engine=dict(info=ge.info, trajectory=traj, sweeps=SWEEPS),
               effective=dict(sites=int(len(I["S"])), active=int(active.sum()), sweeps_executed=int(len(traj) - 1), sweeps_requested=SWEEPS,
                              block_um=ge.info["geometry"]["block_um"], block_rule=ge.info["geometry"]["rule"], quantile_arg=A.quantile,
                              accepted_total=int(sum(t["accepted"] for t in traj[1:])), cap_rejected_total=int(sum(t["cap_rejected"] for t in traj[1:]))
                              if "cap_rejected" in traj[1] else None),
               cap_check=dict(rule="cap_um = 1 x tau_G (tau_G = the engine's own anchor scale)", tau_G_um=tau_pre, cap_requested_um=cap_req,
                              engine_cap2=eng.cap2, engine_cap_um=float(np.sqrt(eng.cap2)), gpu_state_cap2=gpu_cap2, gpu_state_cap_um=float(np.sqrt(gpu_cap2)),
                              cap_rejected_per_sweep=[t.get("cap_rejected") for t in traj[1:]], proposals_per_sweep=[t.get("proposals") for t in traj[1:]]),
               start_energy=traj[0]["total"], final_energy_from_scratch=Ef, final_energy_bit_identical=bool(final_same),
               wall=dict(cpu_engine_setup_s=round(t_cpu_setup, 1), gpu_setup_s=round(t_gpu_setup, 1), sweep_median_s=float(np.median(sec)),
                         sweep_mean_s=float(sec.mean()), sweeps_total_s=float(sec.sum()), total_s=round(time.time() - t_all, 1),
                         gpu_peak_mem_gb=round(torch_peak(), 2)),
               code_md5=dict(gpu_engine=IN.md5(f"{here}/engine_gpu.py"), cpu_engine=IN.md5(f"{here}/engine_cpu.py"), driver=IN.md5(os.path.abspath(__file__)),
                             inputs=IN.md5(f"{here}/inputs.py"), cuda_so=IN.md5(ge.lib.so_path)))
    json.dump(res, open(out_json, "w"), indent=1, default=float)
    P(f"RESEAT DONE -> {out_npz} md5 {res['output']['md5']}; tau_G {tau_pre:.4f} um; cap {cap_req:.4f} um (engine {np.sqrt(eng.cap2):.4f}, GPU "
      f"{np.sqrt(gpu_cap2):.4f}); sweeps {res['effective']['sweeps_executed']}; accepted {res['effective']['accepted_total']:,}; cap-blocked "
      f"{res['effective']['cap_rejected_total']:,}; final energy {'BIT-IDENTICAL' if final_same else 'DIFFERENT'}; wall {res['wall']['total_s']} s; "
      f"GPU peak {res['wall']['gpu_peak_mem_gb']} GB; L {res['effective']['block_um']:.1f} um")


if __name__ == "__main__":
    P(f"orbis.reseat.run --sample {A.sample} --guess {A.guess} --reader {A.reader} | threads {A.threads} | quantile {A.quantile} | block_um {A.block_um}")
    run()
