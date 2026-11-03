"""Training command line: the flow model on fresh training views of cell stores, data-parallel over GPUs.

Launch with torchrun (one process per GPU), e.g.
  torchrun --nproc_per_node=1 -m orbis.train.train --prebuilt_dir DIR --views_stores S1,S2 --out OUT [options]
Checkpoints: OUT/orbis_epoch<E>.pt (state_dict = EMA weights, raw_state_dict, epoch, args) and OUT/latest.pt (adds the
optimiser state; --resume_ckpt continues from it).
"""
import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from orbis.data.prebuilt import PrebuiltGraphs, read_infos
from orbis.data.views import N_VIEW_TYPES, FreshViewDataset, to_gpu_batch, unit_box_positions
from orbis.model.dit import DiT3D
from orbis.model.load import build_model
from orbis.model.sdpa_attention import SDPAMultiheadAttention
from orbis.train.flow import LinearPath
from orbis.train.global_batch import loss_divisor, micro_plan
from orbis.train.losses import CoordMSELoss, PlaneBalancedDistLoss

WEIGHT_DECAY = 1e-12


def is_main():
    """True on rank 0 (or without a process group)."""
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log0(m):
    """Print on rank 0."""
    if is_main():
        print(f"[train] {m}", flush=True)


class TrainStep(torch.nn.Module):
    """Network + loss as one module whose output is the scalar loss (what DistributedDataParallel wraps)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, z, batched, loss_fn):
        return loss_fn(self.net(z), batched)


def attention_logit_max(net, step_mod, path, loss_fn, fb, seed, dev):
    """Largest |q.k| / sqrt(head_dim) of every attention module on one fixed view at t = 0.5 and 0.95 (noise from a
    private generator; no gradient; the global random streams are not used)."""
    amx, hooks = {}, []

    def hook(name):
        def h(mod, inp, out):
            B, N, _ = inp[0].shape
            q = mod.q_proj(inp[0]).view(B, N, mod.num_heads, mod.head_dim).transpose(1, 2)
            k = mod.k_proj(inp[1]).view(B, inp[1].shape[1], mod.num_heads, mod.head_dim).transpose(1, 2)
            if hasattr(mod, "q_norm"):
                q, k = mod.q_norm(q), mod.k_norm(k)
            q, k = q.float(), k.float()
            with torch.autocast("cuda", enabled=False):
                v = 0.0
                for c0 in range(0, q.shape[2], 2048):
                    v = max(v, float((q[:, :, c0:c0 + 2048] @ k.transpose(-1, -2)).abs().max()) / math.sqrt(q.shape[-1]))
            amx[name] = max(amx.get(name, 0.0), v)
        return h
    for name, mod in net.named_modules():
        if isinstance(mod, SDPAMultiheadAttention):
            hooks.append(mod.register_forward_hook(hook(name)))
    g = torch.Generator(device=dev).manual_seed(seed + 777)
    with torch.no_grad():
        for tf in (0.5, 0.95):
            nz = torch.randn(fb.positions.shape, generator=g, device=dev, dtype=fb.positions.dtype)
            z, _, _, _ = path.make_zt(fb, Y0=nz, t=torch.full((1, 1), tf, device=dev))
            step_mod(z, fb, loss_fn)
    for h in hooks:
        h.remove()
    return amx


def parse_args(argv=None):
    """Command-line options."""
    ap = argparse.ArgumentParser(description="Train the ORBIS flow model on fresh views of cell stores.")
    ap.add_argument("--prebuilt_dir", required=True,
                    help="prebuilt training graph directory: supplies the gene count, the position dimension and, with "
                         "--views_per_epoch 0, the number of views per epoch (its graph count)")
    ap.add_argument("--views_stores", required=True, help="comma-separated cell stores the training views are drawn from")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--resume_ckpt", default="", help="latest.pt of an interrupted run: continue from its next epoch")
    ap.add_argument("--gene_columns_start", type=int, default=0)
    ap.add_argument("--gene_columns_end", type=int, default=1122, help="end - start must equal the stores' gene count")
    ap.add_argument("--epochs", type=int, default=3500)
    ap.add_argument("--lr", type=float, default=5e-4, help="AdamW (AMSGrad) learning rate")
    ap.add_argument("--eps", type=float, default=1e-3, help="training times are clamped to [eps, 1 - eps]")
    ap.add_argument("--t_mu", type=float, default=0.0, help="training time law: logit-normal mean")
    ap.add_argument("--t_sigma", type=float, default=1.0, help="training time law: logit-normal SD (both components)")
    ap.add_argument("--t_mix_p", type=float, default=0.0, help="probability of drawing t from the second logit-normal component")
    ap.add_argument("--t_mix_mu", type=float, default=2.32, help="mean of the second logit-normal component")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=50, help="also save orbis_epoch<E>.pt every this many epochs (0 = final only)")
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay of the weights per optimiser step (> 0)")
    ap.add_argument("--ema_rampup", type=float, default=0.0, help="EMA ramp-up ratio: half-life at most this x the steps taken (0 = off)")
    ap.add_argument("--grad_clip", type=float, default=0.0, help="clip the global gradient norm to this value (0 = off)")
    ap.add_argument("--lr_decay_epochs", type=int, default=0, help="decay the learning rate linearly to 0 over the last N epochs (0 = constant)")
    ap.add_argument("--qk_norm", type=int, default=0, help="1 = LayerNorm on queries and keys in every attention")
    ap.add_argument("--log_attn", type=int, default=0,
                    help="1 = log the largest attention logit per block every epoch on a copy of the run's first training view "
                         "(one GPU only; the copy is thinned with its own draw)")
    ap.add_argument("--log_gnorm", type=int, default=0, help="1 = with --grad_clip 0, log the gradient norm of every step")
    ap.add_argument("--accum", type=int, default=1, help="views per optimiser step on each GPU (gradient accumulation)")
    ap.add_argument("--global_batch", type=int, default=0,
                    help="views per optimiser step summed over all GPUs, independent of the GPU count (0 = use --accum)")
    ap.add_argument("--views_whole", default="", help="comma-separated store indices allowed in WHOLE views (default: all)")
    ap.add_argument("--views_mix", default="0.5,0.25,0.25", help="probabilities of WHOLE, SECTION and BLOCK views")
    ap.add_argument("--views_n", default="10000,20000", help="cells per view, uniform in this range")
    ap.add_argument("--views_thin", type=float, default=1.0, help="binomial count thinning with keep probability U(views_thin, 1); 1 = off")
    ap.add_argument("--views_workers", type=int, default=4, help="DataLoader workers drawing views (each has its own random stream)")
    ap.add_argument("--views_pin", type=int, default=1, help="1 = pinned host memory for the views")
    ap.add_argument("--views_fold_axis", type=int, default=-1, help="fold every view across the plane x_axis = 0 (x -> |x|); -1 = off")
    ap.add_argument("--views_per_epoch", type=int, default=0, help="views per epoch (0 = the graph count of --prebuilt_dir)")
    ap.add_argument("--view_label", type=int, default=0, help="1 = the network reads the view label (sampling uses WHOLE)")
    ap.add_argument("--loss_mse", type=int, default=0, help="1 = coordinate MSE; 0 = plane-balanced pairwise-distance loss")
    ap.add_argument("--plane_balanced", type=float, default=0.25, help="within-plane weight of the pairwise-distance loss (--loss_mse 0)")
    ap.add_argument("--unit_box_views", type=int, default=0, help="1 = rescale every view's positions per axis to [-0.5, 0.5], then centre")
    ap.add_argument("--dit_hidden", type=int, default=256)
    ap.add_argument("--dit_layers", type=int, default=8)
    ap.add_argument("--dit_heads", type=int, default=8)
    ap.add_argument("--xt_precond", type=int, default=0)
    ap.add_argument("--edm_weight", type=int, default=0)
    ap.add_argument("--data_rms", type=float, default=0.0, help="per-axis RMS of the training layouts (0 = default)")
    ap.add_argument("--noise_scale", type=float, default=1.0, help="SD of the flow's noise")
    args = ap.parse_args(argv)
    assert args.ema > 0, "--ema: a decay > 0"
    assert args.loss_mse or args.plane_balanced > 0, "--loss_mse 0 needs --plane_balanced > 0"
    return args


def main(argv=None):
    """Train; rank 0 writes the checkpoints."""
    args = parse_args(argv)
    use_cuda = torch.cuda.is_available()
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    GB_K, GB_REAL = 1, [True]
    if args.global_batch > 0:
        GB_K, GB_REAL = micro_plan(args.global_batch, world, rank)
        args.accum = GB_K
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if use_cuda:
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
    else:
        dev = torch.device("cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    infos = read_infos(args.prebuilt_dir)
    graphs = PrebuiltGraphs(args.prebuilt_dir)
    ntr = len(graphs)
    assert infos.num_genes == args.gene_columns_end - args.gene_columns_start, \
        f"expected {args.gene_columns_end - args.gene_columns_start} genes got {infos.num_genes}"
    pos_dim = graphs.pos_dim
    n_view_types = N_VIEW_TYPES if args.view_label else 0
    saved_args = {**vars(args), "world_size": world, "pos_dim": pos_dim, "n_view_types": n_view_types}
    net = build_model(saved_args, dev)

    start_ep, resume_sd = 0, None
    if args.resume_ckpt:
        resume_sd = torch.load(args.resume_ckpt, map_location=dev, weights_only=True)
        net.load_state_dict(resume_sd.get("raw_state_dict") or resume_sd["state_dict"])
        start_ep = int(resume_sd.get("epoch", -1)) + 1
        log0(f"resumed from {args.resume_ckpt}: continuing at epoch {start_ep}")
    step_mod = TrainStep(net).to(dev)
    ddp = DDP(step_mod, device_ids=[local_rank] if use_cuda else None, output_device=local_rank if use_cuda else None,
              find_unused_parameters=True)
    path = LinearPath(eps=args.eps, t_mu=args.t_mu, t_sigma=args.t_sigma, noise_scale=args.noise_scale, t_mix_p=args.t_mix_p,
                      t_mix_mu=args.t_mix_mu)
    loss_fn = CoordMSELoss() if args.loss_mse else PlaneBalancedDistLoss(w_within=args.plane_balanced)
    opt = torch.optim.AdamW(list(net.parameters()), lr=args.lr, amsgrad=True, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    if resume_sd is not None and resume_sd.get("opt_state") is not None:
        opt.load_state_dict(resume_sd["opt_state"])
    nstep = ntr // world
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    if resume_sd is not None and resume_sd.get("raw_state_dict") is not None:
        ema = {k: v.detach().clone().to(dev) for k, v in resume_sd["state_dict"].items()}
    n_opt_steps, steps_per_ep, gnorms = [0], [1], []

    def do_step():
        if args.lr_decay_epochs > 0:
            f = min(1.0, max(0.0, (args.epochs - n_opt_steps[0] / steps_per_ep[0]) / args.lr_decay_epochs))
            for g in opt.param_groups:
                g["lr"] = args.lr * f
        if args.grad_clip > 0:
            gnorms.append(float(torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)))
        elif args.log_gnorm:
            gnorms.append(float(torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm() for p in net.parameters()
                                                                      if p.grad is not None]))))
        opt.step()
        n_opt_steps[0] += 1
        beta = args.ema
        if args.ema_rampup > 0:
            hl = min(np.log(0.5) / np.log(args.ema), args.ema_rampup * n_opt_steps[0])
            beta = 0.5 ** (1.0 / max(hl, 1e-8))
        with torch.no_grad():
            for k, v in net.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(beta).add_(v.detach(), alpha=1 - beta)
                else:
                    ema[k].copy_(v)

    def save_ckpt(p, ep, opt_state=False):
        ck = {"state_dict": ema, "raw_state_dict": net.state_dict(), "epoch": ep, "args": saved_args}
        if opt_state:
            ck["opt_state"] = opt.state_dict()
        torch.save(ck, p)

    if is_main():
        os.makedirs(args.out, exist_ok=True)
        log0(f"world={world} prebuilt graphs={ntr} genes={infos.num_genes} pos_dim={pos_dim} "
             f"batch per optimiser step={args.global_batch if args.global_batch > 0 else world * args.accum} epochs={args.epochs}")
        log0(f"model params={sum(p.numel() for p in net.parameters()) / 1e6:.2f}M lr={args.lr} wd={WEIGHT_DECAY}")

    dit = next((m for m in net.modules() if isinstance(m, DiT3D)), None) if args.view_label else None
    vds = FreshViewDataset(args.views_stores.split(","), seed=args.seed * 100003 + rank * 1009 + start_ep,
                           mix=[float(x) for x in args.views_mix.split(",")],
                           n_range=tuple(int(x) for x in args.views_n.split(",")), thin_min=args.views_thin,
                           whole_animals=[int(x) for x in args.views_whole.split(",")] if args.views_whole else None,
                           fold_axis=args.views_fold_axis)
    view_it = iter(torch.utils.data.DataLoader(vds, batch_size=None, num_workers=args.views_workers, pin_memory=bool(args.views_pin),
                                               prefetch_factor=4 if args.views_workers else None,
                                               persistent_workers=bool(args.views_workers)))
    if is_main():
        v0 = to_gpu_batch(next(view_it), dev)
        P0 = v0.positions[0][v0.node_mask[0]]
        log0(f"[views] stores={args.views_stores} mix={args.views_mix} n={args.views_n} thin>={args.views_thin} | first view: "
             f"label {int(v0.view_type)} store {v0.animal} cells {int(v0.node_mask[0].sum())} keep {v0.thin:.2f} "
             f"pos RMS/axis {P0.pow(2).mean(0).sqrt().tolist()}")
    steps_per_ep[0] = max(1, (args.views_per_epoch if args.views_per_epoch > 0 else nstep) // args.accum)
    if args.global_batch > 0:
        steps_per_ep[0] = max(1, (args.views_per_epoch if args.views_per_epoch > 0 else ntr) // args.global_batch)
        log0(f"global batch {args.global_batch} over {world} GPU(s): {GB_K} micro-step(s) per optimiser step, "
             f"{steps_per_ep[0]} optimiser steps per epoch")
    if start_ep > 0:
        n_opt_steps[0] = start_ep * max(1, (args.views_per_epoch if args.views_per_epoch > 0 else nstep) // args.accum)
        if args.global_batch > 0:
            n_opt_steps[0] = start_ep * steps_per_ep[0]
    if args.global_batch > 0:
        n_views = steps_per_ep[0] * GB_K
    elif args.views_per_epoch > 0:
        n_views = args.views_per_epoch
    else:
        n_views = nstep

    assert not args.log_attn or world == 1, "--log_attn: one GPU"
    attn_fix = None
    train_t0 = time.time()
    for ep in range(start_ep, args.epochs):
        ddp.train()
        if args.accum > 1:
            opt.zero_grad()
        t0, Ls, vstat = time.time(), [], []
        gnorms.clear()
        for k in range(n_views):
            item = next(view_it)
            batched = to_gpu_batch(item, dev)
            if args.unit_box_views:
                batched.positions = unit_box_positions(batched.positions, batched.node_mask)[0]
            if args.log_attn and attn_fix is None:
                attn_fix = to_gpu_batch(item, dev)
            vstat.append((int(batched.view_type), batched.animal, int(batched.node_mask.sum()), batched.thin))
            if dit is not None:
                dit.view_label = batched.view_type
            z, t, Y0, Y1 = path.make_zt(batched)
            loss = ddp(z, batched, loss_fn)
            if args.edm_weight:
                d_, s_ = (args.data_rms or args.noise_scale), args.noise_scale
                tw = t.view(-1)[0]
                loss = loss * ((tw * d_) ** 2 + ((1 - tw) * s_) ** 2) / (((1 - tw) * s_) ** 2)
            if args.global_batch > 0:
                mstep = k % GB_K
                if mstep == 0:
                    opt.zero_grad()
                (loss / loss_divisor(args.global_batch, world) if GB_REAL[mstep] else loss * 0.0).backward()
                if mstep == GB_K - 1:
                    do_step()
                    opt.zero_grad()
                if GB_REAL[mstep]:
                    Ls.append(loss.item())
                continue
            if args.accum == 1:
                opt.zero_grad()
                loss.backward()
                do_step()
            else:
                (loss / args.accum).backward()
                if (k + 1) % args.accum == 0:
                    do_step()
                    opt.zero_grad()
            Ls.append(loss.item())
        lt = torch.tensor([float(np.mean(Ls)) if Ls else 0.0], device=dev)
        dist.all_reduce(lt, op=dist.ReduceOp.SUM)
        lt /= world
        if is_main():
            log0(f"ep {ep:04d} ({time.time() - t0:5.0f}s) loss={lt.item():.5f} views/gpu={n_views}"
                 + (f" lr={opt.param_groups[0]['lr']:.3e} opt_step={n_opt_steps[0]}" if args.lr_decay_epochs > 0 else ""))
            if gnorms:
                g = np.asarray(gnorms)
                log0(f"[grad] ep {ep}: pre-clip norm median {np.median(g):.3g} p99 {np.percentile(g, 99):.3g} max {g.max():.3g}; "
                     + (f"clipped {np.mean(g > args.grad_clip):.1%} of {len(g)} steps (clip {args.grad_clip})" if args.grad_clip > 0
                        else f"no clip ({len(g)} steps)"))
            if vstat and (ep < 3 or (ep + 1) % 25 == 0):
                va = np.array(vstat, dtype=float)
                log0(f"[views] ep {ep}: labels W/Scor/Ssag/Bcor/Bsag {[int((va[:, 0] == i).sum()) for i in range(5)]} stores "
                     f"{dict(zip(*np.unique(va[:, 1].astype(int), return_counts=True)))} cells mean {va[:, 2].mean():.0f} "
                     f"min {va[:, 2].min():.0f} keep mean {va[:, 3].mean():.2f}")
            save_ckpt(os.path.join(args.out, "latest.pt"), ep, opt_state=True)
            if args.save_every > 0 and (ep + 1) % args.save_every == 0 and ep != args.epochs - 1:
                save_ckpt(os.path.join(args.out, f"orbis_epoch{ep}.pt"), ep)
                log0(f"checkpoint: orbis_epoch{ep}.pt")
        dist.barrier()
        if args.log_attn and attn_fix is not None and is_main():
            if dit is not None:
                dit.view_label = attn_fix.view_type
            amx = attention_logit_max(net, step_mod, path, loss_fn, attn_fix, args.seed, dev)
            log0(f"[attn] ep {ep}: max |q.k|/sqrt(d) per block (fixed view, t 0.5 / 0.95): "
                 + " ".join(f"{n.split('.')[-2]}:{v:.4g}" for n, v in amx.items()))

    if is_main():
        p = os.path.join(args.out, f"orbis_epoch{args.epochs - 1}.pt")
        save_ckpt(p, args.epochs - 1)
        tot = time.time() - train_t0
        log0(f"checkpoint saved: {p} ({os.path.getsize(p) / 1e6:.1f} MB); wall-clock {tot:.1f}s ({tot / 3600:.2f}h) over "
             f"{args.epochs - start_ep} epochs, world={world}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
