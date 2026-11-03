"""CUDA niche engine of the reseating step.

Runs the energy and moves of engine_cpu on one GPU. The engine mirrors an engine_cpu.Engine (its graphs, targets, caches, anchor scale
and moment references). Every energy change is computed with the CPU engine's summation orders, in IEEE double without fused
multiply-add (nvcc --fmad=false), so the change of one proposal is bit-identical to engine_cpu.delta_total on the same state. Proposals
whose footprints (the sites whose neighbour lists hold a or b, plus a and b) are disjoint commute exactly, so each round evaluates one
proposal per block of one colour of a 3-D checkerboard in parallel; proposals whose footprint leaves the dilated block are deferred to
a shifted grid or run alone. The moment term is global and is applied by a sequential scan over each round's proposals. Each sweep
ends with an exact recompute from scratch and an energy drift check. The kernels are compiled at first use with nvcc (path given by the
caller) and cached in a build directory."""
import ctypes
import hashlib
import os
import subprocess
import time

import numba as nb
import numpy as np
import torch

from orbis.reseat import engine_cpu as NA

NVCC_FLAGS = ["-O3", "--fmad=false", "-gencode", "arch=compute_86,code=sm_86", "-gencode", "arch=compute_86,code=compute_86",
              "-Xcompiler", "-fPIC", "-shared"]
EPS_BLOCK = 1e-9

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

struct NGState {
  int64_t n, K, r1, r2, use_niche, use_unary, use_mom, W1, W2, Wmax;
  double delta, T, cap2, inv2tau2;
  const double *S, *p, *G, *q1, *q2, *isig1, *isig2, *M, *mu0, *s0;
  double *m1, *m2, *e1, *e2, *SX, *SXX;
  const int64_t *in1p; const int32_t *in1i; const int64_t *in2p; const int32_t *in2i;
  int64_t *cell_at; const int64_t *orig_site;
  int64_t *markA, *markB; unsigned long long *owner;
  double *ev1, *ev2;
  int32_t *valid, *acc; int64_t *oca, *ocb; double *d1, *d2, *dun, *dmom;
  double *terms;
  long long *counters;
};

__device__ __forceinline__ double huber(double z, double delta) {
  double az = fabs(z);
  return az <= delta ? 0.5 * z * z : delta * (az - 0.5 * delta);
}

__device__ double pen_k(double sx0, double sx1, double sx2, double sxx, double Mk, const double* mu0k, double s0k) {
  double mx = sx0 / Mk;
  double my = sx1 / Mk;
  double mz = sx2 / Mk;
  double v = sxx / Mk - (mx * mx + my * my + mz * mz);
  double s = v > 0.0 ? sqrt(v) : 0.0;
  double t0 = mx - mu0k[0], t1 = my - mu0k[1], t2 = mz - mu0k[2];
  double dmu = t0 * t0 + t1 * t1 + t2 * t2;
  double ds = s - s0k;
  return 3.0 * Mk * dmu / (2.0 * s0k * s0k) + Mk * (ds * ds) / (s0k * s0k);
}

__device__ __forceinline__ int entry(int64_t e, int64_t la, int64_t lb, const int32_t* ini, int64_t a0, int64_t b0, int64_t a, int64_t b,
                                     const int64_t* markA, const int64_t* markB, int64_t stamp, int64_t* v, double* wa, double* wb) {
  if (e < la) { *v = ini[a0 + e]; *wa = 1.0; int s = markB[*v] == stamp; *wb = s ? 1.0 : 0.0; return s; }
  if (e < la + lb) { *v = ini[b0 + e - la]; *wa = 0.0; *wb = 1.0; return markA[*v] == stamp; }
  if (e == la + lb) { *v = a; *wa = 0.0; *wb = 0.0; return markB[a] == stamp; }
  *v = b; *wa = 0.0; *wb = 0.0; return markA[b] == stamp;
}

__global__ void k_eval(NGState st, const int64_t* pa, const int64_t* pb, int64_t off, int64_t stamp0) {
  extern __shared__ double sh[];
  const int K = (int)st.K;
  double* dp = sh;
  double* contrib = sh + K;
  unsigned char* omit = (unsigned char*)(contrib + st.Wmax);
  __shared__ int s_ok;
  const int i = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  const int64_t g = off + i;
  const int64_t a = pa[i], b = pb[i];
  const int64_t ca = st.cell_at[a], cb = st.cell_at[b];
  if (tid == 0) {
    const int64_t oa = st.orig_site[ca], ob = st.orig_site[cb];
    double da = 0.0, db = 0.0;
    for (int x = 0; x < 3; x++) {
      double t1 = st.S[b * 3 + x] - st.S[oa * 3 + x];
      double t2 = st.S[a * 3 + x] - st.S[ob * 3 + x];
      da += t1 * t1;
      db += t2 * t2;
    }
    int ok = !(da > st.cap2 || db > st.cap2);
    s_ok = ok;
    st.valid[g] = ok; st.oca[g] = ca; st.ocb[g] = cb;
    st.d1[g] = 0.0; st.d2[g] = 0.0; st.dun[g] = 0.0; st.dmom[g] = 0.0; st.acc[g] = 0;
    if (!ok) atomicAdd((unsigned long long*)&st.counters[0], 1ULL);
  }
  __syncthreads();
  if (!s_ok) return;
  for (int k = tid; k < K; k += nt) dp[k] = st.p[cb * K + k] - st.p[ca * K + k];
  __syncthreads();
  if (st.use_niche) {
    for (int s = 0; s < 2; s++) {
      const int64_t r = s == 0 ? st.r1 : st.r2;
      const int64_t* inp = s == 0 ? st.in1p : st.in2p;
      const int32_t* ini = s == 0 ? st.in1i : st.in2i;
      const double* m = s == 0 ? st.m1 : st.m2;
      const double* ec = s == 0 ? st.e1 : st.e2;
      const double* q = s == 0 ? st.q1 : st.q2;
      const double* isig = s == 0 ? st.isig1 : st.isig2;
      double* evb = s == 0 ? st.ev1 + (int64_t)i * st.W1 : st.ev2 + (int64_t)i * st.W2;
      const int64_t stamp = stamp0 + 2 * (int64_t)i + s + 1;
      const int64_t a0 = inp[a], la = inp[a + 1] - a0, b0 = inp[b], lb = inp[b + 1] - b0;
      for (int64_t t = tid; t < la; t += nt) st.markA[ini[a0 + t]] = stamp;
      for (int64_t t = tid; t < lb; t += nt) st.markB[ini[b0 + t]] = stamp;
      __syncthreads();
      const int64_t nE = la + lb + 2;
      for (int64_t e = tid; e < nE; e += nt) {
        int64_t v; double wa, wb;
        int sk = entry(e, la, lb, ini, a0, b0, a, b, st.markA, st.markB, stamp, &v, &wa, &wb);
        omit[e] = (unsigned char)sk;
        if (sk) continue;
        const double coef = (wa - wb) / (double)r;
        const int64_t cv = v == a ? cb : (v == b ? ca : st.cell_at[v]);
        double ev = 0.0;
        for (int k = 0; k < K; k++) {
          const double mk = m[v * K + k] + coef * dp[k];
          ev += huber((mk - q[cv * K + k]) * isig[k], st.delta);
        }
        evb[e] = ev;
        contrib[e] = ev - ec[v];
      }
      __syncthreads();
      if (tid == 0) {
        double dE = 0.0;
        for (int64_t e = 0; e < nE; e++)
          if (!omit[e]) dE += contrib[e];
        if (s == 0) st.d1[g] = dE; else st.d2[g] = dE;
      }
      __syncthreads();
    }
  }
  if (tid == 0 && st.use_unary) {
    double uo = 0.0, un = 0.0;
    for (int x = 0; x < 3; x++) {
      const double sa = st.S[a * 3 + x], sb = st.S[b * 3 + x], ga = st.G[ca * 3 + x], gb = st.G[cb * 3 + x];
      const double t1 = sa - ga, t2 = sb - gb, t3 = sb - ga, t4 = sa - gb;
      uo += t1 * t1 + t2 * t2;
      un += t3 * t3 + t4 * t4;
    }
    st.dun[g] = (un - uo) * st.inv2tau2;
  }
}

__global__ void k_scan(NGState st, const int64_t* pa, const int64_t* pb, const double* u, int64_t off, int64_t P, int64_t dry) {
  extern __shared__ double sh2[];
  const int K = (int)st.K;
  const int k = threadIdx.x;
  double* contrib = sh2;
  unsigned char* nz = (unsigned char*)(sh2 + K);
  __shared__ int s_acc;
  const int um = (int)st.use_mom;
  double pen_old = 0.0;
  if (um && k < K) pen_old = pen_k(st.SX[3 * k], st.SX[3 * k + 1], st.SX[3 * k + 2], st.SXX[k], st.M[k], st.mu0 + 3 * k, st.s0[k]);
  for (int64_t i = 0; i < P; i++) {
    const int64_t g = off + i;
    if (!st.valid[g]) continue;
    const int64_t a = pa[i], b = pb[i], ca = st.oca[g], cb = st.ocb[g];
    double dpk = 0.0, pn = 0.0, t0 = 0.0, t1 = 0.0, t2 = 0.0, txx = 0.0;
    if (um && k < K) {
      dpk = st.p[cb * K + k] - st.p[ca * K + k];
      const double ax = st.S[a * 3], ay = st.S[a * 3 + 1], az = st.S[a * 3 + 2];
      const double bx = st.S[b * 3], by = st.S[b * 3 + 1], bz = st.S[b * 3 + 2];
      const double sa2 = ax * ax + ay * ay + az * az;
      const double sb2 = bx * bx + by * by + bz * bz;
      t0 = st.SX[3 * k] + dpk * (ax - bx);
      t1 = st.SX[3 * k + 1] + dpk * (ay - by);
      t2 = st.SX[3 * k + 2] + dpk * (az - bz);
      txx = st.SXX[k] + dpk * (sa2 - sb2);
      nz[k] = dpk != 0.0;
      if (dpk != 0.0) {
        pn = pen_k(t0, t1, t2, txx, st.M[k], st.mu0 + 3 * k, st.s0[k]);
        contrib[k] = pn - pen_old;
      }
    }
    __syncthreads();
    if (k == 0) {
      double dm = 0.0;
      if (um)
        for (int kk = 0; kk < K; kk++)
          if (nz[kk]) dm += contrib[kk];
      const double dU = st.d1[g] + st.d2[g] + st.dun[g] + dm;
      int ac = 0;
      if (!dry) ac = (dU <= 0.0) || (u[i] < exp(-dU / st.T));
      s_acc = ac;
      st.acc[g] = ac;
      st.dmom[g] = dm;
      if (ac) {
        st.terms[0] += st.d1[g];
        st.terms[1] += st.d2[g];
        st.terms[2] += st.dun[g];
        st.terms[3] += dm;
        st.counters[1] += 1;
        if (dU > 0.0) { st.counters[2] += 1; st.terms[4] += dU; }
      }
    }
    __syncthreads();
    if (s_acc && um && k < K) {
      st.SX[3 * k] = t0; st.SX[3 * k + 1] = t1; st.SX[3 * k + 2] = t2; st.SXX[k] = txx;
      if (dpk != 0.0) pen_old = pn;
    }
    __syncthreads();
  }
}

__global__ void k_apply(NGState st, const int64_t* pa, const int64_t* pb, int64_t off, int64_t stamp0) {
  extern __shared__ double sh3[];
  const int K = (int)st.K;
  const int i = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  const int64_t g = off + i;
  if (!st.acc[g]) return;
  const int64_t a = pa[i], b = pb[i], ca = st.oca[g], cb = st.ocb[g];
  double* dp = sh3;
  for (int k = tid; k < K; k += nt) dp[k] = st.p[cb * K + k] - st.p[ca * K + k];
  __syncthreads();
  if (st.use_niche) {
    for (int s = 0; s < 2; s++) {
      const int64_t r = s == 0 ? st.r1 : st.r2;
      const int64_t* inp = s == 0 ? st.in1p : st.in2p;
      const int32_t* ini = s == 0 ? st.in1i : st.in2i;
      double* m = s == 0 ? st.m1 : st.m2;
      double* ec = s == 0 ? st.e1 : st.e2;
      const double* evb = s == 0 ? st.ev1 + (int64_t)i * st.W1 : st.ev2 + (int64_t)i * st.W2;
      const int64_t stamp = stamp0 + 2 * (int64_t)i + s + 1;
      const int64_t a0 = inp[a], la = inp[a + 1] - a0, b0 = inp[b], lb = inp[b + 1] - b0;
      for (int64_t t = tid; t < la; t += nt) st.markA[ini[a0 + t]] = stamp;
      for (int64_t t = tid; t < lb; t += nt) st.markB[ini[b0 + t]] = stamp;
      __syncthreads();
      const int64_t nE = la + lb + 2;
      for (int64_t e = tid; e < nE; e += nt) {
        int64_t v; double wa, wb;
        if (entry(e, la, lb, ini, a0, b0, a, b, st.markA, st.markB, stamp, &v, &wa, &wb)) continue;
        const double coef = (wa - wb) / (double)r;
        for (int k = 0; k < K; k++) m[v * K + k] = m[v * K + k] + coef * dp[k];
        ec[v] = evb[e];
      }
      __syncthreads();
    }
  }
  __syncthreads();
  if (tid == 0) { st.cell_at[a] = cb; st.cell_at[b] = ca; }
}

__global__ void k_claim(NGState st, const int64_t* pa, const int64_t* pb, unsigned long long round_id) {
  const int i = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  const unsigned long long tok = (round_id << 32) | (unsigned long long)(i + 1);
  const int64_t a = pa[i], b = pb[i];
  const int64_t a0 = st.in2p[a], la = st.in2p[a + 1] - a0, b0 = st.in2p[b], lb = st.in2p[b + 1] - b0;
  for (int64_t e = tid; e < la + lb + 2; e += nt) {
    int64_t v = e < la ? st.in2i[a0 + e] : (e < la + lb ? st.in2i[b0 + e - la] : (e == la + lb ? a : b));
    unsigned long long old = st.owner[v];
    while (true) {
      if (old == tok) break;
      if ((old >> 32) == round_id) { atomicAdd((unsigned long long*)&st.counters[3], 1ULL); break; }
      unsigned long long got = atomicCAS(&st.owner[v], old, tok);
      if (got == old) break;
      old = got;
    }
  }
}

__global__ void k_env(int64_t n, int64_t K, int64_t r, const int32_t* nbr, const int64_t* cell_at, const double* p, double* m) {
  const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (idx >= n * K) return;
  const int64_t a = idx / K, k = idx % K;
  double acc = 0.0;
  for (int64_t j = 0; j < r; j++) acc += p[cell_at[nbr[a * r + j]] * K + k];
  m[idx] = acc / (double)r;
}

__global__ void k_site_energy(int64_t n, int64_t K, const int64_t* cell_at, const double* m, const double* q, const double* isig, double delta,
                              double* e) {
  const int64_t a = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (a >= n) return;
  const int64_t c = cell_at[a];
  double ev = 0.0;
  for (int64_t k = 0; k < K; k++) ev += huber((m[a * K + k] - q[c * K + k]) * isig[k], delta);
  e[a] = ev;
}

static size_t sh_eval(const NGState* st) { return (size_t)(st->K + st->Wmax) * 8 + (size_t)st->Wmax + 16; }

extern "C" {
int ng_set_device(int dev) { return (int)cudaSetDevice(dev); }
int ng_sync() { return (int)cudaDeviceSynchronize(); }

int ng_sweep(const NGState* st, const int64_t* pa, const int64_t* pb, const double* u, const int64_t* rs, int64_t n_rounds,
             int64_t stamp0, unsigned long long round_id0, int64_t check, int64_t threads, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  const int scan_threads = (int)(((st->K + 31) / 32) * 32);
  const size_t sh_scan = (size_t)st->K * 9 + 16, sh_apply = (size_t)st->K * 8 + 16, she = sh_eval(st);
  int64_t stamp = stamp0;
  for (int64_t r = 0; r < n_rounds; r++) {
    const int64_t o = rs[r], P = rs[r + 1] - rs[r];
    if (P <= 0) continue;
    if (check) k_claim<<<(unsigned)P, (unsigned)threads, 0, s>>>(*st, pa + o, pb + o, round_id0 + (unsigned long long)r);
    k_eval<<<(unsigned)P, (unsigned)threads, she, s>>>(*st, pa + o, pb + o, o, stamp);
    k_scan<<<1, scan_threads, sh_scan, s>>>(*st, pa + o, pb + o, u + o, o, P, 0);
    k_apply<<<(unsigned)P, (unsigned)threads, sh_apply, s>>>(*st, pa + o, pb + o, o, stamp);
    stamp += 2 * P + 2;
  }
  return (int)cudaGetLastError();
}

int ng_eval_only(const NGState* st, const int64_t* pa, const int64_t* pb, const double* u, int64_t P, int64_t stamp0, int64_t threads,
                 void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  const int scan_threads = (int)(((st->K + 31) / 32) * 32);
  k_eval<<<(unsigned)P, (unsigned)threads, sh_eval(st), s>>>(*st, pa, pb, 0, stamp0);
  k_scan<<<1, scan_threads, (size_t)st->K * 9 + 16, s>>>(*st, pa, pb, u, 0, P, 1);
  return (int)cudaGetLastError();
}

int ng_env(int64_t n, int64_t K, int64_t r, const int32_t* nbr, const int64_t* cell_at, const double* p, double* m, void* stream) {
  const int64_t tot = n * K;
  k_env<<<(unsigned)((tot + 255) / 256), 256, 0, (cudaStream_t)stream>>>(n, K, r, nbr, cell_at, p, m);
  return (int)cudaGetLastError();
}

int ng_site_energy(int64_t n, int64_t K, const int64_t* cell_at, const double* m, const double* q, const double* isig, double delta, double* e,
                   void* stream) {
  k_site_energy<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(n, K, cell_at, m, q, isig, delta, e);
  return (int)cudaGetLastError();
}
}
"""

_FIELDS_I = ["n", "K", "r1", "r2", "use_niche", "use_unary", "use_mom", "W1", "W2", "Wmax"]
_FIELDS_D = ["delta", "T", "cap2", "inv2tau2"]
_FIELDS_P = ["S", "p", "G", "q1", "q2", "isig1", "isig2", "M", "mu0", "s0", "m1", "m2", "e1", "e2", "SX", "SXX", "in1p", "in1i", "in2p",
             "in2i", "cell_at", "orig_site", "markA", "markB", "owner", "ev1", "ev2", "valid", "acc", "oca", "ocb", "d1", "d2", "dun", "dmom",
             "terms", "counters"]


class NGState(ctypes.Structure):
    """ctypes mirror of the CUDA state struct (every field 8 bytes, no padding)."""
    _fields_ = [(f, ctypes.c_int64) for f in _FIELDS_I] + [(f, ctypes.c_double) for f in _FIELDS_D] + [(f, ctypes.c_void_p) for f in _FIELDS_P]


_LIB = None


def lib(nvcc, ccbin, build_dir):
    """Compile CUDA_SRC once with nvcc (cached in build_dir by the md5 of the source, compiler and flags) and load it."""
    global _LIB
    if _LIB is not None:
        return _LIB
    key = hashlib.md5((CUDA_SRC + " ".join([nvcc, ccbin] + NVCC_FLAGS)).encode()).hexdigest()[:16]
    so = f"{build_dir}/niche_gpu_{key}.so"
    if not os.path.exists(so):
        os.makedirs(build_dir, exist_ok=True)
        cu = f"{build_dir}/niche_gpu_{key}.cu"
        with open(cu, "w") as f:
            f.write(CUDA_SRC)
        tmp = f"{so}.{os.getpid()}.tmp"
        subprocess.run([nvcc, "-ccbin", ccbin] + NVCC_FLAGS + ["-o", tmp, cu], check=True)
        os.replace(tmp, so)
    L = ctypes.CDLL(so)
    vp, i64, dbl, u64 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_double, ctypes.c_ulonglong
    L.ng_set_device.argtypes = [ctypes.c_int]
    L.ng_sweep.argtypes = [ctypes.POINTER(NGState), vp, vp, vp, vp, i64, i64, u64, i64, i64, vp]
    L.ng_eval_only.argtypes = [ctypes.POINTER(NGState), vp, vp, vp, i64, i64, i64, vp]
    L.ng_env.argtypes = [i64, i64, i64, vp, vp, vp, vp, vp]
    L.ng_site_energy.argtypes = [i64, i64, vp, vp, vp, vp, dbl, vp, vp]
    for fn in ("ng_set_device", "ng_sync", "ng_sweep", "ng_eval_only", "ng_env", "ng_site_energy"):
        getattr(L, fn).restype = ctypes.c_int
    L.so_path = so
    _LIB = L
    return L


def _ok(err, what):
    """Raise on a non-zero CUDA error code."""
    if err != 0:
        raise RuntimeError(f"CUDA error {err} in {what}")


@nb.njit(cache=True, parallel=True)
def influence_boxes(S, in_ptr, in_idx):
    """Per site x: the exact bounding box of x and of the sites whose neighbour list holds x."""
    n = S.shape[0]
    lo = S.copy()
    hi = S.copy()
    for x in nb.prange(n):
        for t in range(in_ptr[x], in_ptr[x + 1]):
            v = in_idx[t]
            for d in range(3):
                if S[v, d] < lo[x, d]:
                    lo[x, d] = S[v, d]
                if S[v, d] > hi[x, d]:
                    hi[x, d] = S[v, d]
    return lo, hi


def reach_of(S, lo, hi, a, b):
    """L-infinity half-extent, around the initiator a, of the footprint box of swap (a, b)."""
    ulo = np.minimum(lo[a], lo[b])
    uhi = np.maximum(hi[a], hi[b])
    return np.maximum(S[a] - ulo, uhi - S[a]).max(1)


def bin_contained(S, lo, hi, a, b, origin, shift, L):
    """Block of each initiator on a grid of side L (offset shift) and whether the footprint box of (a, b) lies inside the block dilated by L / 2."""
    t = lambda s: ((s - origin) + shift) / L
    g = np.floor(t(S[a])).astype(np.int64)
    ulo = t(np.minimum(lo[a], lo[b]))
    uhi = t(np.maximum(hi[a], hi[b]))
    ok = ((ulo >= g - 0.5 + EPS_BLOCK) & (uhi <= g + 1.5 - EPS_BLOCK)).all(1)
    return g, ok


def rounds_of(g, rank, phase_pos):
    """Execution order and round boundaries of proposals binned in blocks: colour = block parity per axis, one proposal per block per round."""
    key = (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]
    col = phase_pos[((g[:, 0] & 1) << 2) | ((g[:, 1] & 1) << 1) | (g[:, 2] & 1)]
    o = np.lexsort((rank, key))
    ks = key[o]
    start = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1]
    blk_start = np.repeat(start, np.diff(np.r_[start, len(o)]))
    rk = np.empty(len(o), np.int64)
    rk[o] = np.arange(len(o)) - blk_start
    o2 = np.lexsort((key, rk, col))
    c2, r2 = col[o2], rk[o2]
    cut = np.r_[0, np.flatnonzero((c2[1:] != c2[:-1]) | (r2[1:] != r2[:-1])) + 1, len(o2)]
    return o2, cut


class GPUEngine:
    """Mirror of an engine_cpu.Engine on one GPU. Sweeps run as a 3-D checkerboard of parallel exact swaps with disjoint footprints; the moment term is applied by a sequential scan per round. block_um = block side L (None: 2 x the quantile of the per-proposal reach); check = runtime footprint-claim check on the first / every / no sweep."""
    def __init__(self, eng, nvcc, ccbin, build_dir, block_um=None, quantile=0.99, device=None, check="first", threads=128, log=None):
        t0 = time.time()
        self.eng, self.log, self.threads = eng, (log or eng.log), int(threads)
        self.n, self.K = eng.n, eng.K
        assert len(eng.scales) == 2 and self.n < 2 ** 31
        s1, s2 = eng.scales
        assert s1["r"] <= s2["r"] and np.array_equal(s1["nbr"], s2["nbr"][:, :s1["r"]]), "the smaller scale's lists must be prefixes"
        assert check in ("first", "all", "none")
        self.check = check
        self.dev = torch.device("cuda", torch.cuda.current_device() if device is None else int(device))
        torch.cuda.set_device(self.dev)
        self.lib = lib(nvcc, ccbin, build_dir)
        _ok(self.lib.ng_set_device(self.dev.index), "set_device")
        self.lo, self.hi = influence_boxes(eng.S, s2["in_ptr"], s2["in_idx"])
        self.act = np.flatnonzero(eng.active).astype(np.int64)
        self.deg = np.diff(eng.prop_ptr)
        h = np.maximum(eng.S - self.lo, self.hi - eng.S).max(1)
        reach, edge = [], []
        src_all = np.repeat(np.arange(self.n, dtype=np.int64), self.deg)
        for c0 in range(0, len(src_all), 4_000_000):
            a_, b_ = src_all[c0:c0 + 4_000_000], eng.prop_idx[c0:c0 + 4_000_000]
            reach.append(reach_of(eng.S, self.lo, self.hi, a_, b_))
            edge.append(np.abs(eng.S[a_] - eng.S[b_]).max(1))
        reach = np.concatenate(reach) if reach else np.zeros(1)
        edge = np.concatenate(edge) if edge else np.zeros(1)
        del src_all
        qs_ = [0.5, 0.9, 0.99, 0.999, 1.0]
        self.L = float(block_um) if block_um is not None else float(2.0 * np.quantile(reach, quantile))
        assert self.L > 0
        R_int = 2.0 * (float(h.max()) + float(edge.max()))
        self.origin = eng.S.min(0) - 2.0 * self.L
        self.geom = dict(block_um=self.L, rule=("given" if block_um is not None else f"2 x quantile {quantile} of the per-proposal reach"),
                         reach_um_quantiles=dict(zip(["p50", "p90", "p99", "p999", "max"], np.quantile(reach, qs_).round(2).tolist())),
                         box_halfextent_um_quantiles=dict(zip(["p50", "p90", "p99", "p999", "max"], np.quantile(h, qs_).round(2).tolist())),
                         edge_um_quantiles=dict(zip(["p50", "p90", "p99", "p999", "max"], np.quantile(edge, qs_).round(2).tolist())),
                         R_int_um=R_int, global_bound_holds=bool(self.L >= R_int),
                         share_reach_le_half_block=float(np.mean(reach <= 0.5 * self.L)),
                         note="L-infinity, engine-frame um; R_int = 2 x (max box half-extent + max proposal edge): blocks >= R_int would make "
                              "every proposal contained; the per-proposal containment check enforces exactness for any L")
        D = self.dev
        f64 = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64)).to(D)
        i64 = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.int64)).to(D)
        i32 = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.int32)).to(D)
        self.t = t = dict(S=f64(eng.S), p=f64(eng.p), G=f64(eng.G), q1=f64(s1["q"]), q2=f64(s2["q"]), isig1=f64(s1["isig"]), isig2=f64(s2["isig"]),
                          M=f64(eng.M), mu0=f64(eng.mu0), s0=f64(eng.s0), m1=f64(s1["m"]), m2=f64(s2["m"]), e1=f64(s1["e"]), e2=f64(s2["e"]),
                          SX=f64(eng.SX), SXX=f64(eng.SXX), in1p=i64(s1["in_ptr"]), in1i=i32(s1["in_idx"]), in2p=i64(s2["in_ptr"]),
                          in2i=i32(s2["in_idx"]), nbr1=i32(s1["nbr"]), nbr2=i32(s2["nbr"]), cell_at=i64(eng.cell_at), orig_site=i64(eng.orig_site),
                          markA=torch.zeros(self.n, dtype=torch.int64, device=D), markB=torch.zeros(self.n, dtype=torch.int64, device=D),
                          owner=torch.zeros(self.n, dtype=torch.int64, device=D), terms=torch.zeros(5, dtype=torch.float64, device=D),
                          counters=torch.zeros(4, dtype=torch.int64, device=D), scratch=torch.empty((self.n, self.K), dtype=torch.float64, device=D))
        self.r = (int(s1["r"]), int(s2["r"]))
        self.W = (2 * int(s1["max_in"]) + 2, 2 * int(s2["max_in"]) + 2)
        assert (self.K + max(self.W)) * 9 + 16 < 48 * 1024 and self.K <= 1024, "shared memory of one proposal block"
        self._alloc_props(max(len(self.act), 1))
        self._alloc_ev(1)
        self.stamp, self.round_id = 0, 1
        self.cell_at = eng.cell_at.copy()
        self.orig_site = eng.orig_site.copy()
        self.info = dict(eng.info, engine="gpu", device=torch.cuda.get_device_name(D), so=self.lib.so_path, threads=self.threads, check=check,
                         geometry=self.geom, gpu_setup_seconds=round(time.time() - t0, 1))
        g = self.geom
        self.log(f"  gpu engine: {torch.cuda.get_device_name(D)}; block side L {self.L:.1f} um ({g['rule']}); reach p50 / p99 / p99.9 / max "
                 f"{g['reach_um_quantiles']['p50']:.1f} / {g['reach_um_quantiles']['p99']:.1f} / {g['reach_um_quantiles']['p999']:.1f} / "
                 f"{g['reach_um_quantiles']['max']:.1f} um; global bound R_int = 2 x (max box half-extent {g['box_halfextent_um_quantiles']['max']:.1f} + "
                 f"max proposal edge {g['edge_um_quantiles']['max']:.1f}) = {R_int:.1f} um (holds for L: {g['global_bound_holds']}); share of "
                 f"proposals with reach <= L/2 {100 * g['share_reach_le_half_block']:.3f} %; setup {self.info['gpu_setup_seconds']} s")

    def _alloc_props(self, m):
        D = self.dev
        self.t.update(valid=torch.zeros(m, dtype=torch.int32, device=D), acc=torch.zeros(m, dtype=torch.int32, device=D),
                      oca=torch.zeros(m, dtype=torch.int64, device=D), ocb=torch.zeros(m, dtype=torch.int64, device=D),
                      d1=torch.zeros(m, dtype=torch.float64, device=D), d2=torch.zeros(m, dtype=torch.float64, device=D),
                      dun=torch.zeros(m, dtype=torch.float64, device=D), dmom=torch.zeros(m, dtype=torch.float64, device=D))
        self.n_props_cap = m
        self._state = None

    def _alloc_ev(self, Pmax):
        self.t.update(ev1=torch.zeros((Pmax, self.W[0]), dtype=torch.float64, device=self.dev),
                      ev2=torch.zeros((Pmax, self.W[1]), dtype=torch.float64, device=self.dev))
        self.Pmax_cap = Pmax
        self._state = None

    def state(self):
        """ctypes state struct holding the device pointers and scalars."""
        if self._state is None:
            e, t = self.eng, self.t
            vals = dict(n=self.n, K=self.K, r1=self.r[0], r2=self.r[1], use_niche=int(e.use[0]), use_unary=int(e.use[1]), use_mom=int(e.use[2]),
                        W1=self.W[0], W2=self.W[1], Wmax=max(self.W), delta=e.delta, T=e.T, cap2=e.cap2, inv2tau2=e.inv2tau2)
            vals.update({f: t[f].data_ptr() for f in _FIELDS_P})
            self._state = NGState(**vals)
        return self._state

    def _stream(self):
        return ctypes.c_void_p(torch.cuda.current_stream(self.dev).cuda_stream)

    def schedule(self, rng):
        """Proposals of one sweep in execution order and the round boundaries; all randomness from rng."""
        eng, S, L = self.eng, self.eng.S, self.L
        a_all = self.act
        n_act = len(a_all)
        u_pick = rng.random(n_act)
        u_acc = rng.random(n_act)
        shift = rng.random(3) * L
        phase_pos = np.empty(8, np.int64)
        phase_pos[rng.permutation(8)] = np.arange(8)
        rank = rng.permutation(n_act).astype(np.int64)
        deg = self.deg[a_all]
        has = deg > 0
        a = a_all[has]
        j = np.minimum((u_pick[has] * deg[has]).astype(np.int64), deg[has] - 1)
        b = eng.prop_idx[eng.prop_ptr[a] + j]
        u, rk = u_acc[has], rank[has]
        g1, ok1 = bin_contained(S, self.lo, self.hi, a, b, self.origin, shift, L)
        i1 = np.flatnonzero(ok1)
        o1, cut1 = rounds_of(g1[i1], rk[i1], phase_pos) if len(i1) else (np.zeros(0, np.int64), np.zeros(1, np.int64))
        d = np.flatnonzero(~ok1)
        shift2 = (shift + 0.5 * L) % L
        g2, ok2 = bin_contained(S, self.lo, self.hi, a[d], b[d], self.origin, shift2, L)
        i2 = d[ok2]
        o2, cut2 = rounds_of(g2[ok2], rk[i2], phase_pos) if len(i2) else (np.zeros(0, np.int64), np.zeros(1, np.int64))
        i3 = d[~ok2]
        i3 = i3[np.argsort(rk[i3], kind="stable")]
        order = np.concatenate([i1[o1], i2[o2], i3])
        n1, n2 = len(i1), len(i2)
        rs = np.concatenate([cut1, cut2[1:] + n1, n1 + n2 + np.arange(1, len(i3) + 1)]).astype(np.int64)
        assert rs[-1] == len(order) == len(a) and np.array_equal(np.sort(order), np.arange(len(a)))
        P = np.diff(rs)
        info = dict(proposals=int(len(a)), rounds=int(len(P)), rounds_pass1=int(len(cut1) - 1), rounds_pass2=int(len(cut2) - 1), solo=int(len(i3)),
                    deferred_pass1=int(len(d)), contained_share_pass1=float(n1 / max(len(a), 1)), P_max=int(P.max()) if len(P) else 0,
                    P_mean=float(P.mean()) if len(P) else 0.0, shift_um=shift.round(3).tolist(), colour_order=np.argsort(phase_pos).tolist())
        return a[order], b[order], u[order], rs, info

    def recompute(self, check_cache=True):
        """Exact recompute of the caches and of the energy from scratch, with the cache and moment drift checks."""
        e, t, L, st = self.eng, self.t, self.lib, self._stream()
        niche, mdiff = [], []
        for r, nbr, m, ec, q, isig in ((self.r[0], t["nbr1"], t["m1"], t["e1"], t["q1"], t["isig1"]),
                                       (self.r[1], t["nbr2"], t["m2"], t["e2"], t["q2"], t["isig2"])):
            _ok(L.ng_env(self.n, self.K, r, nbr.data_ptr(), t["cell_at"].data_ptr(), t["p"].data_ptr(), t["scratch"].data_ptr(), st), "env")
            d = float((t["scratch"] - m).abs().max().item())
            mdiff.append(d)
            if check_cache and e.use[0]:
                assert d <= 1e-9, f"m cache drift {d}"
            m.copy_(t["scratch"])
            _ok(L.ng_site_energy(self.n, self.K, t["cell_at"].data_ptr(), m.data_ptr(), q.data_ptr(), isig.data_ptr(), e.delta, ec.data_ptr(), st),
                "site_energy")
            niche.append(float(ec.cpu().numpy().sum()))
        cell = t["cell_at"].cpu().numpy()
        SXc, SXXc, Mc = NA.moments_state(cell, e.p, e.S)
        SXg, SXXg = t["SX"].cpu().numpy(), t["SXX"].cpu().numpy()
        if e.use[2]:
            assert np.allclose(SXc, SXg, rtol=1e-9, atol=1e-6) and np.allclose(SXXc, SXXg, rtol=1e-9, atol=1e-3), "moment drift"
        t["SX"].copy_(torch.from_numpy(np.ascontiguousarray(SXc)))
        t["SXX"].copy_(torch.from_numpy(np.ascontiguousarray(SXXc)))
        E = dict(niche=niche if e.use[0] else [],
                 unary=float(((e.S - e.G[cell]) ** 2).sum() * e.inv2tau2) if e.use[1] else 0.0,
                 moments=NA.moments_penalty(SXc, SXXc, Mc, e.mu0, e.s0) if e.use[2] else 0.0)
        E["total"] = float(sum(E["niche"]) + E["unary"] + E["moments"])
        self.cell_at = cell
        return E, mdiff, dict(SX_maxdiff=float(np.abs(SXc - SXg).max()), SXX_maxdiff=float(np.abs(SXXc - SXXg).max()))

    def run(self, n_sweeps=8, seed=20261019, on_sweep=None):
        """Run n_sweeps sweeps; sweep sw draws its randomness from default_rng([seed, sw]). Returns the per-sweep record."""
        E0, md0, _ = self.recompute()
        traj = [dict(sweep=0, **E0)]
        self.log(f"  sweep 0: U {E0['total']:.3f} (niche {[round(x, 3) for x in E0['niche']]}, unary {E0['unary']:.3f}, moments "
                 f"{E0['moments']:.3f}); start caches vs a GPU recompute: max |dm| {max(md0):.1e}")
        U_run = E0["total"]
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(self.schedule, np.random.default_rng([seed, 1])) if n_sweeps >= 1 else None
        for sw in range(1, n_sweeps + 1):
            t0 = time.time()
            a, b, u, rs, sinfo = fut.result()
            if sw < n_sweeps:
                fut = pool.submit(self.schedule, np.random.default_rng([seed, sw + 1]))
            t1 = time.time()
            if len(a) > self.n_props_cap:
                self._alloc_props(len(a))
            if sinfo["P_max"] > self.Pmax_cap:
                self._alloc_ev(sinfo["P_max"])
            D = self.dev
            ta, tb, tu = (torch.from_numpy(a).to(D), torch.from_numpy(b).to(D), torch.from_numpy(np.ascontiguousarray(u)).to(D))
            self.t["terms"].zero_()
            self.t["counters"].zero_()
            check = int(self.check == "all" or (self.check == "first" and sw == 1))
            rs_c = np.ascontiguousarray(rs, dtype=np.int64)
            _ok(self.lib.ng_sweep(ctypes.byref(self.state()), ta.data_ptr(), tb.data_ptr(), tu.data_ptr(), rs_c.ctypes.data, len(rs_c) - 1,
                                  self.stamp, self.round_id, check, self.threads, self._stream()), "sweep launch")
            _ok(self.lib.ng_sync(), "sweep")
            t2 = time.time()
            self.stamp += int(2 * (rs_c[-1] - rs_c[0]) + 2 * (len(rs_c) - 1))
            self.round_id += len(rs_c) - 1
            terms = self.t["terms"].cpu().numpy()
            cnt = self.t["counters"].cpu().numpy()
            assert cnt[3] == 0, f"sweep {sw}: {cnt[3]} footprint conflicts (independence violated)"
            U_run += terms[0] + terms[1] + terms[2] + terms[3]
            E, mdiff, mom = self.recompute()
            drift = U_run - E["total"]
            assert abs(drift) <= 1e-6 * max(1.0, abs(E["total"])), f"energy drift {drift}"
            U_run = E["total"]
            disp = np.linalg.norm(self.eng.S[self.site_of()] - self.eng.S[self.orig_site], axis=1)
            moved = disp > 0
            n_prop = sinfo["proposals"]
            rec = dict(sweep=sw, proposals=n_prop, cap_rejected=int(cnt[0]), accepted=int(cnt[1]), accepted_uphill=int(cnt[2]),
                       uphill_dU_sum=float(terms[4]), d_niche=[float(terms[0]), float(terms[1])], d_unary=float(terms[2]), d_moments=float(terms[3]),
                       drift=float(drift), cells_displaced=int(moved.sum()),
                       disp_um=dict(median_moved=float(np.median(disp[moved])) if moved.any() else 0.0,
                                    p90_moved=float(np.percentile(disp[moved], 90)) if moved.any() else 0.0, max=float(disp.max())),
                       seconds=round(time.time() - t0, 2), **E,
                       gpu=dict(sinfo, seconds_schedule_wait=round(t1 - t0, 2), seconds_kernels=round(t2 - t1, 2), seconds_recompute=round(time.time() - t2, 2),
                                claim_check=bool(check), footprint_conflicts=int(cnt[3]), m_cache_maxdiff=mdiff, **mom))
            traj.append(rec)
            self.log(f"  sweep {sw}: proposals {n_prop:,}, cap-rejected {cnt[0]:,}, accepted {cnt[1]:,} ({cnt[2]:,} uphill); U {E['total']:.3f} "
                     f"(niche {[round(x, 3) for x in E['niche']]}, unary {E['unary']:.3f}, moments {E['moments']:.3f}); drift {drift:.2e}; "
                     f"displaced {moved.sum():,} cells, median {rec['disp_um']['median_moved']:.1f} um, p90 {rec['disp_um']['p90_moved']:.1f}, "
                     f"max {disp.max():.1f}; {rec['seconds']} s (schedule wait {t1 - t0:.1f}, kernels {t2 - t1:.1f}); rounds {sinfo['rounds']:,} "
                     f"(P mean {sinfo['P_mean']:.0f}, max {sinfo['P_max']}), deferred {sinfo['deferred_pass1']:,} -> solo {sinfo['solo']:,}"
                     f"{'; claim check: 0 conflicts' if check else ''}")
            if on_sweep is not None:
                on_sweep(sw, self)
        pool.shutdown()
        self.sync_to_cpu()
        return traj

    def sync_to_cpu(self):
        """Copy the GPU state (assignment, caches, moment sums) into the mirrored CPU engine."""
        e, t = self.eng, self.t
        e.cell_at[:] = t["cell_at"].cpu().numpy()
        for sc, mk, ek in zip(e.scales, ("m1", "m2"), ("e1", "e2")):
            sc["m"][:] = t[mk].cpu().numpy()
            sc["e"][:] = t[ek].cpu().numpy()
        e.SX[:] = t["SX"].cpu().numpy()
        e.SXX[:] = t["SXX"].cpu().numpy()
        self.cell_at = e.cell_at.copy()

    def site_of(self):
        """site_of[c] = the site holding cell c."""
        s = np.empty(self.n, np.int64)
        s[self.cell_at] = np.arange(self.n)
        return s

    def output_layout(self, P_sites):
        """Row c = the position (from P_sites, the sample's rows) of the site now holding cell c."""
        return np.asarray(P_sites)[self.site_of()]
