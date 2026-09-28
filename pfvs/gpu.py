# =============================================================================
#    Copyright (C) 2026  Liam McAllister Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Optional CUDA backend for the batched coni pipeline (lattice
#               setup + search per p-vector), see fp_kernel/cuda/pfvs_gpu.cu.
#               Built only with PFVS_CUDA=1 (needs nvcc); loaded via ctypes.
#               Results are exact and identical to the CPU path's.
# -----------------------------------------------------------------------------

import ctypes
import glob
import math
import os

import numpy as np

_i64p = ctypes.POINTER(ctypes.c_int64)
_i32p = ctypes.POINTER(ctypes.c_int32)


class _Input(ctypes.Structure):
    _fields_ = [("n_geo", ctypes.c_int), ("h", _i32p),
                ("Q", _i64p), ("qmax", _i64p), ("linmin", _i64p),
                ("kappa", _i64p), ("Mbasis", _i64p),
                ("n_p", ctypes.c_int64), ("ps", _i64p), ("pgeo", _i32p),
                ("device", ctypes.c_int), ("batch", ctypes.c_int64),
                ("verbose", ctypes.c_int)]


class _Output(ctypes.Structure):
    _fields_ = [("n", ctypes.c_int64), ("M", _i64p), ("Kn", _i64p),
                ("q", _i64p), ("pidx", _i64p),
                ("pstat", ctypes.POINTER(ctypes.c_int8)),
                ("seconds_gpu", ctypes.c_double), ("err", ctypes.c_char * 256)]


_lib = None
_lib_error = None


def _load():
    global _lib, _lib_error
    if _lib is not None or _lib_error is not None:
        return _lib
    path = os.environ.get("PFVS_GPU_LIB")
    if not path:
        here = os.path.join(os.path.dirname(__file__), "fp_kernel")
        found = sorted(glob.glob(os.path.join(here, "libpfvs_gpu*.so")))
        path = found[0] if found else None
    if not path:
        _lib_error = "the CUDA backend was not built (install with PFVS_CUDA=1)"
        return None
    try:
        lib = ctypes.CDLL(path)
    except OSError as e:
        _lib_error = f"could not load {path}: {e}"
        return None
    lib.pfg_max_h11.restype = ctypes.c_int
    lib.pfg_device_count.restype = ctypes.c_int
    lib.pfg_coni_batch.argtypes = [ctypes.POINTER(_Input), ctypes.POINTER(_Output)]
    lib.pfg_coni_batch.restype = ctypes.c_int
    lib.pfg_output_free.argtypes = [ctypes.POINTER(_Output)]
    _lib = lib
    return lib


def available() -> bool:
    """
    **Description:**
    Whether the CUDA backend is built and a CUDA device is present.

    **Returns:**
    *(bool)* True if `coniZpM(..., device="gpu")` can run.
    """
    lib = _load()
    return lib is not None and lib.pfg_device_count() > 0


def unavailable_reason() -> str:
    """Why `available()` is False ('' if it is True)."""
    lib = _load()
    if lib is None:
        return _lib_error
    if lib.pfg_device_count() == 0:
        return "no CUDA device found"
    return ""


def max_h11() -> int:
    """The largest h11 the GPU build supports (compile-time)."""
    lib = _load()
    if lib is None:
        raise RuntimeError(_lib_error)
    return lib.pfg_max_h11()


def _qmax(Q, dilation):
    return int(math.floor(dilation * Q + 1e-7))


def coni_batch_multi(geoms, ps, pgeo, device=0, batch=0, verbose=False):
    """
    **Description:**
    The batched coni pipeline on the GPU for p-vectors of several geometries
    at once (what keeps the device busy when each geometry has few
    p-vectors).

    **Arguments:**
    - `geoms` *(list of dict)*: Per geometry: `kappa` (h, h, h), `Mbasis`
        (h, h), `Q`, `dilation`, `M0min`.
    - `ps` *(list/array of int arrays)*: The p-vectors, each of length h of
        its geometry, *with* the leading 0 (as `_coni_batch` takes them).
    - `pgeo` *(array of int)*: The geometry index of each p-vector.
    - `device` *(int, optional)*: CUDA device ordinal.
    - `batch` *(int, optional)*: p-vectors per device batch (0: default).
    - `verbose` *(bool, optional)*: Per-batch timings on stderr.

    **Returns:**
    *(tuple)* `(M, Kn, q, pidx, pstat, seconds_gpu)`: for every lattice point
    M = Binter c and Kn = Z Binter c (rows of shape (N, max_h11()), first h
    entries valid), q = c^T mat c, and the index of its p-vector; pstat[i] is
    0 if p-vector i was done on the GPU and 1 if it must go through the CPU
    path (none of its points are returned). Points are grouped by p-vector
    (ascending) and sorted by (M, Kn) within one.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError(_lib_error)
    MH = lib.pfg_max_h11()
    n_geo = len(geoms)
    h = np.array([np.asarray(g["kappa"]).shape[0] for g in geoms], dtype=np.int32)
    if n_geo and (h.min() < 3 or h.max() > MH):
        raise ValueError(f"the GPU build supports 3 <= h11 <= {MH}")
    Q = np.array([int(g["Q"]) for g in geoms], dtype=np.int64)
    qmax = np.array([_qmax(int(g["Q"]), g["dilation"]) for g in geoms], dtype=np.int64)
    linmin = np.array([math.ceil(g["M0min"] - 1e-9) for g in geoms], dtype=np.int64)
    kap = np.concatenate([np.ascontiguousarray(g["kappa"], dtype=np.int64).ravel() for g in geoms]) \
        if n_geo else np.zeros(1, dtype=np.int64)
    Mb = np.concatenate([np.ascontiguousarray(g["Mbasis"], dtype=np.int64).ravel() for g in geoms]) \
        if n_geo else np.zeros(1, dtype=np.int64)
    pgeo = np.ascontiguousarray(pgeo, dtype=np.int32)
    n_p = len(pgeo)
    P = np.zeros((max(n_p, 1), MH), dtype=np.int64)
    if isinstance(ps, np.ndarray) and ps.ndim == 2:
        P[:n_p, :ps.shape[1]] = ps
    else:
        for i, p in enumerate(ps):
            P[i, :len(p)] = p
    if n_p and (np.any(pgeo < 0) or np.any(pgeo >= n_geo)):
        raise ValueError("pgeo: geometry index out of range")
    if n_p and np.any(P[:n_p][np.arange(MH)[None, :] >= h[pgeo][:, None]] != 0):
        raise ValueError("a p-vector is longer than its geometry's h11")

    inp = _Input(n_geo, h.ctypes.data_as(_i32p), Q.ctypes.data_as(_i64p),
                 qmax.ctypes.data_as(_i64p), linmin.ctypes.data_as(_i64p),
                 kap.ctypes.data_as(_i64p), Mb.ctypes.data_as(_i64p),
                 n_p, P.ctypes.data_as(_i64p), pgeo.ctypes.data_as(_i32p),
                 int(device), int(batch), int(bool(verbose)))
    out = _Output()
    rc = lib.pfg_coni_batch(ctypes.byref(inp), ctypes.byref(out))
    if rc != 0:
        raise RuntimeError(f"GPU pipeline failed: {out.err.decode(errors='replace')}")
    try:
        n = out.n
        def arr(ptr, shape):
            if n == 0:
                return np.zeros(shape, dtype=np.int64)
            return np.ctypeslib.as_array(ptr, shape=shape).copy()
        M, Kn = arr(out.M, (n, MH)), arr(out.Kn, (n, MH))
        q, pidx = arr(out.q, (n,)), arr(out.pidx, (n,))
        pstat = np.ctypeslib.as_array(out.pstat, shape=(n_p,)).copy() if n_p \
            else np.zeros(0, dtype=np.int8)
        secs = out.seconds_gpu
    finally:
        lib.pfg_output_free(ctypes.byref(out))
    return M, Kn, q, pidx, pstat.astype(np.int8), secs


def coni_batch(kappa, Mbasis, ps, Q, dilation, M0min, max_N_out, device=0):
    """
    GPU counterpart of `fp_kernel._coni_batch` for one geometry (the default
    lattice options: extra LLL, cut-aware basis): returns (M, Kn, q, pidx,
    pstat) with the same meaning -- pstat 1 marks p-vectors for the per-p
    CPU path (including any with more than max_N_out lattice points, so that
    path reports it as the CPU kernel would).
    """
    kappa = np.asarray(kappa)
    h = kappa.shape[0]
    ps = np.asarray(ps, dtype=np.int64)
    geoms = [dict(kappa=kappa, Mbasis=Mbasis, Q=Q, dilation=dilation, M0min=M0min)]
    M, Kn, q, pidx, pstat, _ = coni_batch_multi(geoms, ps, np.zeros(len(ps), dtype=np.int32),
                                                device=device)
    counts = np.bincount(pidx, minlength=len(ps)) if len(pidx) else np.zeros(len(ps), dtype=np.int64)
    over = counts > max_N_out
    if np.any(over):
        pstat[over] = 1
        keep = ~over[pidx]
        M, Kn, q, pidx = M[keep], Kn[keep], q[keep], pidx[keep]
    return M[:, :h], Kn[:, :h], q, pidx, pstat
