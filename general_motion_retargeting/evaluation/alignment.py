"""Time alignment: resampling and DTW."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def resample_trajectory(
    features: np.ndarray,
    src_fps: float,
    tgt_fps: float,
) -> np.ndarray:
    """
    Linearly resample (T, D) along time to new length ceil(T * tgt_fps / src_fps).

    Uses uniform sampling in source time.
    """
    if src_fps <= 0 or tgt_fps <= 0:
        raise ValueError("fps must be positive")
    features = np.asarray(features, dtype=np.float64)
    t_src = features.shape[0]
    if t_src < 2:
        return features.copy()
    t_new = max(2, int(np.ceil(t_src * tgt_fps / src_fps)))
    x_old = np.linspace(0.0, 1.0, num=t_src)
    x_new = np.linspace(0.0, 1.0, num=t_new)
    out = np.stack([np.interp(x_new, x_old, features[:, j]) for j in range(features.shape[1])], axis=1)
    return out.astype(np.float64)


def resample_full_trajectory(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
    src_fps: float,
    tgt_fps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample root_pos, root_rot, dof_pos together (same time base)."""
    t_src = root_pos.shape[0]
    t_new = max(2, int(np.ceil(t_src * tgt_fps / src_fps)))
    x_old = np.linspace(0.0, 1.0, num=t_src)
    x_new = np.linspace(0.0, 1.0, num=t_new)

    rp = np.stack([np.interp(x_new, x_old, root_pos[:, j]) for j in range(3)], axis=1)
    rr = np.stack([np.interp(x_new, x_old, root_rot[:, j]) for j in range(4)], axis=1)
    rr = rr / (np.linalg.norm(rr, axis=1, keepdims=True) + 1e-12)
    dq = np.stack([np.interp(x_new, x_old, dof_pos[:, j]) for j in range(dof_pos.shape[1])], axis=1)
    return rp, rr, dq


def dtw_distance(
    a: np.ndarray,
    b: np.ndarray,
    window: Optional[int] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    DTW with Euclidean local cost on rows of a (Ta, D) and b (Tb, D).

    Returns (total_cost, path_a_indices, path_b_indices) with same length K.
    window: if set, only cells with abs((i-1)-(j-1)) <= window when filling dp[i,j].
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"a,b must be (Ta,D),(Tb,D) with same D, got {a.shape}, {b.shape}")
    ta, tb = a.shape[0], b.shape[0]
    inf = 1e30
    dp = np.full((ta + 1, tb + 1), inf, dtype=np.float64)
    ptr_i = np.zeros((ta + 1, tb + 1), dtype=np.int32)
    ptr_j = np.zeros((ta + 1, tb + 1), dtype=np.int32)
    dp[0, 0] = 0.0

    def allowed(i: int, j: int) -> bool:
        if window is None:
            return True
        return abs((i - 1) - (j - 1)) <= window

    for i in range(1, ta + 1):
        for j in range(1, tb + 1):
            if not allowed(i, j):
                continue
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            opts = [
                (dp[i - 1, j], i - 1, j),
                (dp[i, j - 1], i, j - 1),
                (dp[i - 1, j - 1], i - 1, j - 1),
            ]
            best, pi, pj = min(opts, key=lambda x: x[0])
            if best >= inf / 2:
                continue
            dp[i, j] = cost + best
            ptr_i[i, j] = pi
            ptr_j[i, j] = pj

    if dp[ta, tb] >= inf / 2:
        raise RuntimeError("DTW: no valid path (try larger window or check dimensions)")

    pa_rev, pb_rev = [], []
    i, j = ta, tb
    while i > 0 and j > 0:
        pa_rev.append(i - 1)
        pb_rev.append(j - 1)
        ni, nj = int(ptr_i[i, j]), int(ptr_j[i, j])
        if ni == i and nj == j:
            break
        i, j = ni, nj

    pa = np.array(pa_rev[::-1], dtype=np.int32)
    pb = np.array(pb_rev[::-1], dtype=np.int32)
    return float(dp[ta, tb]), pa, pb


def warp_sequence_along_path(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
    path_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index trajectory B by DTW path indices path_b."""
    idx = np.asarray(path_b, dtype=int)
    return root_pos[idx].copy(), root_rot[idx].copy(), dof_pos[idx].copy()
