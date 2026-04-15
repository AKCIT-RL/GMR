"""Joint-space metrics: RMSE/MAE, jerk, joint limit violations."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def q_position_errors(q_a: np.ndarray, q_b: np.ndarray) -> Dict[str, Any]:
    """Per-joint MAE/RMSE and global RMSE for dof sequences (same T)."""
    diff = q_a - q_b
    mae = np.mean(np.abs(diff), axis=0)
    rmse_j = np.sqrt(np.mean(diff**2, axis=0))
    rmse_global = float(np.sqrt(np.mean(diff**2)))
    return {
        "mae_per_joint": mae.tolist(),
        "rmse_per_joint": rmse_j.tolist(),
        "mae_mean": float(np.mean(mae)),
        "rmse_global": rmse_global,
    }


def jerk_metrics(q: np.ndarray, fps: float) -> Dict[str, Any]:
    """Third time derivative of joint angles (L2 norm per frame) via np.gradient."""
    dt = 1.0 / fps
    if q.shape[0] < 4:
        return {"jerk_l2_mean": 0.0, "jerk_l2_p95": 0.0, "jerk_l2_max": 0.0}
    d1 = np.gradient(q, dt, axis=0, edge_order=2)
    d2 = np.gradient(d1, dt, axis=0, edge_order=2)
    j3 = np.gradient(d2, dt, axis=0, edge_order=2)
    jerk = np.linalg.norm(j3, axis=1)
    return {
        "jerk_l2_mean": float(np.mean(jerk)),
        "jerk_l2_p95": float(np.percentile(jerk, 95)),
        "jerk_l2_max": float(np.max(jerk)),
    }


def joint_limit_violations(
    dof_pos: np.ndarray,
    xml_path: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Count frames with any joint outside XML range (KinematicsModel limits)."""
    import torch

    from general_motion_retargeting.kinematics_model import KinematicsModel

    km = KinematicsModel(xml_path, torch.device(device))
    lo, hi = km.get_dof_limits()
    lo = lo.cpu().numpy()
    hi = hi.cpu().numpy()
    d = dof_pos.shape[1]
    if d != lo.shape[0]:
        raise ValueError(f"dof_pos width {d} != model num_dof {lo.shape[0]} for {xml_path}")
    below = dof_pos < lo[None, :]
    above = dof_pos > hi[None, :]
    viol = below | above
    frames_any = np.any(viol, axis=1)
    return {
        "violation_rate": float(np.mean(frames_any)),
        "num_frames": int(dof_pos.shape[0]),
        "num_violation_frames": int(np.sum(frames_any)),
        "max_violation_magnitude": float(np.max(np.maximum(lo - dof_pos, dof_pos - hi).clip(min=0.0))),
    }


def compare_jerk(q_a: np.ndarray, q_b: np.ndarray, fps: float) -> Dict[str, Any]:
    ja = jerk_metrics(q_a, fps)
    jb = jerk_metrics(q_b, fps)
    return {"mocap": ja, "genmo": jb, "abs_diff_mean": abs(ja["jerk_l2_mean"] - jb["jerk_l2_mean"])}
