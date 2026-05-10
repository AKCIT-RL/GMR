"""Per-robot single-trajectory metrics for cross-embodiment retargeting evaluation (RQ2).

These metrics are computed from a single GMR motion pickle (no MoCap reference
needed) and quantify the kinematic feasibility / quality of the retargeted motion
for a given robot. They are designed to be comparable across robots with
different morphologies and DoF counts.

Metrics
-------
- success: pickle loaded with no NaN/Inf in dof/root.
- joint_limit_violation_rate: fraction of frames where any joint is outside the
  XML limits.
- joint_limit_margin: per-joint mean margin (rad), normalized by joint range.
- workspace_coverage: per-joint span used / total range, mean across joints.
- joint_jerk: L2 norm of d3q/dt3 — mean / p95 / max.
- joint_velocity: L2 norm of dq/dt — mean / p95 / max (Nm-stress proxy).
- foot_sliding: mean tangential foot velocity while in contact (m/s) — only when
  foot body names are provided.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _has_nan_or_inf(arr: np.ndarray) -> bool:
    return bool(np.any(~np.isfinite(arr)))


def success_metric(root_pos: np.ndarray, root_rot: np.ndarray, dof_pos: np.ndarray) -> Dict[str, Any]:
    nan_root = _has_nan_or_inf(root_pos) or _has_nan_or_inf(root_rot)
    nan_dof = _has_nan_or_inf(dof_pos)
    return {
        "success": bool(not (nan_root or nan_dof)),
        "nan_in_root": bool(nan_root),
        "nan_in_dof": bool(nan_dof),
    }


def joint_limit_metrics(dof_pos: np.ndarray, xml_path: str, device: str = "cpu") -> Dict[str, Any]:
    """Limit violation rate, magnitude, and per-joint normalized margin.

    The margin is normalized so 0 means at the limit and 1 means perfectly
    centered in the joint range. Reported as mean across joints and frames.
    """
    import torch

    from general_motion_retargeting.kinematics_model import KinematicsModel

    km = KinematicsModel(xml_path, torch.device(device))
    lo, hi = km.get_dof_limits()
    lo = lo.cpu().numpy()
    hi = hi.cpu().numpy()
    d = dof_pos.shape[1]
    if d != lo.shape[0]:
        raise ValueError(f"dof_pos width {d} != model num_dof {lo.shape[0]} for {xml_path}")

    rng = np.maximum(hi - lo, 1e-6)
    below = dof_pos < lo[None, :]
    above = dof_pos > hi[None, :]
    viol = below | above
    frames_any = np.any(viol, axis=1)
    over = np.maximum(lo[None, :] - dof_pos, dof_pos - hi[None, :]).clip(min=0.0)

    # Margin to nearest limit, normalized by half-range.
    center = 0.5 * (hi + lo)
    half = 0.5 * rng
    margin_norm = 1.0 - np.abs(dof_pos - center[None, :]) / half[None, :]
    margin_norm = np.clip(margin_norm, -np.inf, 1.0)

    return {
        "violation_rate": float(np.mean(frames_any)),
        "num_frames": int(dof_pos.shape[0]),
        "num_violation_frames": int(np.sum(frames_any)),
        "max_violation_magnitude_rad": float(np.max(over)),
        "mean_violation_magnitude_rad": float(np.mean(over[viol])) if np.any(viol) else 0.0,
        "mean_normalized_margin": float(np.mean(margin_norm)),
        "p05_normalized_margin": float(np.percentile(margin_norm, 5)),
    }


def workspace_coverage(dof_pos: np.ndarray, xml_path: str, device: str = "cpu") -> Dict[str, Any]:
    """Fraction of each joint's range traversed during the motion."""
    import torch

    from general_motion_retargeting.kinematics_model import KinematicsModel

    km = KinematicsModel(xml_path, torch.device(device))
    lo, hi = km.get_dof_limits()
    lo = lo.cpu().numpy()
    hi = hi.cpu().numpy()
    rng = np.maximum(hi - lo, 1e-6)

    used = dof_pos.max(axis=0) - dof_pos.min(axis=0)
    cov = np.clip(used / rng, 0.0, 1.0)
    return {
        "mean_coverage": float(np.mean(cov)),
        "p95_coverage": float(np.percentile(cov, 95)),
        "per_joint_coverage": cov.tolist(),
    }


def joint_velocity_metrics(dof_pos: np.ndarray, fps: float) -> Dict[str, Any]:
    if dof_pos.shape[0] < 2:
        return {"vel_l2_mean": 0.0, "vel_l2_p95": 0.0, "vel_l2_max": 0.0}
    dt = 1.0 / fps
    d1 = np.gradient(dof_pos, dt, axis=0, edge_order=2)
    v = np.linalg.norm(d1, axis=1)
    return {
        "vel_l2_mean_rad_s": float(np.mean(v)),
        "vel_l2_p95_rad_s": float(np.percentile(v, 95)),
        "vel_l2_max_rad_s": float(np.max(v)),
    }


def joint_jerk_metrics(dof_pos: np.ndarray, fps: float) -> Dict[str, Any]:
    if dof_pos.shape[0] < 4:
        return {"jerk_l2_mean": 0.0, "jerk_l2_p95": 0.0, "jerk_l2_max": 0.0}
    dt = 1.0 / fps
    d1 = np.gradient(dof_pos, dt, axis=0, edge_order=2)
    d2 = np.gradient(d1, dt, axis=0, edge_order=2)
    d3 = np.gradient(d2, dt, axis=0, edge_order=2)
    j = np.linalg.norm(d3, axis=1)
    return {
        "jerk_l2_mean_rad_s3": float(np.mean(j)),
        "jerk_l2_p95_rad_s3": float(np.percentile(j, 95)),
        "jerk_l2_max_rad_s3": float(np.max(j)),
    }


def foot_sliding(
    xml_path: str,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    foot_body_names: List[str],
    fps: float,
    contact_height_thr: float = 0.05,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Tangential foot velocity while a foot is below ``contact_height_thr`` (m).

    A simple proxy that does not require an MJCF contact solver. Returns mean
    horizontal speed of foot bodies during low-height frames.
    """
    import torch

    from general_motion_retargeting.kinematics_model import KinematicsModel

    dev = torch.device(device)
    km = KinematicsModel(xml_path, dev)
    rp = torch.tensor(root_pos, dtype=torch.float32, device=dev)
    rr = torch.tensor(root_rot_xyzw, dtype=torch.float32, device=dev)
    dq = torch.tensor(dof_pos, dtype=torch.float32, device=dev)
    bp, _ = km.forward_kinematics(rp, rr, dq)
    bp = bp.detach().cpu().numpy()  # (T, J, 3)

    dt = 1.0 / fps
    out_per_foot = {}
    speeds = []
    for name in foot_body_names:
        try:
            idx = km.get_body_idx(name)
        except (ValueError, KeyError):
            out_per_foot[name] = {"present": False}
            continue
        traj = bp[:, idx, :]
        # Velocity in xy plane.
        vxy = np.gradient(traj[:, :2], dt, axis=0, edge_order=2)
        speed = np.linalg.norm(vxy, axis=1)
        # Contact mask: foot z below threshold above floor (assumes floor ~0).
        z = traj[:, 2]
        contact = z < (z.min() + contact_height_thr)
        in_contact_speed = speed[contact] if np.any(contact) else np.array([0.0])
        out_per_foot[name] = {
            "present": True,
            "mean_horizontal_speed_m_s": float(np.mean(in_contact_speed)),
            "p95_horizontal_speed_m_s": float(np.percentile(in_contact_speed, 95)),
            "contact_fraction": float(np.mean(contact)),
        }
        speeds.append(float(np.mean(in_contact_speed)))
    summary = {
        "per_foot": out_per_foot,
        "mean_in_contact_speed_m_s": float(np.mean(speeds)) if speeds else 0.0,
    }
    return summary


# Per-robot foot body names (best-effort; missing ones are skipped gracefully).
ROBOT_FOOT_BODIES: Dict[str, List[str]] = {
    "unitree_g1": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "unitree_g1_with_hands": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "unitree_h1": ["left_ankle_link", "right_ankle_link"],
    "unitree_h1_2": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "booster_t1": ["left_foot_link", "right_foot_link"],
    "booster_t1_29dof": ["left_foot_link", "right_foot_link"],
    "stanford_toddy": ["left_foot", "right_foot"],
    "fourier_n1": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "engineai_pm01": ["left_ankle_roll", "right_ankle_roll"],
    "kuavo_s45": ["zarm_l_end_effector", "zarm_r_end_effector"],
    "hightorque_hi": ["l_ankle_roll", "r_ankle_roll"],
    "booster_k1": ["left_foot_link", "right_foot_link"],
    "pnd_adam_lite": ["left_ankle_roll", "right_ankle_roll"],
    "tienkung": ["L_foot_Link", "R_foot_Link"],
}


def evaluate_single_trajectory(
    pkl_path: str,
    xml_path: str,
    robot: str,
    root_rot_convention: str = "xyzw",
    device: str = "cpu",
    compute_foot_sliding: bool = True,
) -> Dict[str, Any]:
    """Run all single-trajectory metrics and return a JSON-friendly dict."""
    from general_motion_retargeting.evaluation.io import load_motion_pkl

    traj = load_motion_pkl(pkl_path, root_rot_convention=root_rot_convention)  # type: ignore[arg-type]

    out: Dict[str, Any] = {
        "pkl": pkl_path,
        "robot": robot,
        "xml": xml_path,
        "fps": traj.fps,
        "num_frames": int(traj.dof_pos.shape[0]),
        "num_dof": int(traj.dof_pos.shape[1]),
    }
    out["status"] = success_metric(traj.root_pos, traj.root_rot, traj.dof_pos)
    if not out["status"]["success"]:
        return out

    out["joint_limits"] = joint_limit_metrics(traj.dof_pos, xml_path, device=device)
    out["workspace"] = workspace_coverage(traj.dof_pos, xml_path, device=device)
    out["velocity"] = joint_velocity_metrics(traj.dof_pos, traj.fps)
    out["jerk"] = joint_jerk_metrics(traj.dof_pos, traj.fps)

    if compute_foot_sliding and robot in ROBOT_FOOT_BODIES:
        try:
            out["foot_sliding"] = foot_sliding(
                xml_path,
                traj.root_pos,
                traj.root_rot,
                traj.dof_pos,
                ROBOT_FOOT_BODIES[robot],
                traj.fps,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            out["foot_sliding"] = {"error": str(exc)}
    return out
