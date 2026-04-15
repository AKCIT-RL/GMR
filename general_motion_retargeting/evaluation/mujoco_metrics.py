"""MuJoCo-based metrics: self-collision proxy via contact distances.

Torque / accurate dynamics from ``mj_inverse`` are environment-dependent (contacts,
timestep); not computed here by default — see module docstring in evaluation README.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

import mujoco as mj

from general_motion_retargeting.evaluation.io import quat_to_wxyz


def _body_name(model: mj.MjModel, bid: int) -> str:
    if bid < 0:
        return ""
    n = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
    return n or ""


def self_collision_rate(
    xml_path: str,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    penetration_dist: float = -1e-4,
    exclude_body_substrings: Sequence[str] = ("floor", "ground", "plane", "world"),
) -> Dict[str, Any]:
    """
    Fraction of frames where MuJoCo reports a penetrating contact between two
    distinct non-excluded bodies (typical self-collision proxy).

    Quaternion in ``root_rot_xyzw`` is converted to MuJoCo scalar-first (wxyz)
    for ``qpos[3:7]``.
    """
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    nq = model.nq
    if nq < 7 + dof_pos.shape[1]:
        raise ValueError(
            f"Model nq={nq} smaller than 7 + dof_width={dof_pos.shape[1]} — check XML vs trajectory."
        )
    if root_pos.shape[0] != dof_pos.shape[0]:
        raise ValueError("root_pos and dof_pos length mismatch")

    ex = tuple(s.lower() for s in exclude_body_substrings)
    n_frames = root_pos.shape[0]
    bad = 0

    for t in range(n_frames):
        data.qpos[:] = 0.0
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = quat_to_wxyz(root_rot_xyzw[t])
        data.qpos[7 : 7 + dof_pos.shape[1]] = dof_pos[t]
        mj.mj_forward(model, data)

        frame_pen = False
        for c in range(data.ncon):
            con = data.contact[c]
            b1 = model.geom_bodyid[con.geom1]
            b2 = model.geom_bodyid[con.geom2]
            n1 = _body_name(model, b1).lower()
            n2 = _body_name(model, b2).lower()
            if any(s in n1 for s in ex) or any(s in n2 for s in ex):
                continue
            if b1 <= 0 or b2 <= 0:
                continue
            if b1 == b2:
                continue
            if con.dist < penetration_dist:
                frame_pen = True
        if frame_pen:
            bad += 1

    return {
        "self_collision_frame_rate": float(bad / max(1, n_frames)),
        "num_frames": int(n_frames),
        "num_frames_with_penetration": int(bad),
        "note": "Counts penetrating contacts (dist < threshold) between distinct bodies; tune exclude list for scenes with floor.",
    }


def self_collision_rates_pair(
    xml_path: str,
    traj_a: Tuple[np.ndarray, np.ndarray, np.ndarray],
    traj_b: Tuple[np.ndarray, np.ndarray, np.ndarray],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run self_collision_rate on two trajectories (root_pos, root_rot_xyzw, dof)."""
    ra, rqa, da = traj_a
    rb, rqb, db = traj_b
    return {
        "mocap": self_collision_rate(xml_path, ra, rqa, da, **kwargs),
        "genmo": self_collision_rate(xml_path, rb, rqb, db, **kwargs),
    }
