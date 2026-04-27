"""Orientation normalization: canonicalize the initial heading before FK comparison."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def canonicalize_heading(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    center_xy: bool = True,
    center_z: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove the initial yaw and origin offset so trajectories are comparable.

    Steps:
      1. Extract yaw from frame 0 root_rot and rotate all frames so the
         robot starts facing +X.
      2. If center_xy: subtract frame 0 XY so the trajectory starts at the origin.
      3. If center_z: subtract frame 0 Z so both trajectories start at the same
         height, removing the absolute Z offset (and partially the vertical drift
         introduced by monocular depth estimation in video-based pipelines).

    Args:
        root_pos: (T, 3) world positions.
        root_rot_xyzw: (T, 4) quaternions in scipy/xyzw (scalar-last) convention.
        center_xy: translate XY so frame 0 is at (0, 0).
        center_z: translate Z so frame 0 is at z=0.

    Returns:
        new_pos (T, 3), new_rot_xyzw (T, 4) — both normalized.
    """
    r0 = R.from_quat(root_rot_xyzw[0])
    fwd = r0.apply([1.0, 0.0, 0.0])
    fwd[2] = 0.0
    norm = np.linalg.norm(fwd)
    if norm < 1e-6:
        return root_pos.copy(), root_rot_xyzw.copy()
    fwd /= norm
    yaw = float(np.arctan2(fwd[1], fwd[0]))
    inv_yaw = R.from_euler("z", -yaw)

    pos = root_pos.copy()
    if center_xy:
        pos[:, 0] -= root_pos[0, 0]
        pos[:, 1] -= root_pos[0, 1]
    if center_z:
        pos[:, 2] -= root_pos[0, 2]
    new_pos = inv_yaw.apply(pos)

    rots = R.from_quat(root_rot_xyzw)
    new_rot = (inv_yaw * rots).as_quat()
    return new_pos.astype(np.float64), new_rot.astype(np.float64)
