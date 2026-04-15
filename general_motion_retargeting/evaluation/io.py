"""Load and validate GMR motion pickle files."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

QuatConvention = Literal["xyzw", "wxyz"]


@dataclass
class Trajectory:
    """Robot trajectory after GMR (matches scripts/*_to_robot.py motion_data)."""

    root_pos: np.ndarray  # (T, 3)
    root_rot: np.ndarray  # (T, 4) quaternion, internal convention xyzw for FK
    dof_pos: np.ndarray  # (T, D)
    fps: float
    source_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.root_pos = np.asarray(self.root_pos, dtype=np.float64)
        self.root_rot = np.asarray(self.root_rot, dtype=np.float64)
        self.dof_pos = np.asarray(self.dof_pos, dtype=np.float64)
        if self.root_pos.ndim != 2 or self.root_pos.shape[1] != 3:
            raise ValueError(f"root_pos must be (T,3), got {self.root_pos.shape}")
        if self.root_rot.ndim != 2 or self.root_rot.shape[1] != 4:
            raise ValueError(f"root_rot must be (T,4), got {self.root_rot.shape}")
        if self.dof_pos.ndim != 2:
            raise ValueError(f"dof_pos must be (T,D), got {self.dof_pos.shape}")
        t = self.root_pos.shape[0]
        if self.root_rot.shape[0] != t or self.dof_pos.shape[0] != t:
            raise ValueError("root_pos, root_rot, dof_pos must have same T")


def quat_to_xyzw(q: np.ndarray, convention: QuatConvention) -> np.ndarray:
    """Convert quaternion to xyzw layout (vector part first, scalar last)."""
    q = np.asarray(q, dtype=np.float64)
    if convention == "xyzw":
        return q.copy()
    # wxyz -> xyzw
    return q[..., [1, 2, 3, 0]].copy()


def quat_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """xyzw -> wxyz for MuJoCo qpos[3:7]."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    return q[..., [3, 0, 1, 2]].copy()


def load_motion_pkl(
    path: str,
    root_rot_convention: QuatConvention = "xyzw",
) -> Trajectory:
    """
    Load motion_data pickle written by GMR scripts.

    Parameters
    ----------
    path
        Path to .pkl file.
    root_rot_convention
        Quaternion order stored in ``root_rot``. Use ``xyzw`` for outputs from
        ``gvhmr_to_robot.py`` / ``bvh_to_robot.py``. Use ``wxyz`` for legacy
        ``xsens_bvh_to_robot.py`` saves (raw MuJoCo qpos slice).
    """
    with open(path, "rb") as f:
        data: dict[str, Any] = pickle.load(f)

    required = ("root_pos", "root_rot", "dof_pos", "fps")
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key {k!r} in {path}")

    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot = quat_to_xyzw(np.asarray(data["root_rot"], dtype=np.float64), root_rot_convention)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    fps = float(data["fps"])

    return Trajectory(
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
        fps=fps,
        source_path=path,
    )


def assert_compatible_pair(a: Trajectory, b: Trajectory) -> None:
    if a.dof_pos.shape[1] != b.dof_pos.shape[1]:
        raise ValueError(
            f"dof_pos width mismatch: {a.dof_pos.shape[1]} vs {b.dof_pos.shape[1]} "
            "(expected same robot / same q layout)."
        )
