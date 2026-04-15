"""Task-space metrics via KinematicsModel.forward_kinematics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from general_motion_retargeting.kinematics_model import KinematicsModel


def _batch_fk(
    km: KinematicsModel,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns body_pos (T, J, 3), body_rot (T, J, 4) as numpy."""
    rp = torch.tensor(root_pos, dtype=torch.float32, device=device)
    rr = torch.tensor(root_rot_xyzw, dtype=torch.float32, device=device)
    dq = torch.tensor(dof_pos, dtype=torch.float32, device=device)
    bp, br = km.forward_kinematics(rp, rr, dq)
    return bp.detach().cpu().numpy(), br.detach().cpu().numpy()


def mean_body_position_error(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    body_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Mean over time of average L2 error over selected bodies (MPJPE-like on robot links).

    pos_* shape (T, J, 3).
    """
    if body_indices is None:
        body_indices = list(range(pos_a.shape[1]))
    idx = np.asarray(body_indices, dtype=int)
    d = np.linalg.norm(pos_a[:, idx, :] - pos_b[:, idx, :], axis=-1)
    per_frame = np.mean(d, axis=1)
    return {
        "mean_mpjpe_like_m": float(np.mean(per_frame)),
        "p95_mpjpe_like_m": float(np.percentile(per_frame, 95)),
        "per_body_mean_m": np.mean(d, axis=0).tolist(),
        "body_indices_used": idx.tolist(),
    }


def end_effector_rmse(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    body_indices: Sequence[int],
) -> Dict[str, Any]:
    """Per-EE RMSE (m) averaged over time."""
    idx = np.asarray(body_indices, dtype=int)
    errs = []
    for b in idx:
        d = np.linalg.norm(pos_a[:, b, :] - pos_b[:, b, :], axis=-1)
        errs.append(float(np.sqrt(np.mean(d**2))))
    return {"ee_body_indices": idx.tolist(), "rmse_per_ee_m": errs, "mean_rmse_m": float(np.mean(errs))}


def bone_cosine_similarity(
    body_pos: np.ndarray,
    parent_indices: np.ndarray,
    exclude_root: bool = True,
) -> np.ndarray:
    """
    Per-frame mean cosine between bone direction vectors (child - parent) vs another sequence.

    This function returns bone unit vectors for one sequence: (T, num_bones, 3).
    num_bones = J - 1 skipping root if exclude_root.
    """
    j = body_pos.shape[1]
    vecs = []
    for child in range(1, j):
        p = int(parent_indices[child])
        v = body_pos[:, child, :] - body_pos[:, p, :]
        n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-10
        vecs.append(v / n)
    if not vecs:
        return np.zeros((body_pos.shape[0], 0, 3))
    return np.stack(vecs, axis=1)


def mean_bone_cosine_between(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    parent_indices: np.ndarray,
) -> Dict[str, Any]:
    va = bone_cosine_similarity(pos_a, parent_indices)
    vb = bone_cosine_similarity(pos_b, parent_indices)
    if va.shape[1] == 0:
        return {"mean_cosine": 1.0}
    c = np.sum(va * vb, axis=-1)
    c = np.clip(c, -1.0, 1.0)
    return {
        "mean_cosine": float(np.mean(c)),
        "p05_cosine": float(np.percentile(c, 5)),
    }


def fk_full_metrics(
    xml_path: str,
    root_pos_a: np.ndarray,
    root_rot_a: np.ndarray,
    dof_a: np.ndarray,
    root_pos_b: np.ndarray,
    root_rot_b: np.ndarray,
    dof_b: np.ndarray,
    ee_body_names: Optional[List[str]] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    dev = torch.device(device)
    km = KinematicsModel(xml_path, dev)
    pa, _ = _batch_fk(km, root_pos_a, root_rot_a, dof_a, dev)
    pb, _ = _batch_fk(km, root_pos_b, root_rot_b, dof_b, dev)
    parent = km.parent_indices.cpu().numpy()

    all_idx = list(range(len(km.body_names)))
    mpjpe = mean_body_position_error(pa, pb, all_idx)

    out: Dict[str, Any] = {"mpjpe_like": mpjpe, "bone_cosine": mean_bone_cosine_between(pa, pb, parent)}

    if ee_body_names:
        idx = [km.get_body_idx(n) for n in ee_body_names]
        out["end_effector"] = end_effector_rmse(pa, pb, idx)
    return out
