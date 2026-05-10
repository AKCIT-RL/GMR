#!/usr/bin/env python3
"""Headless batch retarget from HMR4D/GVHMR ``.pt`` files to a robot.

Counterpart to ``smplx_to_robot_dataset.py`` but for HMR4D outputs. Walks
``--src_folder`` for ``*.pt`` files and writes one ``.pkl`` per clip.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle

import numpy as np
from natsort import natsorted

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import (
    get_gvhmr_data_offline_fast,
    load_gvhmr_pred_file,
)

HERE = pathlib.Path(__file__).parent
SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"


def process_file(pt_path: str, out_path: str, robot: str, tgt_fps: int = 30) -> bool:
    try:
        smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(pt_path, SMPLX_FOLDER)
    except Exception as exc:  # noqa: BLE001
        print(f"[load][ERR] {pt_path}: {exc}")
        return False

    try:
        frames, fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    except Exception as exc:  # noqa: BLE001
        print(f"[align][ERR] {pt_path}: {exc}")
        return False

    retarget = GMR(actual_human_height=height, src_human="smplx", tgt_robot=robot)
    qpos_list = []
    for fr in frames:
        qpos_list.append(retarget.retarget(fr).copy())
    qpos = np.array(qpos_list)
    if qpos.size == 0:
        print(f"[empty] {pt_path}")
        return False

    root_pos = qpos[:, :3]
    # MuJoCo qpos quat is wxyz -> store xyzw to match smplx_to_robot_dataset
    root_rot = qpos[:, 3:7][:, [1, 2, 3, 0]]
    dof_pos = qpos[:, 7:]

    motion = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(motion, f)
    print(f"[ok] {pt_path} -> {out_path}")
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--robot", required=True)
    p.add_argument("--src_folder", required=True, help="Folder with HMR4D *.pt files")
    p.add_argument("--tgt_folder", required=True)
    p.add_argument("--override", action="store_true")
    p.add_argument("--tgt_fps", type=int, default=30)
    args = p.parse_args()

    src = pathlib.Path(args.src_folder).resolve()
    tgt = pathlib.Path(args.tgt_folder).resolve()
    tgt.mkdir(parents=True, exist_ok=True)

    pts = []
    for dirpath, _, filenames in os.walk(src):
        for fn in natsorted(filenames):
            if fn.endswith(".pt"):
                pts.append(pathlib.Path(dirpath) / fn)
    if not pts:
        print(f"[warn] no .pt files under {src}")
        return

    n_ok = 0
    for pt in pts:
        rel = pt.relative_to(src)
        out = tgt / rel.with_suffix(".pkl")
        if out.exists() and not args.override:
            print(f"[skip] {out} exists")
            continue
        if process_file(str(pt), str(out), args.robot, tgt_fps=args.tgt_fps):
            n_ok += 1
    print(f"[done] {n_ok}/{len(pts)} clips retargeted to {args.robot}")


if __name__ == "__main__":
    main()
