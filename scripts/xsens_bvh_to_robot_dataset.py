#!/usr/bin/env python3
"""Headless batch retarget from Xsens BVH files to a robot.

Counterpart to ``smplx_to_robot_dataset.py`` / ``gvhmr_to_robot_dataset.py`` but
for Xsens BVH motion capture. Walks ``--src_folder`` for ``*.bvh`` files and
writes one ``.pkl`` per clip in the canonical motion_data format used by the
evaluation tooling.

Note: GMR only ships Xsens IK configs for ``unitree_g1`` and ``unitree_h1_2``.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle
from types import SimpleNamespace

import numpy as np
from natsort import natsorted

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.xsens import load_xsens_file

HERE = pathlib.Path(__file__).parent


def process_file(bvh_path: str, out_path: str, robot: str, scale: float = 0.01) -> bool:
    args = SimpleNamespace(
        bvh_file=bvh_path,
        scale=scale,
        reset_to_zero=False,
        start=None,
        end=None,
        bvh_format="3DSM",
    )
    try:
        frames, height, frame_time = load_xsens_file(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[load][ERR] {bvh_path}: {exc}")
        return False

    fps = int(round(1.0 / frame_time))
    retarget = GMR(src_human="bvh_xsens", tgt_robot=robot, actual_human_height=height)
    qpos_list = []
    for fr in frames:
        qpos_list.append(retarget.retarget(fr).copy())
    qpos = np.array(qpos_list)
    if qpos.size == 0:
        print(f"[empty] {bvh_path}")
        return False

    root_pos = qpos[:, :3]
    # Match smplx_to_robot_dataset.py convention: store root_rot as xyzw
    # (MuJoCo qpos[3:7] is wxyz, so reorder).
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
    print(f"[ok] {bvh_path} -> {out_path}")
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--robot", required=True)
    p.add_argument("--src_folder", required=True, help="Folder with Xsens .bvh files")
    p.add_argument("--tgt_folder", required=True)
    p.add_argument("--override", action="store_true")
    p.add_argument("--scale", type=float, default=0.01)
    args = p.parse_args()

    src = pathlib.Path(args.src_folder).resolve()
    tgt = pathlib.Path(args.tgt_folder).resolve()
    tgt.mkdir(parents=True, exist_ok=True)

    bvhs = []
    for dirpath, _, filenames in os.walk(src):
        for fn in natsorted(filenames):
            if fn.endswith(".bvh"):
                bvhs.append(pathlib.Path(dirpath) / fn)
    if not bvhs:
        print(f"[warn] no .bvh files under {src}")
        return

    n_ok = 0
    for bvh in bvhs:
        rel = bvh.relative_to(src)
        out = tgt / rel.with_suffix(".pkl")
        if out.exists() and not args.override:
            print(f"[skip] {out} exists")
            continue
        if process_file(str(bvh), str(out), args.robot, scale=args.scale):
            n_ok += 1
    print(f"[done] {n_ok}/{len(bvhs)} clips retargeted to {args.robot}")


if __name__ == "__main__":
    main()
