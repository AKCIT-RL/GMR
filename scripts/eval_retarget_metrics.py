#!/usr/bin/env python3
"""CLI: compare two GMR trajectory pickles (e.g. MoCap vs GENMO) and write JSON metrics."""

import argparse
import pathlib

from general_motion_retargeting.evaluation.pipeline import EvalConfig, run_evaluation
from general_motion_retargeting.evaluation.report import save_report
from general_motion_retargeting.params import ROBOT_XML_DICT


def main() -> None:
    robots = sorted(ROBOT_XML_DICT.keys())
    p = argparse.ArgumentParser(description="GMR retargeting metrics (MoCap vs GENMO .pkl pair)")
    p.add_argument("--mocap_pkl", type=str, required=True)
    p.add_argument("--genmo_pkl", type=str, required=True)
    p.add_argument("--robot", type=str, required=True, choices=robots)
    p.add_argument("--xml_path", type=str, default=None, help="Override MuJoCo XML path")
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument(
        "--mocap_root_rot_convention",
        type=str,
        default="wxyz",
        choices=("xyzw", "wxyz"),
        help="Quaternion order in mocap pickle root_rot (xsens legacy: wxyz).",
    )
    p.add_argument(
        "--genmo_root_rot_convention",
        type=str,
        default="xyzw",
        choices=("xyzw", "wxyz"),
        help="Quaternion order in genmo pickle root_rot (gvhmr/bvh export: xyzw).",
    )
    p.add_argument("--target_fps", type=float, default=None, help="Resample both to this fps (default: min of inputs)")
    p.add_argument("--dtw_window", type=int, default=None, help="Sakoe–Chiba window on frame indices (optional)")
    p.add_argument("--no_fk", action="store_true", help="Skip FK / MPJPE-like / EE metrics")
    p.add_argument("--no_normalize_orientation", action="store_true", help="Disable initial-yaw normalization before FK (not recommended for cross-source comparisons)")
    p.add_argument("--mujoco", action="store_true", help="Run MuJoCo self-collision proxy")
    p.add_argument(
        "--ee_bodies",
        type=str,
        default=None,
        help="Comma-separated MuJoCo body names for EE RMSE (default: per-robot built-in list)",
    )
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    ee = None
    if args.ee_bodies:
        ee = [s.strip() for s in args.ee_bodies.split(",") if s.strip()]

    cfg = EvalConfig(
        mocap_pkl=str(pathlib.Path(args.mocap_pkl).resolve()),
        genmo_pkl=str(pathlib.Path(args.genmo_pkl).resolve()),
        robot=args.robot,
        xml_path=str(pathlib.Path(args.xml_path).resolve()) if args.xml_path else None,
        mocap_root_rot_convention=args.mocap_root_rot_convention,
        genmo_root_rot_convention=args.genmo_root_rot_convention,
        target_fps=args.target_fps,
        dtw_window=args.dtw_window,
        use_fk=not args.no_fk,
        normalize_orientation=not args.no_normalize_orientation,
        use_mujoco=args.mujoco,
        ee_body_names=ee,
        device=args.device,
    )
    report = run_evaluation(cfg)
    save_report(args.out_json, report)
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
