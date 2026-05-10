#!/usr/bin/env python3
"""RQ2: per-robot single-trajectory metrics.

Computes kinematic feasibility metrics for one GMR pickle on one robot and
writes a JSON report. Used by the batch script to evaluate the same input
clips across multiple robot morphologies.

Example
-------
    python scripts/eval_rq2_per_robot.py \
        --pkl ~/eval_rq2/retargets/unitree_g1/pedalada.pkl \
        --robot unitree_g1 \
        --out_json ~/eval_rq2/metrics/unitree_g1/pedalada.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

from general_motion_retargeting.evaluation.single_robot_metrics import (
    evaluate_single_trajectory,
)
from general_motion_retargeting.params import ROBOT_XML_DICT


def main() -> None:
    robots = sorted(ROBOT_XML_DICT.keys())
    p = argparse.ArgumentParser(description="RQ2 per-robot single-pkl metrics")
    p.add_argument("--pkl", required=True)
    p.add_argument("--robot", required=True, choices=robots)
    p.add_argument("--xml_path", default=None, help="Override MuJoCo XML path")
    p.add_argument("--out_json", required=True)
    p.add_argument(
        "--root_rot_convention",
        default="xyzw",
        choices=("xyzw", "wxyz"),
        help="Quaternion order in pickle root_rot (smplx_to_robot.py uses xyzw).",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--no_foot_sliding", action="store_true")
    args = p.parse_args()

    xml = args.xml_path or str(pathlib.Path(ROBOT_XML_DICT[args.robot]).resolve())
    report = evaluate_single_trajectory(
        pkl_path=str(pathlib.Path(args.pkl).resolve()),
        xml_path=xml,
        robot=args.robot,
        root_rot_convention=args.root_rot_convention,
        device=args.device,
        compute_foot_sliding=not args.no_foot_sliding,
    )

    out_path = pathlib.Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
