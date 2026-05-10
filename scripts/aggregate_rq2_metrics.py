#!/usr/bin/env python3
"""RQ2: aggregate per-clip per-robot JSONs into a summary CSV + LaTeX table.

Walks ``--metrics_root`` expecting layout::

    metrics_root/<robot>/<clip>.json

Writes:
- ``<out_dir>/per_clip.csv`` — flat row per (robot, clip)
- ``<out_dir>/per_robot.csv`` — mean ± std per robot
- ``<out_dir>/per_robot.tex`` — LaTeX-ready table
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from typing import Any, Dict, List


def _flatten(report: Dict[str, Any]) -> Dict[str, float]:
    """Pull a fixed set of scalar metrics out of a per-clip report."""
    out: Dict[str, float] = {}
    out["success"] = float(bool(report.get("status", {}).get("success", False)))

    jl = report.get("joint_limits", {})
    out["jl_violation_rate"] = float(jl.get("violation_rate", math.nan))
    out["jl_max_mag_rad"] = float(jl.get("max_violation_magnitude_rad", math.nan))
    out["jl_mean_norm_margin"] = float(jl.get("mean_normalized_margin", math.nan))

    ws = report.get("workspace", {})
    out["workspace_mean_coverage"] = float(ws.get("mean_coverage", math.nan))

    vel = report.get("velocity", {})
    out["vel_mean_rad_s"] = float(vel.get("vel_l2_mean_rad_s", math.nan))
    out["vel_p95_rad_s"] = float(vel.get("vel_l2_p95_rad_s", math.nan))

    jerk = report.get("jerk", {})
    out["jerk_mean_rad_s3"] = float(jerk.get("jerk_l2_mean_rad_s3", math.nan))
    out["jerk_p95_rad_s3"] = float(jerk.get("jerk_l2_p95_rad_s3", math.nan))

    fs = report.get("foot_sliding", {})
    out["foot_sliding_mean_m_s"] = float(fs.get("mean_in_contact_speed_m_s", math.nan))
    return out


METRIC_COLS: List[str] = [
    "success",
    "jl_violation_rate",
    "jl_max_mag_rad",
    "jl_mean_norm_margin",
    "workspace_mean_coverage",
    "vel_mean_rad_s",
    "vel_p95_rad_s",
    "jerk_mean_rad_s3",
    "jerk_p95_rad_s3",
    "foot_sliding_mean_m_s",
]


def _mean_std(xs: List[float]) -> (float, float):
    finite = [x for x in xs if not math.isnan(x)]
    if not finite:
        return math.nan, math.nan
    n = len(finite)
    mu = sum(finite) / n
    var = sum((x - mu) ** 2 for x in finite) / max(n - 1, 1)
    return mu, math.sqrt(var)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_root", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    root = pathlib.Path(args.metrics_root)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    by_robot: Dict[str, List[Dict[str, float]]] = {}

    for robot_dir in sorted(root.iterdir()):
        if not robot_dir.is_dir():
            continue
        robot = robot_dir.name
        for jpath in sorted(robot_dir.glob("*.json")):
            with open(jpath) as f:
                report = json.load(f)
            flat = _flatten(report)
            row = {"robot": robot, "clip": jpath.stem, **flat}
            rows.append(row)
            by_robot.setdefault(robot, []).append(flat)

    # per_clip.csv
    per_clip_path = out_dir / "per_clip.csv"
    with open(per_clip_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["robot", "clip", *METRIC_COLS])
        w.writeheader()
        w.writerows(rows)

    # per_robot.csv with mean ± std
    per_robot_path = out_dir / "per_robot.csv"
    summary_rows = []
    with open(per_robot_path, "w", newline="") as f:
        cols = ["robot", "n_clips"]
        for m in METRIC_COLS:
            cols += [f"{m}_mean", f"{m}_std"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for robot, flats in sorted(by_robot.items()):
            r: Dict[str, Any] = {"robot": robot, "n_clips": len(flats)}
            for m in METRIC_COLS:
                mu, sd = _mean_std([fl[m] for fl in flats])
                r[f"{m}_mean"] = mu
                r[f"{m}_std"] = sd
            w.writerow(r)
            summary_rows.append(r)

    # per_robot.tex
    tex_path = out_dir / "per_robot.tex"
    headers = [
        ("Robot", "robot"),
        ("N", "n_clips"),
        ("Success", "success_mean"),
        ("JL Viol \\% $\\downarrow$", "jl_violation_rate_mean"),
        ("Margin $\\uparrow$", "jl_mean_norm_margin_mean"),
        ("Coverage $\\uparrow$", "workspace_mean_coverage_mean"),
        ("Vel p95 $\\downarrow$", "vel_p95_rad_s_mean"),
        ("Jerk p95 $\\downarrow$", "jerk_p95_rad_s3_mean"),
        ("Foot slide $\\downarrow$", "foot_sliding_mean_m_s_mean"),
    ]

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            if math.isnan(v):
                return "--"
            return f"{v:.3f}"
        return str(v)

    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}\n\\toprule\n")
        f.write(" & ".join(h for h, _ in headers) + " \\\\\n\\midrule\n")
        for r in summary_rows:
            f.write(" & ".join(fmt(r.get(k, "")) for _, k in headers) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"Wrote {per_clip_path}")
    print(f"Wrote {per_robot_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
