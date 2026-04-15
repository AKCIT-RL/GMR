"""End-to-end evaluation: load pair, resample, DTW align, metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from general_motion_retargeting.evaluation.alignment import dtw_distance, resample_full_trajectory
from general_motion_retargeting.evaluation.defaults import ee_bodies_for_robot
from general_motion_retargeting.evaluation.io import assert_compatible_pair, load_motion_pkl
from general_motion_retargeting.evaluation.joint_metrics import (
    compare_jerk,
    joint_limit_violations,
    q_position_errors,
)
from general_motion_retargeting.params import ROBOT_XML_DICT


@dataclass
class EvalConfig:
    mocap_pkl: str
    genmo_pkl: str
    robot: str
    xml_path: Optional[str] = None
    mocap_root_rot_convention: str = "wxyz"
    genmo_root_rot_convention: str = "xyzw"
    target_fps: Optional[float] = None
    dtw_window: Optional[int] = None
    use_fk: bool = True
    use_mujoco: bool = False
    ee_body_names: Optional[List[str]] = None
    device: str = "cpu"


def _resolve_xml(robot: str, xml_path: Optional[str]) -> str:
    if xml_path:
        return str(Path(xml_path).resolve())
    if robot not in ROBOT_XML_DICT:
        raise KeyError(f"Unknown robot {robot!r}; pass xml_path= or use a key from ROBOT_XML_DICT.")
    return str(Path(ROBOT_XML_DICT[robot]).resolve())


def run_evaluation(cfg: EvalConfig) -> Dict[str, Any]:
    traj_m = load_motion_pkl(cfg.mocap_pkl, cfg.mocap_root_rot_convention)  # type: ignore[arg-type]
    traj_g = load_motion_pkl(cfg.genmo_pkl, cfg.genmo_root_rot_convention)  # type: ignore[arg-type]
    assert_compatible_pair(traj_m, traj_g)

    xml = _resolve_xml(cfg.robot, cfg.xml_path)
    tgt_fps = cfg.target_fps if cfg.target_fps is not None else min(traj_m.fps, traj_g.fps)

    rpm, rrm, dqm = resample_full_trajectory(
        traj_m.root_pos, traj_m.root_rot, traj_m.dof_pos, traj_m.fps, tgt_fps
    )
    rpg, rrg, dqg = resample_full_trajectory(
        traj_g.root_pos, traj_g.root_rot, traj_g.dof_pos, traj_g.fps, tgt_fps
    )

    feats_m = dqm
    feats_g = dqg
    dtw_cost, path_m, path_g = dtw_distance(feats_m, feats_g, window=cfg.dtw_window)
    K = len(path_m)
    dqm_w = dqm[path_m]
    dqg_w = dqg[path_g]
    rpm_w = rpm[path_m]
    rrm_w = rrm[path_m]
    rpg_w = rpg[path_g]
    rrg_w = rrg[path_g]

    q_err = q_position_errors(dqm_w, dqg_w)
    q_err_raw = None
    if dqm.shape[0] == dqg.shape[0]:
        q_err_raw = q_position_errors(dqm, dqg)

    jerk_cmp = compare_jerk(dqm, dqg, tgt_fps)
    lim_m = joint_limit_violations(dqm, xml, cfg.device)
    lim_g = joint_limit_violations(dqg, xml, cfg.device)

    ee = cfg.ee_body_names
    if ee is None:
        ee = ee_bodies_for_robot(cfg.robot)

    fk_block = None
    if cfg.use_fk:
        from general_motion_retargeting.evaluation.fk_metrics import fk_full_metrics

        fk_block = fk_full_metrics(
            xml,
            rpm_w,
            rrm_w,
            dqm_w,
            rpg_w,
            rrg_w,
            dqg_w,
            ee_body_names=ee,
            device=cfg.device,
        )

    mj_block = None
    if cfg.use_mujoco:
        from general_motion_retargeting.evaluation.mujoco_metrics import self_collision_rates_pair

        mj_block = self_collision_rates_pair(
            xml,
            (rpm, rrm, dqm),
            (rpg, rrg, dqg),
        )

    report: Dict[str, Any] = {
        "meta": {
            "robot": cfg.robot,
            "xml": xml,
            "mocap_pkl": cfg.mocap_pkl,
            "genmo_pkl": cfg.genmo_pkl,
            "target_fps": tgt_fps,
            "dtw_path_length": int(K),
            "dtw_total_cost": float(dtw_cost),
            "dtw_mean_step_cost": float(dtw_cost / max(1, K)),
        },
        "joint_space": {
            "q_error_dtw_aligned": q_err,
            "q_error_raw_same_length": q_err_raw,
            "jerk": jerk_cmp,
            "joint_limit_violations_mocap": lim_m,
            "joint_limit_violations_genmo": lim_g,
        },
        "fk_task_space": fk_block,
        "mujoco": mj_block,
    }
    return report
