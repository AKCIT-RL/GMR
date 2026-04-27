# GMR retargeting evaluation

Compare two robot trajectories produced by GMR (e.g. Xsens BVH vs GENMO/GVHMR), after saving `motion_data` pickles from `xsens_bvh_to_robot.py` / `gvhmr_to_robot.py`.

## Dependencies with uv

From the repository root (requires [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
uv run pytest tests/test_evaluation_metrics.py -v
```

To run the CLI without activating a venv:

```bash
uv run python scripts/eval_retarget_metrics.py --help
```

## CLI

```bash
python scripts/eval_retarget_metrics.py \
  --mocap_pkl /path/to/mocap.pkl \
  --genmo_pkl /path/to/genmo.pkl \
  --robot unitree_g1 \
  --out_json report.json
```

- **Quaternion convention:** `xsens_bvh_to_robot.py` saves `root_rot` as MuJoCo **wxyz**; `gvhmr_to_robot.py` and `bvh_to_robot.py` save **xyzw**. Defaults match this (`--mocap_root_rot_convention wxyz`, `--genmo_root_rot_convention xyzw`).
- **`--no_fk`:** skip task-space metrics (KinematicsModel FK).
- **`--mujoco`:** self-collision proxy via contact penetration; requires a scene XML without extra moving objects, or tune exclusions in code.
- **Torque / `mj_inverse`:** not implemented. Torques depend on contacts, timestep, and actuator model; treat as future work if you need hardware-style limits.

## Orientation normalization

BVH and video-based (SMPL-X/GVHMR) pipelines often produce trajectories that differ in two ways unrelated to retargeting quality:

**1. Global heading** — the robot may face +X in one trajectory and −X in the other depending on how the person was oriented during capture. Computing FK metrics on raw world-frame positions inflates errors even for identical joint poses.

**2. Vertical (Z) drift** — monocular video estimators (GVHMR) infer depth from a single camera. In motions with horizontal displacement (trot, run), the model accumulates small per-frame Z errors that sum to a visible upward drift (up to ~0.13 m over a full motion). BVH capture has no such drift since 3D positions are measured directly. This drift inflates MPJPE and EE RMSE and dominates the ranking for locomotion motions.

By default (`normalize_orientation=True`), the pipeline applies `canonicalize_heading` to both trajectories before FK:

1. Extracts the yaw of frame 0's `root_rot` and rotates all frames so the robot starts facing +X
2. Subtracts frame 0 XY so the trajectory starts at the origin
3. Subtracts frame 0 Z so both trajectories start at the same height

This makes FK metrics (MPJPE, EE RMSE, bone cosine) **invariant to initial heading, XY offset, and absolute Z offset**. Joint-space metrics (`dof_pos` RMSE/MAE, DTW) are already rotation-invariant and are **not affected**.

> **Note on Z normalization:** subtracting frame 0 Z removes the absolute height offset but not the accumulated drift within the motion. Motions with large drift (trot, run) will still show higher MPJPE than static motions — but the metric now reflects genuine pose divergence rather than a systematic estimator bias.

To disable and see raw world-frame errors:

```bash
python scripts/eval_retarget_metrics.py ... --no_normalize_orientation
```

## Metrics

| Block | Content |
|-------|---------|
| DTW | Dynamic time warping on resampled `dof_pos`; aligned path for joint/FK errors |
| Joint | RMSE/MAE per joint (DTW-aligned), jerk (`np.gradient`), joint limit violation rate (XML limits) |
| FK | Mean body-frame L2 error (MPJPE-like), optional EE RMSE, mean bone-direction cosine — computed after orientation normalization by default |
| MuJoCo | Optional: fraction of frames with penetrating robot–robot contacts |

## Programmatic use

```python
from general_motion_retargeting.evaluation import EvalConfig, run_evaluation
from general_motion_retargeting.evaluation.report import save_report

cfg = EvalConfig(mocap_pkl="a.pkl", genmo_pkl="b.pkl", robot="unitree_g1")
save_report("out.json", run_evaluation(cfg))
```
