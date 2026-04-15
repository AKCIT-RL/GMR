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

## Metrics

| Block | Content |
|-------|---------|
| DTW | Dynamic time warping on resampled `dof_pos`; aligned path for joint/FK errors |
| Joint | RMSE/MAE per joint (DTW-aligned), jerk (`np.gradient`), joint limit violation rate (XML limits) |
| FK | Mean body-frame L2 error (MPJPE-like), optional EE RMSE, mean bone-direction cosine |
| MuJoCo | Optional: fraction of frames with penetrating robot–robot contacts |

## Programmatic use

```python
from general_motion_retargeting.evaluation import EvalConfig, run_evaluation
from general_motion_retargeting.evaluation.report import save_report

cfg = EvalConfig(mocap_pkl="a.pkl", genmo_pkl="b.pkl", robot="unitree_g1")
save_report("out.json", run_evaluation(cfg))
```
