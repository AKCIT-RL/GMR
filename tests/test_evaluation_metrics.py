"""Unit tests for general_motion_retargeting.evaluation."""

import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from general_motion_retargeting.evaluation.alignment import dtw_distance, resample_full_trajectory
from general_motion_retargeting.evaluation.io import quat_to_xyzw
from general_motion_retargeting.evaluation.joint_metrics import jerk_metrics, q_position_errors
from general_motion_retargeting.evaluation.pipeline import EvalConfig, run_evaluation
from general_motion_retargeting.params import ROBOT_XML_DICT


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _write_pkl(path: str, T: int, D: int, fps: float = 30.0, q_scale: float = 0.0) -> None:
    t = np.linspace(0, 1, T)
    root_pos = np.stack([np.zeros(T), np.zeros(T), 0.8 * np.ones(T)], axis=1)
    # xyzw identity
    root_rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (T, 1))
    dof_pos = np.zeros((T, D), dtype=np.float64)
    dof_pos[:, 0] = q_scale * np.sin(2 * np.pi * t)
    with open(path, "wb") as f:
        pickle.dump({"fps": fps, "root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof_pos}, f)


class TestIO(unittest.TestCase):
    def test_quat_xyzw_roundtrip(self) -> None:
        wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        xyzw = quat_to_xyzw(wxyz, "wxyz")
        np.testing.assert_allclose(xyzw, np.array([0.0, 0.0, 0.0, 1.0]))


class TestAlignment(unittest.TestCase):
    def test_dtw_identical(self) -> None:
        a = np.random.randn(20, 5)
        cost, pa, pb = dtw_distance(a, a.copy())
        self.assertLess(cost, 1e-6)
        np.testing.assert_array_equal(pa, pb)

    def test_resample_short(self) -> None:
        rp = np.zeros((10, 3))
        rr = np.tile([0, 0, 0, 1.0], (10, 1))
        dq = np.linspace(0, 1, 10)[:, None]
        rp2, rr2, dq2 = resample_full_trajectory(rp, rr, dq, 30.0, 30.0)
        self.assertEqual(dq2.shape[0], 10)


class TestJointMetrics(unittest.TestCase):
    def test_q_error_zero(self) -> None:
        q = np.random.randn(15, 7)
        r = q_position_errors(q, q.copy())
        self.assertLess(r["rmse_global"], 1e-9)

    def test_jerk_positive(self) -> None:
        t = np.linspace(0, 1, 40)[:, None]
        q = np.sin(10 * t)
        j = jerk_metrics(q, fps=30.0)
        self.assertGreater(j["jerk_l2_mean"], 0.0)


@unittest.skipUnless(_have_torch(), "torch required for KinematicsModel joint limits / FK")
class TestPipelineSynthetic(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = self._tmpdir.name

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_identical_trajectories_small_dof(self) -> None:
        # Use 29 DOF to match G1 mocap XML; FK + limits run on real model.
        D = 29
        p1 = os.path.join(self.tmp, "a.pkl")
        p2 = os.path.join(self.tmp, "b.pkl")
        _write_pkl(p1, T=40, D=D, fps=30.0, q_scale=0.1)
        _write_pkl(p2, T=40, D=D, fps=30.0, q_scale=0.1)

        cfg = EvalConfig(
            mocap_pkl=p1,
            genmo_pkl=p2,
            robot="unitree_g1",
            mocap_root_rot_convention="xyzw",
            genmo_root_rot_convention="xyzw",
            use_mujoco=False,
        )
        rep = run_evaluation(cfg)
        self.assertLess(rep["joint_space"]["q_error_dtw_aligned"]["rmse_global"], 1e-6)
        self.assertIsNotNone(rep["fk_task_space"])
        self.assertLess(rep["fk_task_space"]["mpjpe_like"]["mean_mpjpe_like_m"], 0.05)


@unittest.skipUnless(_have_torch(), "torch required")
@unittest.skipUnless(
    Path(ROBOT_XML_DICT["unitree_g1"]).exists(),
    "G1 asset XML missing",
)
class TestPipelineSmokeG1(unittest.TestCase):
    """Smoke test with mismatched length (still 29 DOF)."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = self._tmpdir.name

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_different_lengths(self) -> None:
        D = 29
        p1 = os.path.join(self.tmp, "a.pkl")
        p2 = os.path.join(self.tmp, "b.pkl")
        _write_pkl(p1, T=60, D=D, fps=30.0, q_scale=0.2)
        _write_pkl(p2, T=40, D=D, fps=30.0, q_scale=0.1)
        cfg = EvalConfig(
            mocap_pkl=p1,
            genmo_pkl=p2,
            robot="unitree_g1",
            mocap_root_rot_convention="xyzw",
            genmo_root_rot_convention="xyzw",
            use_fk=True,
            use_mujoco=False,
        )
        rep = run_evaluation(cfg)
        self.assertIn("dtw_path_length", rep["meta"])
        self.assertGreater(rep["meta"]["dtw_path_length"], 0)
        self.assertIsNotNone(rep["fk_task_space"])


if __name__ == "__main__":
    unittest.main()
