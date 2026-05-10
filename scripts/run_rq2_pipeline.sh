#!/usr/bin/env bash
# RQ2 end-to-end pipeline: retarget the same generative motion clips to N
# robots, evaluate per-robot kinematic metrics, and aggregate into a
# comparison table.
#
# Input formats (auto-detected from file extension under <input_dir>):
#   *.pt   — HMR4D / GVHMR predictions (use INPUT_FORMAT=gvhmr to force)
#   *.npz  — SMPL-X stageii (AMASS-style; use INPUT_FORMAT=smplx to force)
#
# Usage:
#   bash scripts/run_rq2_pipeline.sh <input_dir> <out_root> [robot1 robot2 ...]
#
# Defaults to 6 morphologically diverse robots if none are provided.
#
# Output layout (out_root):
#   retargets/<robot>/<clip>.pkl    — GMR pickle per clip per robot
#   metrics/<robot>/<clip>.json     — per-clip metrics
#   summary/per_clip.csv            — flat row per (robot, clip)
#   summary/per_robot.csv           — mean ± std per robot
#   summary/per_robot.tex           — LaTeX table

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <smplx_input_dir> <out_root> [robot1 robot2 ...]"
  exit 2
fi

SMPLX_DIR=$(realpath "$1")
OUT_ROOT=$(realpath -m "$2")
shift 2

if [[ $# -ge 1 ]]; then
  ROBOTS=("$@")
else
  ROBOTS=(unitree_g1 booster_t1_29dof unitree_h1_2 fourier_n1 engineai_pm01 stanford_toddy)
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
NUM_CPUS=${NUM_CPUS:-4}
INPUT_FORMAT=${INPUT_FORMAT:-auto}   # auto | smplx | gvhmr
HERE=$(cd "$(dirname "$0")" && pwd)
GMR_ROOT=$(cd "$HERE/.." && pwd)

# Auto-detect input format from file extensions in $SMPLX_DIR.
if [[ "$INPUT_FORMAT" == "auto" ]]; then
  if find "$SMPLX_DIR" -maxdepth 3 -type f -name '*.pt' | head -1 | grep -q .; then
    INPUT_FORMAT=gvhmr
  elif find "$SMPLX_DIR" -maxdepth 3 -type f -name '*.npz' | head -1 | grep -q .; then
    INPUT_FORMAT=smplx
  else
    echo "[RQ2][ERR] no .pt or .npz files under $SMPLX_DIR"
    exit 3
  fi
fi

echo "[RQ2] Input dir    : $SMPLX_DIR"
echo "[RQ2] Input format : $INPUT_FORMAT"
echo "[RQ2] Output root  : $OUT_ROOT"
echo "[RQ2] Robots       : ${ROBOTS[*]}"
echo "[RQ2] Python       : $PYTHON_BIN"

mkdir -p "$OUT_ROOT/retargets" "$OUT_ROOT/metrics" "$OUT_ROOT/summary" "$OUT_ROOT/logs"

# ---------- Stage 1: retarget ----------
for robot in "${ROBOTS[@]}"; do
  TGT="$OUT_ROOT/retargets/$robot"
  LOG="$OUT_ROOT/logs/retarget_${robot}.log"
  mkdir -p "$TGT"
  echo "[RQ2][retarget] $robot -> $TGT (log: $LOG)"
  if [[ "$INPUT_FORMAT" == "gvhmr" ]]; then
    ( cd "$GMR_ROOT" && "$PYTHON_BIN" scripts/gvhmr_to_robot_dataset.py \
        --robot "$robot" \
        --src_folder "$SMPLX_DIR" \
        --tgt_folder "$TGT" ) > "$LOG" 2>&1 || {
          echo "[RQ2][retarget][WARN] $robot failed; see $LOG"; }
  else
    ( cd "$GMR_ROOT" && "$PYTHON_BIN" scripts/smplx_to_robot_dataset.py \
        --robot "$robot" \
        --src_folder "$SMPLX_DIR" \
        --tgt_folder "$TGT" \
        --num_cpus "$NUM_CPUS" ) > "$LOG" 2>&1 || {
          echo "[RQ2][retarget][WARN] $robot failed; see $LOG"; }
  fi
done

# ---------- Stage 2: per-clip metrics ----------
for robot in "${ROBOTS[@]}"; do
  SRC="$OUT_ROOT/retargets/$robot"
  DST="$OUT_ROOT/metrics/$robot"
  mkdir -p "$DST"
  if [[ ! -d "$SRC" ]]; then
    echo "[RQ2][eval][skip] $SRC missing"
    continue
  fi
  while IFS= read -r -d '' pkl; do
    clip=$(basename "$pkl" .pkl)
    out="$DST/$clip.json"
    if [[ -s "$out" ]]; then
      continue
    fi
    echo "[RQ2][eval] $robot/$clip"
    ( cd "$GMR_ROOT" && "$PYTHON_BIN" scripts/eval_rq2_per_robot.py \
        --pkl "$pkl" \
        --robot "$robot" \
        --out_json "$out" ) || {
          echo "[RQ2][eval][WARN] failed: $robot/$clip"; }
  done < <(find "$SRC" -type f -name '*.pkl' -print0)
done

# ---------- Stage 3: aggregate ----------
echo "[RQ2][aggregate]"
( cd "$GMR_ROOT" && "$PYTHON_BIN" scripts/aggregate_rq2_metrics.py \
    --metrics_root "$OUT_ROOT/metrics" \
    --out_dir "$OUT_ROOT/summary" )

echo "[RQ2] Done. See $OUT_ROOT/summary/"
