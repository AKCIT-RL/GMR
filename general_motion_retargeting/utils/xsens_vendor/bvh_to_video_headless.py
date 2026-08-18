"""
bvh_to_video_headless.py
------------------------
Processa um BVH aplicando offsets.json (já existente) e renderiza
um vídeo em modo OFFSCREEN — sem abrir nenhuma janela Qt ou MuJoCo.

Uso:
    uv run python general_motion_retargeting/utils/xsens_vendor/bvh_to_video_headless.py \
        --bvh_file ../mocap/hello.bvh \
        --offsets offsets.json \
        --output output.mp4 \
        --scale 0.01 \
        --reset_to_zero
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# CRITICAL: definir backend OpenGL ANTES de qualquer import do mujoco.
#   egl   → renderização GPU sem display X11 (recomendado em servidores)
#   osmesa → renderização CPU (fallback se não houver GPU/EGL)
# ---------------------------------------------------------------------------
os.environ.setdefault("MUJOCO_GL", "egl")
# Se EGL falhar, tente: os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import mujoco
import cv2

# ---------------------------------------------------------------------------
# Configuração do caminho para os módulos do projeto
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from BVHParser import BVHParser, Anim, quat_fk
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

channel_names = ["X", "Y", "Z"]


def load_offsets(path: str) -> dict:
    """Carrega offsets.json e retorna o dicionário {joint_name: {X, Y, Z}}."""
    if not os.path.exists(path):
        print(f"[WARN] offsets.json não encontrado em '{path}'. Usando zeros.")
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    print(f"[INFO] Offsets carregados de '{path}'.")
    return data


def apply_offsets_to_rotations(
    rotations: np.ndarray,
    joint_names: list,
    offsets_dict: dict,
) -> np.ndarray:
    """
    Aplica os offsets (em graus) às rotações Euler.

    rotations: (T, N, 3)
    """
    new_rotations = rotations.copy()
    for j, name in enumerate(joint_names):
        joint_data = offsets_dict.get(name, {"X": 0.0, "Y": 0.0, "Z": 0.0})
        for c, ch in enumerate(channel_names):
            new_rotations[:, j, c] += joint_data.get(ch, 0.0)
    return new_rotations


# ---------------------------------------------------------------------------
# Renderização offscreen
# ---------------------------------------------------------------------------

def render_bvh_to_video(
    parser: BVHParser,
    anim: Anim,
    output_path: str,
    width: int = 1280,
    height: int = 720,
    fps: float | None = None,
    scale: float = 0.01,
):
    """Renderiza a animação BVH para um arquivo de vídeo sem abrir janelas."""

    fps = fps or (1.0 / parser.frame_time)
    frames_len = len(parser.frames)
    frame_time = parser.frame_time

    # ------------------------------------------------------------------
    # Gera XML e carrega modelo MuJoCo
    # ------------------------------------------------------------------
    xml_content = parser.generate_mujoco_xml(frame_0=anim.pos[0, 0])
    xml_file = "human_skeleton_headless.xml"
    with open(xml_file, "w") as f:
        f.write(xml_content)
    print(f"[INFO] MuJoCo XML salvo em '{xml_file}'")

    model = mujoco.MjModel.from_xml_path(xml_file)
    data = mujoco.MjData(model)

    # Configura câmera livre (tracking do body 0)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 0
    cam.distance = 5.0
    cam.azimuth = 135.0
    cam.elevation = -15.0

    opt = mujoco.MjvOption()

    # ------------------------------------------------------------------
    # Writer de vídeo
    # ------------------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Renderizando {frames_len} frames a {fps:.1f} fps → '{output_path}'")

    # Usa context manager: faz renderer.close() automaticamente ao sair
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for frame_idx in range(frames_len):
            if frame_idx % 50 == 0:
                print(f"  frame {frame_idx}/{frames_len}", end="\r", flush=True)

            for idx, name in enumerate(anim.bones):
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint"
                )
                if joint_id > 0:
                    qpos_idx = model.jnt_qposadr[joint_id]
                    data.qpos[qpos_idx : qpos_idx + 4] = anim.quats[frame_idx, idx, :]
                else:
                    # root (Hips)
                    data.qpos[0:3] = anim.pos[frame_idx, 0, :]
                    data.qpos[3:7] = anim.quats[frame_idx, 0, :]

            data.qvel[:] = 0
            mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=cam, scene_option=opt)
            img_rgb = renderer.render()  # (H, W, 3) uint8 RGB

            # OpenCV usa BGR
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            writer.write(img_bgr)

    writer.release()
    print(f"\n[OK] Vídeo salvo em '{os.path.abspath(output_path)}'")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="BVH → vídeo headless (sem janelas)")
    ap.add_argument("--bvh_file", required=True, help="Arquivo BVH de entrada")
    ap.add_argument(
        "--offsets",
        default="offsets.json",
        help="Arquivo JSON de offsets (default: offsets.json no CWD)",
    )
    ap.add_argument(
        "--output",
        default="output.mp4",
        help="Caminho do vídeo de saída (default: output.mp4)",
    )
    ap.add_argument("--scale", type=float, default=0.01, help="Escala de deslocamento")
    ap.add_argument(
        "--reset_to_zero",
        action="store_true",
        default=False,
        help="Zera deslocamento e rotação Z inicial",
    )
    ap.add_argument("--width", type=int, default=1280, help="Largura do vídeo (px)")
    ap.add_argument("--height", type=int, default=720, help="Altura do vídeo (px)")
    ap.add_argument("--start", type=int, default=None, help="Frame inicial (inclusivo)")
    ap.add_argument("--end", type=int, default=None, help="Frame final (exclusivo)")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # 1. Parsear BVH
    # ------------------------------------------------------------------
    parser = BVHParser("zxy", args.scale)
    with open(args.bvh_file, "r") as f:
        bvh_text = f.read()
    rotations, positions = parser.parse(
        bvh_text, start=args.start, end=args.end, reset_to_zero=False
    )
    joint_names = parser.names

    # ------------------------------------------------------------------
    # 2. Aplicar offsets do JSON (equivale ao botão "Apply and Preview")
    # ------------------------------------------------------------------
    offsets_dict = load_offsets(args.offsets)
    rotations = apply_offsets_to_rotations(rotations, joint_names, offsets_dict)

    # ------------------------------------------------------------------
    # 3. Pós-processamento (quats + reset_to_zero)
    # ------------------------------------------------------------------
    quats, positions, offsets_arr, parents = parser._MOTION_data_post_processing(
        rotations, positions, reset_to_zero=args.reset_to_zero
    )
    anim = Anim(quats, positions, offsets_arr, parents, joint_names)

    # ------------------------------------------------------------------
    # 4. Renderizar vídeo offscreen
    # ------------------------------------------------------------------
    render_bvh_to_video(
        parser=parser,
        anim=anim,
        output_path=args.output,
        width=args.width,
        height=args.height,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
