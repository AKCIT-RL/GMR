from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_path", type=str, required=True)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/robot_motion.mp4")
    parser.add_argument(
        "--headless",
        action="store_true",
        default=None,
        help="Força modo headless (sem display). Detectado automaticamente se DISPLAY não estiver definido.",
    )
    args = parser.parse_args()

    robot_type = args.robot
    robot_motion_path = args.robot_motion_path

    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")

    # Detecta headless automaticamente
    headless = args.headless
    if headless is None:
        headless = os.environ.get("DISPLAY", "") == ""

    # No modo headless, força gravação de vídeo (caso contrário não há saída visual)
    record_video = args.record_video
    if headless and not record_video:
        print("[INFO] Modo headless detectado: ativando gravação de vídeo automaticamente.")
        record_video = True

    (
        motion_data,
        motion_fps,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        motion_local_body_pos,
        motion_link_body_list,
    ) = load_robot_motion(robot_motion_path)

    env = RobotMotionViewer(
        robot_type=robot_type,
        motion_fps=motion_fps,
        camera_follow=not headless,
        record_video=record_video,
        video_path=args.video_path,
        headless=headless,
    )

    n_frames = len(motion_root_pos)

    if headless:
        # Modo headless: roda uma passagem e salva vídeo
        for frame_idx in tqdm(range(n_frames), desc="Renderizando"):
            env.step(
                motion_root_pos[frame_idx],
                motion_root_rot[frame_idx],
                motion_dof_pos[frame_idx],
                rate_limit=False,  # Sem rate limit: renderiza o mais rápido possível
            )
    else:
        # Modo interativo: loop contínuo (Ctrl+C para sair)
        frame_idx = 0
        while True:
            env.step(
                motion_root_pos[frame_idx],
                motion_root_rot[frame_idx],
                motion_dof_pos[frame_idx],
                rate_limit=True,
            )
            frame_idx = (frame_idx + 1) % n_frames

    env.close()