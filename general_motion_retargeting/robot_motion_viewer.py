import os
import time

# Deve ser definido ANTES de qualquer import do mujoco para funcionar em servidores
# sem display X11. egl = GPU headless; osmesa = CPU software fallback.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco as mj
import mujoco.viewer as mjv
import imageio
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1

class RobotMotionViewer:
    def __init__(self,
                robot_type,
                camera_follow=True,
                motion_fps=30,
                transparent_robot=0,
                # video recording
                record_video=False,
                video_path=None,
                video_width=1280,
                video_height=720,
                keyboard_callback=None,
                # headless: pula launch_passive, usa apenas Renderer offscreen
                headless=None,
                ):
        
        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        mj.mj_step(self.model, self.data)
        
        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video

        # Auto-detecta headless se não especificado: sem DISPLAY → headless
        if headless is None:
            headless = os.environ.get("DISPLAY", "") == ""
        self.headless = headless

        if self.headless:
            # Modo sem janela: câmera gerenciada manualmente
            self.viewer = None
            self._cam = mj.MjvCamera()
            self._cam.type = mj.mjtCamera.mjCAMERA_TRACKING
            self._cam.trackbodyid = self.model.body(self.robot_base).id
            self._cam.distance = self.viewer_cam_distance
            self._cam.azimuth = 135.0
            self._cam.elevation = -10.0
            self._opt = mj.MjvOption()
            self._opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot
            print("[RobotMotionViewer] Modo HEADLESS (sem display X11).")
        else:
            self.viewer = mjv.launch_passive(
                model=self.model,
                data=self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=keyboard_callback
            )
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot

        # Renderer offscreen: usado no modo headless OU quando record_video=True
        if self.headless or self.record_video:
            self._renderer = mj.Renderer(self.model, height=video_height, width=video_width)
        else:
            self._renderer = None

        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)
            if video_dir and not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"[RobotMotionViewer] Gravando vídeo em {self.video_path}")
        
    def step(self, 
            # robot data
            root_pos, root_rot, dof_pos, 
            # human data
            human_motion_data=None, 
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization    
            human_pos_offset=np.array([0.0, 0.0, 0]),
            # rate limit
            rate_limit=True, 
            follow_camera=True,
            ):
        """
        Atualiza o estado do robô e renderiza um frame.

        Suporta dois modos:
        - Normal (headless=False): usa mujoco.viewer.launch_passive (requer X11)
        - Headless (headless=True): usa apenas mujoco.Renderer offscreen (sem display)
        """
        
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot  # quat scalar-first para o MuJoCo
        self.data.qpos[7:] = dof_pos
        
        mj.mj_forward(self.model, self.data)

        if self.headless:
            # Modo headless: apenas renderiza offscreen (para vídeo se solicitado)
            if self.record_video:
                self._renderer.update_scene(self.data, camera=self._cam, scene_option=self._opt)
                img = self._renderer.render()
                self.mp4_writer.append_data(img)
            if rate_limit:
                self.rate_limiter.sleep()
        else:
            # Modo normal com janela
            if follow_camera:
                self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id]
                self.viewer.cam.distance = self.viewer_cam_distance
                self.viewer.cam.elevation = -10

            if human_motion_data is not None:
                self.viewer.user_scn.ngeom = 0
                for human_body_name, (pos, rot) in human_motion_data.items():
                    draw_frame(
                        pos,
                        R.from_quat(rot, scalar_first=True).as_matrix(),
                        self.viewer,
                        human_point_scale,
                        pos_offset=human_pos_offset,
                        joint_name=human_body_name if show_human_body_name else None
                    )

            self.viewer.sync()
            if rate_limit:
                self.rate_limiter.sleep()

            if self.record_video:
                self._renderer.update_scene(self.data, camera=self.viewer.cam)
                img = self._renderer.render()
                self.mp4_writer.append_data(img)
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            time.sleep(0.5)
        if self._renderer is not None:
            self._renderer.close()
        if self.record_video:
            self.mp4_writer.close()
            print(f"[RobotMotionViewer] Vídeo salvo em {self.video_path}")
