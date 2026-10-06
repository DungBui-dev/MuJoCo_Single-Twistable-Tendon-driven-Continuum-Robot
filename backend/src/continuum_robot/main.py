import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
main.py  -- v1.0.2
====================
Single Twistable Tendon-Driven Continuum Robot Simulation

v1.0.2 Changes:
  [T1] SoroTwistFK (MATLAB-accurate) duoc tich hop, hien thi tip pos bang FK chinh xac
  [T3] Floating-point noise cleanup (_clean) trong tat ca output
  [T4] Rebuilt 3D environment: floor on dinh, 4-light setup
  [T5] WASD toi uu: threading-safe key queue, khong lag/xung dot Slider
       Reset dong bo: R/Backspace ep qpos+ctrl ve 0 ngay lap tuc

Dieu khien Ban phim (WASD):
  W : Pull tendon -- uon cong (giam ctrl[0])
  S : Push tendon -- thang lai (tang ctrl[0])
  A : Twist trai  -- xoan toan than
  D : Twist phai  -- xoan toan than
  R : RESET TOAN BO (qpos=0, ctrl=0, velocity=0)
  Q : Giam push-pull dan dan
  E : Giam twist dan dan
  [ : Giam buoc WASD (min 0.01)
  ] : Tang buoc WASD (max 1.0)

Camera 3D (built-in MuJoCo viewer):
  Chuot trai + keo : Orbit
  Chuot phai + keo : Pan
  Scroll           : Zoom
  Ctrl+A           : Hien/an Actuator sliders
"""

import sys
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer
import yaml
from pathlib import Path

# OpenCV (optional)
try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False
    print("[WARN] opencv-python not installed. Endoscope camera disabled.")

from continuum_robot.robots.robot_builder import generate_continuum_robot_xml
from continuum_robot.algorithms.keyboard_ctrl import KeyboardController
from continuum_robot.core.sim_env import SoftRobotEnv
from continuum_robot.sensors.virtual_sensors import VirtualSensorSuite
from continuum_robot.utils.kinematics import (
    get_tip_frame_from_mujoco,
    get_tip_euler_xyz,
    compute_tip_bending_angle_deg,
    get_sorotwist_fk,
)
from continuum_robot.utils.logger import DataLogger

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs/robot_params.yaml" if "main.py" in __file__ else "../../src/continuum_robot/configs/robot_params.yaml"))
ROBOT_ID    = 2

# max adjustable step


def _load_params(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _print_banner(robot_id: int, params: dict):
    print()
    print("=" * 64)
    print("  Single Twistable Tendon-Driven Continuum Robot  v1.0.2")
    print(f"  Robot ID: {robot_id}  |  Integrator: {params.get('integrator','?')}")
    print("=" * 64)


def _print_guide():
    print()
    print("  WASD Keyboard Control:")
    print("    W  = Pull tendon   (bend forward)")
    print("    S  = Push tendon   (straighten)")
    print("    A  = Twist LEFT    (full body)")
    print("    D  = Twist RIGHT   (full body)")
    print("    R  = RESET ALL     (qpos=0, ctrl=0, vel=0)")
    print("    Q  = Release push-pull gradually")
    print("    E  = Release twist gradually")
    print("    [  = Decrease step size")
    print("    ]  = Increase step size")
    print()
    print("  3D Camera (MuJoCo viewer):")
    print("    Left drag   = Orbit (rotate view)")
    print("    Right drag  = Pan")
    print("    Scroll      = Zoom in/out")
    print("    Ctrl+A      = Toggle Actuator sliders panel")
    print()
    print("  ESC in endoscope = close camera | Close viewer = end sim & save CSV")
    print()




# ─────────────────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    params = _load_params(CONFIG_PATH)
    _print_banner(ROBOT_ID, params)

    # -- 1. Sinh XML model -------------------------------------------------------
    print("\n[1/5] Generating XML model (v1.0.2)...")
    xml_path, m_notches = generate_continuum_robot_xml(
        robot_id=ROBOT_ID, config_path=CONFIG_PATH
    )

    # -- 2. Load model -----------------------------------------------------------
    print("[2/5] Loading MuJoCo model...")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data  = mujoco.MjData(model)
    except Exception as e:
        print(f"  [ERROR] Cannot load model: {e}")
        return 1

    print(f"       DOFs: {model.nv}  |  Bodies: {model.nbody}  "
          f"|  Sensors: {model.nsensor}  |  nu: {model.nu}"
          f"  |  dt: {model.opt.timestep:.4f}s")

    # -- 3. Subsystems -----------------------------------------------------------
    print("[3/5] Initializing subsystems...")
    env     = SoftRobotEnv(model, data)
    sensors = VirtualSensorSuite(model, data)

    # SoroTwistFK instance (Task 1)
    fk_solver = get_sorotwist_fk()

    log_file     = params.get("log_file", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/simulation_log.csv")))
    log_interval = int(params.get("log_interval", 5))
    logger       = DataLogger(log_file, n_notches=env.n_notches)

    cam_w   = int(params.get("camera_width",  640))
    cam_h   = int(params.get("camera_height", 360))
    cam_fps = float(params.get("camera_fps",  30))
    cam_interval = max(1, int(1.0 / (cam_fps * float(model.opt.timestep))))

    print(f"       n_notches: {env.n_notches} | Camera: {cam_w}x{cam_h}@{cam_fps:.0f}fps")
    print(f"       SoroTwistFK: y_c={fk_solver.y_c:.4f}mm  "
          f"y_bar={fk_solver.y_bar:.4f}mm  I_s={fk_solver.I_s:.3e}m^4")

    # -- 4. Keyboard controller (Task 5) -----------------------------------------
    kb = KeyboardController()

    # -- 5. Renderer endoscope ---------------------------------------------------
    renderer      = None
    camera_active = False
    if _CV2:
        try:
            renderer = mujoco.Renderer(model, height=cam_h, width=cam_w)
            camera_active = True
            cv2.namedWindow("Endoscopic View | Tip Camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Endoscopic View | Tip Camera", cam_w, cam_h)
        except Exception as e:
            print(f"  [WARN] Camera init failed: {e}")

    print(f"       Endoscope camera: {'ON' if camera_active else 'OFF'}")

    _print_guide()
    print("[4/5] Launching passive viewer...")
    mujoco.mj_forward(model, data)
    tip_frame_init  = get_tip_frame_from_mujoco(model, data)
    tip_pos_initial = tip_frame_init["pos"].copy()

    # -- 6. Simulation loop ------------------------------------------------------
    step_count  = 0
    wall_start  = time.time()
    last_status = wall_start

    with mujoco.viewer.launch_passive(
        model, data,
        show_left_ui=True,
        show_right_ui=True,
        key_callback=kb.key_callback,   # WASD binding (GLFW thread)
    ) as viewer:

        # Goc nhin 3D mac dinh
        viewer.cam.azimuth   = 30.0
        viewer.cam.elevation = -20.0
        viewer.cam.distance  = 0.18
        viewer.cam.lookat[:] = [0.0, 0.0, 0.025]

        print("[5/5] Simulation running. Use WASD to control robot.\n")

        while viewer.is_running():

            # ── Task 5: xu ly phim pending va cap nhat ctrl ─────────────────
            # flush_pending() xu ly hang doi phim, ap dung ctrl, xu ly reset
            kb.flush_pending(model, data)

            # ── Tien mo phong ───────────────────────────────────────────────
            mujoco.mj_step(model, data)

            # ── Camera noi soi ──────────────────────────────────────────────
            if camera_active and (step_count % cam_interval == 0):
                try:
                    renderer.update_scene(data, camera="endoscope")
                    pixels = renderer.render()
                    img = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

                    # HUD voi Task 3: du lieu da duoc _clean() tu kinematics
                    tip_fr = get_tip_frame_from_mujoco(model, data)
                    ba     = compute_tip_bending_angle_deg(
                        tip_fr["pos"], tip_pos_initial)
                    euler  = get_tip_euler_xyz(tip_fr["rotmat"])
                    ctrl   = kb.ctrl

                    # Task 1: hien thi tip pos tu SoroTwistFK
                    pp_mm = ctrl[0] * 5.0   # rough N -> mm mapping
                    tw_rad = ctrl[1] * 10.0  # rough Nm -> rad mapping

                    hud = [
                        f"t = {data.time:.3f} s",
                        f"Bend: {ba:.1f} deg",
                        f"Roll: {euler[0]:.1f} deg",
                        f"[W/S] PP: {ctrl[0]:+.2f} N",
                        f"[A/D] TW: {ctrl[1]:+.2f} Nm",
                        f"[R]=Reset  [/]=Step",
                    ]
                    for j, line in enumerate(hud):
                        cv2.putText(img, line, (10, 22 + j * 22),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (0, 255, 120), 1, cv2.LINE_AA)

                    cv2.putText(img, "ENDOSCOPIC VIEW [sim] v1.0.2",
                                (10, cam_h - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.40, (160, 160, 160), 1, cv2.LINE_AA)

                    cv2.imshow("Endoscopic View | Tip Camera", img)
                    key = cv2.waitKey(1)
                    if key == 27:
                        cv2.destroyAllWindows()
                        camera_active = False
                        print("  [Camera] Endoscope window closed.")
                except Exception:
                    pass

            # ── Log CSV (Task 3: du lieu da _clean tu kinematics) ───────────
            if step_count % log_interval == 0:
                tip_fr     = get_tip_frame_from_mujoco(model, data)
                euler_deg  = get_tip_euler_xyz(tip_fr["rotmat"])
                sensor_rd  = sensors.read_all()
                joint_ang  = env.get_joint_angles()
                bend_angle = compute_tip_bending_angle_deg(
                    tip_fr["pos"], tip_pos_initial)

                logger.log_step(
                    time_sim          = float(data.time),
                    ctrl              = data.ctrl.copy(),
                    bend_angles       = joint_ang["bend"],
                    twist_angles      = joint_ang["twist"],
                    tip_pos           = tip_fr["pos"],
                    tip_euler_deg     = euler_deg,
                    tendon_length     = sensor_rd["tendon_length"],
                    tendon_disp       = sensor_rd["tendon_disp"],
                    tendon_force      = sensor_rd["actuator_push_pull_N"],
                    tip_contact_force = sensor_rd["tip_contact_force_N"],
                    bending_angle_deg = bend_angle,
                )

            # ── Status print moi 5 giay ─────────────────────────────────────
            now = time.time()
            if now - last_status >= 5.0:
                st  = env.get_full_state()
                tip = st["tip_pos"]
                ba  = compute_tip_bending_angle_deg(tip, tip_pos_initial)
                print(
                    f"  t={data.time:.1f}s | Bend={ba:.1f}deg | "
                    f"Tip=({tip[0]*1e3:.1f},{tip[1]*1e3:.1f},{tip[2]*1e3:.1f})mm | "
                    f"PP={data.ctrl[0]:+.2f}N TW={data.ctrl[1]:+.2f}Nm | "
                    f"Log={logger.rows_written:,}"
                )
                last_status = now

            step_count += 1
            viewer.sync()

    # Ket thuc
    total_wall = time.time() - wall_start
    print()
    print("=" * 62)
    print(f"  Simulation ended after {total_wall:.1f}s real time")
    print(f"  Sim time: {data.time:.3f}s  |  Steps: {step_count:,}")
    print("=" * 62)

    if camera_active and _CV2:
        cv2.destroyAllWindows()
    if renderer is not None:
        try:
            renderer.close()
        except Exception:
            pass

    logger.save()
    print(f"\n  Hysteresis analysis:")
    print(f"    python backend/src/continuum_robot/utils/hysteresis_plot.py --csv {log_file} --smooth --save")
    return 0


if __name__ == "__main__":
    sys.exit(main())
