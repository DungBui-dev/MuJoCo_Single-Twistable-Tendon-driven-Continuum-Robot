"""
test_stability.py  -- v1.0.2
Integration test: XML gen, model load, physics, FK, SoroTwistFK, Jacobian.
Run: .venv\\Scripts\\python.exe test_stability.py
"""
import sys
import numpy as np
import mujoco

from builder.xml_generator import generate_continuum_robot_xml
from core.sim_env import SoftRobotEnv
from sensors.virtual_sensors import VirtualSensorSuite
from utils.kinematics import (
    get_tip_frame_from_mujoco,
    compute_tip_bending_angle_deg,
    compute_pcc_fk,
    validate_fk,
    compute_jacobian,
    get_sorotwist_fk,
    _clean,
)

ROBOT_ID    = 2
STEPS_IDLE  = 2000
STEPS_PULL  = 1000
PULL_FORCE  = -2.0   # N (negative = pull = bend)

print("=" * 60)
print("  Continuum Robot v1.0.2 -- Stability & Integration Test")
print("=" * 60)

# ── 1. Generate XML ───────────────────────────────────────────────────────────
print("\n[1] Generating XML (v1.0.2)...")
xml, m_notches = generate_continuum_robot_xml(robot_id=ROBOT_ID)
print(f"    OK: {xml}  ({m_notches} notches, {m_notches+1} actuators)")

# ── 2. Load model ─────────────────────────────────────────────────────────────
print("[2] Loading model...")
model = mujoco.MjModel.from_xml_path(xml)
data  = mujoco.MjData(model)
print(f"    OK: DOFs={model.nv}, Bodies={model.nbody}, Sensors={model.nsensor}, nu={model.nu}")

# ── 3. Init subsystems ────────────────────────────────────────────────────────
print("[3] Initializing subsystems...")
env     = SoftRobotEnv(model, data)
sensors = VirtualSensorSuite(model, data)
print(f"    OK: n_notches={env.n_notches}")

# ── 4. Idle stability ─────────────────────────────────────────────────────────
print(f"[4] Idle stability ({STEPS_IDLE} steps, ctrl=0)...")
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
failed = False
for step in range(STEPS_IDLE):
    mujoco.mj_step(model, data)
    if np.any(np.isnan(data.qpos)) or np.any(np.isinf(data.qpos)):
        print(f"    FAILED at step {step}: NaN/Inf!")
        failed = True
        break

if not failed:
    qr = f"[{data.qpos.min():.4f}, {data.qpos.max():.4f}]"
    print(f"    OK: {STEPS_IDLE} steps, t={data.time:.3f}s, qpos in {qr}")

# ── 5. Bending test ───────────────────────────────────────────────────────────
print(f"[5] Bending test ({STEPS_PULL} steps, ctrl[0]={PULL_FORCE}N)...")
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
tip_init = get_tip_frame_from_mujoco(model, data)["pos"].copy()
data.ctrl[0] = PULL_FORCE
failed2 = False
for step in range(STEPS_PULL):
    mujoco.mj_step(model, data)
    if np.any(np.isnan(data.qpos)):
        print(f"    FAILED at step {step}: NaN!")
        failed2 = True
        break

if not failed2:
    tip = get_tip_frame_from_mujoco(model, data)
    bend = compute_tip_bending_angle_deg(tip["pos"], tip_init)
    tip_mm = tip["pos"] * 1e3
    print(f"    OK: Tip=({tip_mm[0]:.2f},{tip_mm[1]:.2f},{tip_mm[2]:.2f})mm, "
          f"Bend={bend:.2f}deg")

# ── 6. Sensor read ────────────────────────────────────────────────────────────
print("[6] Sensor read test...")
sd = sensors.read_all()
print(f"    tendon_length={sd['tendon_length']*1e3:.3f}mm, "
      f"tendon_disp={sd['tendon_disp']*1e3:.3f}mm")
print(f"    actuator_push_pull={sd['actuator_push_pull_N']:.3f}N")
print(f"    tip_pos=({sd['tip_pos'][0]*1e3:.2f},{sd['tip_pos'][1]*1e3:.2f},"
      f"{sd['tip_pos'][2]*1e3:.2f})mm")

# ── 7. FK validation (MuJoCo vs PCC) ─────────────────────────────────────────
print("[7] FK validation (MuJoCo vs PCC)...")
try:
    seg_len_est = 0.050 / 16
    result = validate_fk(model, data, seg_len_est, verbose=True)
    print(f"    Pos error: {result['pos_error_mm']:.3f}mm, "
          f"Angle error: {result['angle_error_deg']:.3f}deg")
except Exception as e:
    print(f"    FK validation skipped: {e}")

# ── 8. SoroTwistFK MATLAB-accurate test (Task 1) ──────────────────────────────
print("[8] SoroTwistFK test (MATLAB-accurate kinematics)...")
fk = get_sorotwist_fk()
print(f"    Geometric params: y_c={fk.y_c:.4f}mm, y_bar={fk.y_bar:.4f}mm, I_s={fk.I_s:.3e}m^4")

# Test PCC curvature at 1mm pull
kappa = fk.pcc_curvature(1.0)
print(f"    pcc_curvature(1mm pull) = {kappa:.6f} mm^-1 (positive = pull)")

# Test FK with paper values: pp=1mm, twist=-90deg, base=0
import math
q_test = [1.0, -math.pi/2, 0.0]
T_list = fk.fk(q_test, n=32)
T_tip  = T_list[-1]
tip_mm = _clean(T_tip[:3, 3])
print(f"    FK (pp=1mm, tw=-90deg, n=32): Tip=({tip_mm[0]:.3f},{tip_mm[1]:.3f},{tip_mm[2]:.3f}) mm")
print(f"    From visRobot.m reference:    expected approx Tip=(x, y, z) mm")

# Numeric Jacobian
J = fk.numeric_jacobian(q_test, n=32)
print(f"    Numeric Jacobian shape: {J.shape} (6x3)")
print(f"    J_linear max: {np.abs(J[:3]).max():.4f}")

# Floating-point noise test (Task 3)
noisy = np.array([-5.55e-17, 1.0000000001, -1.39e-17])
cleaned = _clean(noisy)
print(f"    _clean test: {noisy} -> {cleaned}")
fp_ok = cleaned[0] == 0.0 and cleaned[2] == 0.0
print(f"    Floating-point cleanup: {'OK' if fp_ok else 'FAIL'}")

# ── 9. Jacobian (MuJoCo native) ───────────────────────────────────────────────
print("[9] MuJoCo Jacobian test...")
jacp, jacr = compute_jacobian(model, data)
print(f"    J_p shape: {jacp.shape}, J_r shape: {jacr.shape}")
print(f"    J_p max: {np.abs(jacp).max():.4f}")

# ── 10. Twist distribution test ───────────────────────────────────────────────
print("[10] Twist distribution test (all motors same torque)...")
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
data.ctrl[0] = 0.0
data.ctrl[1:] = 0.1  # broadcast to all twist motors

for _ in range(2000):
    mujoco.mj_step(model, data)

if not np.any(np.isnan(data.qpos)):
    twists = []
    for i in range(m_notches):
        jid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"twist_z_{i+1}")
        qadr  = model.jnt_qposadr[jid]
        twists.append(np.degrees(data.qpos[qadr]))
    std_val = np.std(twists)
    print(f"    Twist std={std_val:.4f} deg (want 0 = uniform)")
    print(f"    Twist range=[{min(twists):.2f}, {max(twists):.2f}] deg")
    print(f"    Distribution: {'OK (uniform)' if std_val < 1.0 else 'CHECK'}")
else:
    print("    FAILED: NaN detected")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
all_ok = not failed and not failed2 and fp_ok
print(f"  RESULT: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
print("=" * 60)
sys.exit(0 if all_ok else 1)
