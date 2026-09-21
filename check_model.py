"""
check_model.py
==============
Script nhanh để xem trước XML model trong MuJoCo viewer.
Tự động tái tạo XML trước khi load để đảm bảo model luôn cập nhật.
"""

import sys

# Tái tạo XML (đảm bảo dùng config mới nhất)
from builder.xml_generator import generate_continuum_robot_xml

ROBOT_ID = 2   # Thay đổi để xem robot khác

print(f"[check_model] Generating XML for Robot {ROBOT_ID}...")
xml_path, m_notches = generate_continuum_robot_xml(robot_id=ROBOT_ID)
print(f"[check_model] Loading: {xml_path}  ({m_notches} notches, {m_notches+1} twist motors)")

import mujoco
import mujoco.viewer

try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    print(f"[check_model] Model OK — DOFs: {model.nv}, Bodies: {model.nbody}, "
          f"Sensors: {model.nsensor}")
    print(f"[check_model] Launching viewer...")
    print(f"             Nhấn Ctrl+A để hiện Actuator sliders")
    mujoco.viewer.launch(model, data)
except Exception as e:
    print(f"[check_model] LỖI: {e}")
    sys.exit(1)
