"""
sensors/virtual_sensors.py
===========================
Đọc dữ liệu cảm biến ảo từ mô phỏng MuJoCo.

Sơ đồ sensor → sensordata:
  Sensor XML block định nghĩa thứ tự các sensor.
  MuJoCo đóng gói tất cả vào data.sensordata (mảng float64 liên tục).
  model.sensor_adr[id] → chỉ số bắt đầu trong sensordata
  model.sensor_dim[id] → số phần tử

Nguồn dữ liệu bổ sung (không qua sensor block):
  • ten_length[id]   : chiều dài tức thời của gân [m]
  • ten_velocity[id] : vận tốc gân [m/s]
  • ten_force[id]    : lực thụ động trong gân [N]
  • cfrc_ext[body_id]: lực/mô-men ngoài tác dụng lên body [N, N·m]
"""

from __future__ import annotations

import numpy as np
import mujoco


class VirtualSensorSuite:
    """
    Bộ cảm biến ảo đầy đủ cho robot liên tục PPT.

    Đọc dữ liệu từ:
    1. Sensor block trong XML (qua data.sensordata)
    2. Tendon state (data.ten_*)
    3. Body external forces (data.cfrc_ext)
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data  = data

        self._smap: dict[str, tuple[int, int]] = {}
        self._tendon_id              = -1
        self._twist_tendon_id        = -1  # v1.0.1
        self._tip_body_id            = -1
        self._n_notches              = 0
        self._init_tendon_length     = 0.0
        self._init_tw_tendon_length  = 0.0

        self._build_sensor_map()
        self._find_bodies()
        self._record_init()

    # ── Khởi tạo ──────────────────────────────────────────────────────────

    def _build_sensor_map(self):
        """Xây dựng bảng tra cứu: tên sensor → (adr, dim) trong sensordata."""
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                self._smap[name] = (
                    int(self.model.sensor_adr[i]),
                    int(self.model.sensor_dim[i]),
                )

    def _find_bodies(self):
        """Tìm ID các đối tượng cần thiết."""
        # Tendon
        self._tendon_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_TENDON, "ppt_tendon")

        # Dem so notch tu ten body segment
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith("segment_"):
                self._n_notches = max(
                    self._n_notches, int(name.split("_")[1]))

        # Body dot cuoi
        tip_name = f"segment_{self._n_notches}"
        self._tip_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, tip_name)

        # v1.0.1: twist_tendon
        self._twist_tendon_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_TENDON, "twist_tendon")

    def _record_init(self):
        """Ghi chieu dai gan ban dau de tinh Dl."""
        mujoco.mj_forward(self.model, self.data)
        if self._tendon_id >= 0:
            self._init_tendon_length = float(
                self.data.ten_length[self._tendon_id])
        if self._twist_tendon_id >= 0:
            self._init_tw_tendon_length = float(
                self.data.ten_length[self._twist_tendon_id])


    # ── Đọc sensor ────────────────────────────────────────────────────────

    def _read(self, name: str) -> np.ndarray:
        """Đọc giá trị sensor theo tên. Trả về array zeros nếu không tìm thấy."""
        if name in self._smap:
            adr, dim = self._smap[name]
            return self.data.sensordata[adr: adr + dim].copy()
        return np.zeros(1)

    # ── 1. Gân (Tendon) ───────────────────────────────────────────────────

    def read_tendon_length(self) -> float:
        """Chiều dài tức thời của gân [m]."""
        if self._tendon_id >= 0:
            return float(self.data.ten_length[self._tendon_id])
        return float(self._read("tendon_length")[0])

    def read_tendon_displacement(self) -> float:
        """
        Độ dịch chuyển gân Δl [m] so với trạng thái không biến dạng.
        Δl < 0 : gân ngắn lại (pull → uốn)
        Δl > 0 : gân dài ra (push → thẳng lại)
        """
        return self.read_tendon_length() - self._init_tendon_length

    def read_tendon_velocity(self) -> float:
        """Vận tốc gân [m/s]."""
        if self._tendon_id >= 0:
            return float(self.data.ten_velocity[self._tendon_id])
        return float(self._read("tendon_vel")[0])

    def read_tendon_passive_force(self) -> float:
        """
        Lực actuator push-pull [N].
        MuJoCo 3.x: ten_force da bi xoa, doc tu actuator_force thay the.
        """
        if len(self.data.actuator_force) > 0:
            return float(self.data.actuator_force[0])
        return 0.0

    # ── 2. Actuator ───────────────────────────────────────────────────────

    def read_actuator_forces(self) -> dict:
        """
        Lực/mô-men đầu ra thực tế của các actuator [N hoặc N·m].

        Trả về
        ------
        dict:
            push_pull_N : Lực kéo-đẩy gân [N]
            twist_Nm    : Mô-men xoắn [N·m]
        """
        pp = self._read("push_pull_frc")
        tw = self._read("twist_frc")
        return {
            "push_pull_N": float(pp[0]) if len(pp) > 0 else 0.0,
            "twist_Nm":    float(tw[0]) if len(tw) > 0 else 0.0,
        }

    # ── 3. Cảm biến tip (6 trục ảo) ──────────────────────────────────────

    def read_tip_position(self) -> np.ndarray:
        """
        Vị trí tip từ sensor framepos [m].
        Tương đương data.site_xpos nhưng qua pipeline sensor.
        """
        raw = self._read("tip_pos_sensor")
        if len(raw) >= 3:
            return raw[:3]
        return np.zeros(3)

    def read_tip_quaternion(self) -> np.ndarray:
        """Quaternion hướng tip [w, x, y, z]."""
        raw = self._read("tip_quat_sensor")
        if len(raw) >= 4:
            return raw[:4]
        return np.array([1., 0., 0., 0.])

    def read_tip_linear_velocity(self) -> np.ndarray:
        """Vận tốc tịnh tiến tip [m/s] trong world frame."""
        raw = self._read("tip_linvel_sensor")
        if len(raw) >= 3:
            return raw[:3]
        return np.zeros(3)

    def read_tip_contact_force(self) -> dict:
        """
        Lực và mô-men ngoại tác dụng lên đốt cuối [N, N·m].

        Đọc từ data.cfrc_ext: lực và mô-men tổng hợp do các ràng buộc
        (contact constraints, joint constraints) tác dụng lên body.
        Format: [torque(3), force(3)] in world frame.
        """
        if self._tip_body_id >= 0:
            cfrc = self.data.cfrc_ext[self._tip_body_id].copy()
            return {
                "force_N":    cfrc[3:6].copy(),
                "torque_Nm":  cfrc[0:3].copy(),
                "magnitude_N": float(np.linalg.norm(cfrc[3:6])),
            }
        return {
            "force_N":    np.zeros(3),
            "torque_Nm":  np.zeros(3),
            "magnitude_N": 0.0,
        }

    # ── 4. Điều khiển hiện tại ─────────────────────────────────────────────

    def read_ctrl(self) -> dict:
        """Lệnh điều khiển hiện tại (từ slider UI hoặc code)."""
        ctrl = self.data.ctrl
        return {
            "push_pull_N": float(ctrl[0]) if len(ctrl) > 0 else 0.0,
            "twist_Nm":    float(ctrl[1]) if len(ctrl) > 1 else 0.0,
        }

    # ── 5. Đọc tất cả một lần ────────────────────────────────────────────

    def read_all(self) -> dict:
        """
        Đọc toàn bộ cảm biến và trả về dict thống nhất.
        Dùng cho logging và điều khiển.
        """
        act  = self.read_actuator_forces()
        cf   = self.read_tip_contact_force()
        ctrl = self.read_ctrl()
        return {
            # Gân
            "tendon_length":        self.read_tendon_length(),
            "tendon_disp":          self.read_tendon_displacement(),
            "tendon_vel":           self.read_tendon_velocity(),
            "tendon_passive_force": self.read_tendon_passive_force(),
            # Actuator
            "actuator_push_pull_N": act["push_pull_N"],
            "actuator_twist_Nm":    act["twist_Nm"],
            # Tip
            "tip_pos":              self.read_tip_position(),
            "tip_quat":             self.read_tip_quaternion(),
            "tip_linvel":           self.read_tip_linear_velocity(),
            # Tip contact
            "tip_contact_force_N":  cf["force_N"],
            "tip_contact_torque_Nm":cf["torque_Nm"],
            "tip_contact_mag_N":    cf["magnitude_N"],
            # Lệnh điều khiển
            "ctrl_push_pull_N":     ctrl["push_pull_N"],
            "ctrl_twist_Nm":        ctrl["twist_Nm"],
        }

    # ── Thông tin debug ───────────────────────────────────────────────────

    def print_sensor_map(self):
        """In bảng sensor để debug."""
        print(f"\n{'Sensor Name':<25} {'Adr':>5} {'Dim':>4}")
        print("─" * 38)
        for name, (adr, dim) in sorted(self._smap.items()):
            print(f"{name:<25} {adr:>5} {dim:>4}")
        print(f"Total sensordata size: {len(self.data.sensordata)}\n")
