"""
core/sim_env.py
===============
MuJoCo wrapper class cho robot liên tục PPT.
Cung cấp API cấp cao để truy cập trạng thái mô phỏng.

Lưu ý thiết kế:
  • Class này KHÔNG quản lý vòng lặp mô phỏng (do main.py điều phối).
  • Chỉ đọc/ghi data và cung cấp các hàm truy vấn tiện dụng.
  • Tất cả phép tính FK chính xác đến từ MuJoCo nội bộ.
"""

from __future__ import annotations

import numpy as np
import mujoco


class SoftRobotEnv:
    """
    Wrapper bọc MuJoCo model/data cho robot liên tục PPT.

    Thuộc tính
    ----------
    model      : mujoco.MjModel
    data       : mujoco.MjData
    n_notches  : Số rãnh cắt (segment) của robot
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data  = data

        # Đếm số notch từ tên khớp
        self.n_notches = self._count_notches()

        # Cache ID các đối tượng quan trọng (tránh lookup lặp đi lặp lại)
        self._tip_site_id     = self._get_site_id("tip_site")
        self._base_site_id    = self._get_site_id("base_site")
        self._tendon_id       = self._get_tendon_id("ppt_tendon")
        self._tip_body_id     = self._get_body_id(f"segment_{self.n_notches}")

        # Cache ID khớp theo thứ tự
        self._bend_jnt_ids  = [self._get_joint_id(f"bend_x_{i+1}")
                                for i in range(self.n_notches)]
        self._twist_jnt_ids = [self._get_joint_id(f"twist_z_{i+1}")
                                for i in range(self.n_notches)]

        # Lưu chiều dài gân ban đầu để tính độ dịch chuyển
        self._initial_tendon_length: float | None = None
        self._record_initial_state()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _count_notches(self) -> int:
        count = 0
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and name.startswith("bend_x_"):
                count += 1
        return count

    def _get_site_id(self, name: str) -> int:
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        except Exception:
            return -1

    def _get_body_id(self, name: str) -> int:
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        except Exception:
            return -1

    def _get_joint_id(self, name: str) -> int:
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        except Exception:
            return -1

    def _get_tendon_id(self, name: str) -> int:
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, name)
        except Exception:
            return -1

    def _record_initial_state(self):
        """Chạy forward pass và lưu trạng thái ban đầu."""
        mujoco.mj_forward(self.model, self.data)
        if self._tendon_id >= 0:
            self._initial_tendon_length = float(
                self.data.ten_length[self._tendon_id])
        else:
            self._initial_tendon_length = 0.0

    # ── Điều khiển mô phỏng ───────────────────────────────────────────────

    def reset(self):
        """Đặt lại mô phỏng về trạng thái ban đầu (t=0, q=0)."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._record_initial_state()

    def step(self, ctrl: np.ndarray | None = None):
        """
        Tiến mô phỏng một timestep.

        Tham số
        -------
        ctrl: (n_actuators,) mảng lực điều khiển. Nếu None, dùng ctrl hiện tại.
        """
        if ctrl is not None:
            n = min(len(ctrl), len(self.data.ctrl))
            self.data.ctrl[:n] = ctrl[:n]
        mujoco.mj_step(self.model, self.data)

    def forward(self):
        """Tính FK mà không tiến thời gian (hữu ích khi cần cập nhật sau set qpos)."""
        mujoco.mj_forward(self.model, self.data)

    # ── Truy vấn trạng thái ───────────────────────────────────────────────

    def get_tip_frame(self) -> dict:
        """
        Lấy vị trí và hướng của tip trong world frame (từ MuJoCo FK nội bộ).

        Trả về
        ------
        dict:
            pos    : (3,)  vị trí [m]
            rotmat : (3,3) ma trận quay
            quat   : (4,)  quaternion [w, x, y, z]
        """
        if self._tip_site_id < 0:
            return {"pos": np.zeros(3),
                    "rotmat": np.eye(3),
                    "quat": np.array([1., 0., 0., 0.])}

        pos    = self.data.site_xpos[self._tip_site_id].copy()
        rotmat = self.data.site_xmat[self._tip_site_id].reshape(3, 3).copy()
        quat   = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotmat.flatten())
        return {"pos": pos, "rotmat": rotmat, "quat": quat}

    def get_tip_euler_deg(self) -> np.ndarray:
        """
        Góc Euler ZYX (roll, pitch, yaw) của tip [độ].
        Roll ≈ xoắn tích lũy, Pitch ≈ góc uốn trong mặt phẳng YZ.
        """
        R = self.get_tip_frame()["rotmat"]
        pitch = np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
        yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        return np.array([roll, pitch, yaw])

    def get_joint_angles(self) -> dict:
        """
        Lấy góc tất cả khớp.

        Trả về
        ------
        dict:
            bend  : (m,) góc uốn [rad]
            twist : (m,) góc xoắn [rad]
        """
        bend  = np.zeros(self.n_notches)
        twist = np.zeros(self.n_notches)
        for i, (jb, jt) in enumerate(
                zip(self._bend_jnt_ids, self._twist_jnt_ids)):
            if jb >= 0:
                bend[i]  = self.data.qpos[self.model.jnt_qposadr[jb]]
            if jt >= 0:
                twist[i] = self.data.qpos[self.model.jnt_qposadr[jt]]
        return {"bend": bend, "twist": twist}

    def get_tendon_state(self) -> dict:
        """
        Trạng thái gân: chiều dài, vận tốc, lực actuator, độ dịch chuyển.

        Chiều dài gân ban đầu được ghi lại lúc khởi tạo.
        Δl = length - length_0:
            Δl < 0 : gân bị kéo ngắn lại (bending)
            Δl > 0 : gân được đẩy dài ra (straightening)
        Note: MuJoCo 3.x không có ten_force; dùng actuator_force thay thế.
        """
        if self._tendon_id < 0:
            return {"length": 0., "velocity": 0., "force": 0., "displacement": 0.}
        length   = float(self.data.ten_length[self._tendon_id])
        velocity = float(self.data.ten_velocity[self._tendon_id])
        # Lực actuator push-pull (index 0)
        force = float(self.data.actuator_force[0]) if len(self.data.actuator_force) > 0 else 0.0
        disp     = length - (self._initial_tendon_length or 0.)
        return {
            "length":       length,
            "velocity":     velocity,
            "force":        force,
            "displacement": disp,
        }

    def get_tip_contact_force(self) -> dict:
        """
        Lực/mô-men ngoại lực tác dụng lên đốt cuối (trong world frame).
        cfrc_ext: [torque(3), force(3)] [N·m, N]
        """
        if self._tip_body_id < 0:
            return {"force_N": np.zeros(3), "torque_Nm": np.zeros(3)}
        cfrc = self.data.cfrc_ext[self._tip_body_id].copy()
        return {"force_N": cfrc[3:6], "torque_Nm": cfrc[0:3]}

    def get_full_state(self) -> dict:
        """Tổng hợp toàn bộ trạng thái mô phỏng."""
        tip    = self.get_tip_frame()
        joints = self.get_joint_angles()
        tendon = self.get_tendon_state()
        euler  = self.get_tip_euler_deg()
        cf     = self.get_tip_contact_force()
        return {
            "time":              float(self.data.time),
            "ctrl":              self.data.ctrl.copy(),
            "tip_pos":           tip["pos"],
            "tip_quat":          tip["quat"],
            "tip_euler_deg":     euler,
            "bend_angles_rad":   joints["bend"],
            "twist_angles_rad":  joints["twist"],
            "tendon_length":     tendon["length"],
            "tendon_disp":       tendon["displacement"],
            "tendon_force":      tendon["force"],
            "tip_contact_force": cf["force_N"],
        }

    # ── Thông tin model ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"SoftRobotEnv("
                f"n_notches={self.n_notches}, "
                f"nv={self.model.nv}, "
                f"nbody={self.model.nbody})")
