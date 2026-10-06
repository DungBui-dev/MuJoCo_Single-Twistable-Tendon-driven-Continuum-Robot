"""
utils/logger.py
===============
Ghi dữ liệu mô phỏng ra CSV theo thời gian thực.

Thiết kế:
  • Sử dụng bộ đệm (buffer) trong RAM để giảm I/O.
  • Tự động flush khi buffer đầy hoặc khi gọi save().
  • Thread-safe cơ bản (dùng lock cho buffer).

Cấu trúc CSV:
  time_sim, time_wall,
  ctrl_push_pull_N, ctrl_twist_Nm,
  bend_x_1_rad .. bend_x_m_rad,
  twist_z_1_rad .. twist_z_m_rad,
  tip_x_m, tip_y_m, tip_z_m,
  tip_roll_deg, tip_pitch_deg, tip_yaw_deg,
  tendon_length_m, tendon_disp_m, tendon_force_N,
  tip_fx_N, tip_fy_N, tip_fz_N,
  bending_angle_deg
"""

from __future__ import annotations

import csv
import time
import threading
import numpy as np
from pathlib import Path


class DataLogger:
    """
    Ghi trạng thái mô phỏng ra file CSV với buffer.

    Tham số
    -------
    filepath    : Đường dẫn file CSV đầu ra
    n_notches   : Số rãnh cắt của robot (để tạo header)
    buffer_size : Số hàng tích lũy trước khi flush ra đĩa
    """

    def __init__(
        self,
        filepath: str,
        n_notches: int,
        buffer_size: int = 200,
    ):
        self.filepath    = Path(filepath)
        self.n_notches   = n_notches
        self.buffer_size = buffer_size

        self._buffer: list[list] = []
        self._lock   = threading.Lock()
        self._file   = None
        self._writer = None
        self._rows_written = 0
        self._start_time   = time.time()

        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._open_file()

    # ── Nội bộ ────────────────────────────────────────────────────────────

    def _build_header(self) -> list[str]:
        h = ["time_sim_s", "time_wall_s"]
        h += ["ctrl_push_pull_N", "ctrl_twist_Nm"]
        h += [f"bend_x_{i+1}_rad"  for i in range(self.n_notches)]
        h += [f"twist_z_{i+1}_rad" for i in range(self.n_notches)]
        h += ["tip_x_m", "tip_y_m", "tip_z_m"]
        h += ["tip_roll_deg", "tip_pitch_deg", "tip_yaw_deg"]
        h += ["tendon_length_m", "tendon_disp_m", "tendon_force_N"]
        h += ["tip_fx_N", "tip_fy_N", "tip_fz_N"]
        h += ["bending_angle_deg"]
        return h

    def _open_file(self):
        self._file   = open(self.filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self._build_header())
        self._file.flush()

    def _flush_locked(self):
        """Flush buffer ra file (phải được gọi khi đang giữ lock)."""
        if self._writer and self._buffer:
            self._writer.writerows(self._buffer)
            self._file.flush()
            self._rows_written += len(self._buffer)
            self._buffer.clear()

    # ── API công khai ─────────────────────────────────────────────────────

    def log_step(
        self,
        time_sim:         float,
        ctrl:             np.ndarray,
        bend_angles:      np.ndarray,
        twist_angles:     np.ndarray,
        tip_pos:          np.ndarray,
        tip_euler_deg:    np.ndarray,
        tendon_length:    float,
        tendon_disp:      float,
        tendon_force:     float,
        tip_contact_force: np.ndarray,
        bending_angle_deg: float,
    ):
        """
        Ghi một bước mô phỏng vào buffer.

        Gọi mỗi N timestep từ vòng lặp chính. Sẽ tự flush khi buffer đầy.

        Tham số
        -------
        time_sim          : Thời gian mô phỏng [s]
        ctrl              : (n_act,) lệnh điều khiển [N hoặc N·m]
        bend_angles       : (m,) góc uốn [rad]
        twist_angles      : (m,) góc xoắn [rad]
        tip_pos           : (3,) vị trí tip [m]
        tip_euler_deg     : (3,) [roll, pitch, yaw] [độ]
        tendon_length     : Chiều dài gân hiện tại [m]
        tendon_disp       : Δl = length - length_0 [m]
        tendon_force      : Lực actuator push-pull [N]
        tip_contact_force : (3,) lực tiếp xúc tại tip [N]
        bending_angle_deg : Góc uốn tổng thể của tip [độ]
        """
        wall_time = time.time() - self._start_time

        row: list = [f"{time_sim:.6f}", f"{wall_time:.3f}"]

        # Lệnh điều khiển
        row.append(f"{ctrl[0]:.4f}" if len(ctrl) > 0 else "0.0000")
        row.append(f"{ctrl[1]:.4f}" if len(ctrl) > 1 else "0.0000")

        # Góc khớp uốn và xoắn
        bn = bend_angles[:self.n_notches]
        tn = twist_angles[:self.n_notches]
        # Pad nếu ít hơn n_notches
        if len(bn) < self.n_notches:
            bn = np.pad(bn, (0, self.n_notches - len(bn)))
        if len(tn) < self.n_notches:
            tn = np.pad(tn, (0, self.n_notches - len(tn)))
        row += [f"{a:.7f}" for a in bn]
        row += [f"{a:.7f}" for a in tn]

        # Tip pose
        row += [f"{tip_pos[0]:.7f}", f"{tip_pos[1]:.7f}", f"{tip_pos[2]:.7f}"]
        row += [f"{tip_euler_deg[0]:.4f}",
                f"{tip_euler_deg[1]:.4f}",
                f"{tip_euler_deg[2]:.4f}"]

        # Gân
        row += [f"{tendon_length:.8f}",
                f"{tendon_disp:.8f}",
                f"{tendon_force:.5f}"]

        # Lực tiếp xúc tip
        cf = tip_contact_force
        row += [f"{cf[0]:.5f}", f"{cf[1]:.5f}", f"{cf[2]:.5f}"]

        # Góc uốn tổng thể
        row.append(f"{bending_angle_deg:.4f}")

        with self._lock:
            self._buffer.append(row)
            if len(self._buffer) >= self.buffer_size:
                self._flush_locked()

    def flush(self):
        """Flush thủ công bộ đệm ra đĩa."""
        with self._lock:
            self._flush_locked()

    def save(self):
        """Flush toàn bộ và đóng file. Gọi khi kết thúc mô phỏng."""
        self.flush()
        if self._file:
            self._file.close()
            self._file   = None
            self._writer = None
        elapsed = time.time() - self._start_time
        print(f"[Logger] ✓ Đã lưu {self._rows_written:,} hàng → {self.filepath}")
        print(f"[Logger]   Thời gian ghi: {elapsed:.1f}s")

    @property
    def rows_written(self) -> int:
        return self._rows_written + len(self._buffer)

    def __del__(self):
        """Đảm bảo flush khi object bị thu hồi."""
        try:
            if self._buffer:
                self._flush_locked()
            if self._file:
                self._file.close()
        except Exception:
            pass

