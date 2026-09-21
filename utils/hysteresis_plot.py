"""
utils/hysteresis_plot.py
========================
Phân tích và vẽ biểu đồ trễ cơ học (hysteresis) từ dữ liệu mô phỏng CSV.

Biểu đồ 1 — Vòng trễ (Hysteresis Loop):
  Trục X: Độ dịch chuyển gân Δl [mm]  (âm = kéo = uốn)
  Trục Y: Góc uốn tip [độ]
  Màu sắc phân biệt: Loading (kéo) / Unloading (nhả)

Biểu đồ 2 — Lỗi PCC vs. MuJoCo:
  So sánh góc uốn thực tế (từ FK) với dự đoán tuyến tính (PCC đơn giản)

Biểu đồ 3 — Chuỗi thời gian:
  Tendon displacement, bending angle, twist, control input theo thời gian

Sử dụng:
  python utils/hysteresis_plot.py --csv simulation_log.csv
  python utils/hysteresis_plot.py --csv simulation_log.csv --smooth --save
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Hàm trợ giúp
# ─────────────────────────────────────────────────────────────────────────────

def _savgol(y: np.ndarray, window: int = 51, polyorder: int = 3) -> np.ndarray:
    """Lọc Savitzky-Golay để làm mịn tín hiệu."""
    try:
        from scipy.signal import savgol_filter
        w = min(window, len(y) // 2 * 2 - 1)
        if w < 5:
            return y
        return savgol_filter(y, w, polyorder)
    except ImportError:
        return y


def _detect_direction(x: np.ndarray) -> np.ndarray:
    """
    Phân loại chiều của tín hiệu x theo thời gian:
    +1 = tăng (unloading: gân dài ra, uốn giảm)
    -1 = giảm (loading:  gân ngắn lại, uốn tăng)
     0 = không đổi
    """
    dx   = np.gradient(x)
    sign = np.sign(dx)
    # Nội suy các điểm = 0 từ giá trị trước
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i - 1]
    return sign


def _safe_col(df: pd.DataFrame, col: str, default=0.0) -> np.ndarray:
    """Đọc cột an toàn, trả về mảng zeros nếu không tồn tại."""
    if col in df.columns:
        return df[col].values.astype(float)
    print(f"  [Warn] Cột '{col}' không có trong CSV, dùng giá trị mặc định.")
    return np.full(len(df), default)


# ─────────────────────────────────────────────────────────────────────────────
# Hàm vẽ chính
# ─────────────────────────────────────────────────────────────────────────────

def plot_hysteresis(
    csv_path: str,
    smooth:   bool = False,
    save:     bool = False,
):
    """
    Đọc CSV và tạo bộ biểu đồ phân tích trễ cơ học.

    Tham số
    -------
    csv_path : Đường dẫn file CSV từ DataLogger
    smooth   : Áp dụng lọc Savitzky-Golay cho tín hiệu
    save     : Lưu hình ảnh ra file PNG
    """
    # ── Đọc dữ liệu ──────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[HysteresisPlot] Lỗi: Không tìm thấy file '{csv_path}'")
        print("  Hãy chạy main.py trước để tạo dữ liệu mô phỏng.")
        return

    if len(df) < 10:
        print(f"[HysteresisPlot] Lỗi: Quá ít dữ liệu ({len(df)} hàng). "
              "Cần ít nhất 10 hàng.")
        return

    print(f"[HysteresisPlot] Đọc {len(df):,} hàng từ: {csv_path}")

    # ── Trích xuất cột ────────────────────────────────────────────────────
    t     = _safe_col(df, "time_sim_s")
    disp  = _safe_col(df, "tendon_disp_m") * 1000   # m → mm
    bend  = _safe_col(df, "bending_angle_deg")
    ctrl  = _safe_col(df, "ctrl_push_pull_N")
    twist = _safe_col(df, "ctrl_twist_Nm")
    tip_z = _safe_col(df, "tip_z_m") * 1000          # m → mm
    pitch = _safe_col(df, "tip_pitch_deg")
    roll  = _safe_col(df, "tip_roll_deg")
    tlen  = _safe_col(df, "tendon_length_m") * 1000  # m → mm

    # Lọc mịn nếu yêu cầu
    if smooth:
        bend  = _savgol(bend)
        pitch = _savgol(pitch)
        disp  = _savgol(disp)

    # Chuẩn hóa thời gian [0, 1]
    t_norm = (t - t.min()) / max(t.max() - t.min(), 1e-9)

    # Phân loại loading / unloading
    direction = _detect_direction(disp)
    load_mask   = direction <= 0   # Gân ngắn lại (kéo) → uốn
    unload_mask = direction >  0   # Gân dài ra (nhả) → thẳng

    # ── Figure 1: Vòng trễ chính ─────────────────────────────────────────
    fig1, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig1.suptitle(
        "Phân tích Trễ Cơ học — Single Tendon Continuum Robot (PPT)",
        fontsize=14, fontweight="bold",
    )
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    # Panel trái: Vòng trễ loading/unloading
    ax1 = axes[0]
    if np.any(load_mask):
        ax1.plot(
            disp[load_mask], bend[load_mask],
            "b-o", ms=2, lw=1.8, label="Loading (kéo gân ↓)",
            zorder=3,
        )
    if np.any(unload_mask):
        ax1.plot(
            disp[unload_mask], bend[unload_mask],
            "r--o", ms=2, lw=1.8, label="Unloading (nhả gân ↑)",
            zorder=2,
        )
    # Điểm đầu và cuối
    ax1.scatter([disp[0]], [bend[0]], c="green",  s=60, zorder=5,
                label=f"Bắt đầu ({disp[0]:.2f} mm, {bend[0]:.1f}°)")
    ax1.scatter([disp[-1]], [bend[-1]], c="purple", s=60, marker="*", zorder=5,
                label=f"Kết thúc ({disp[-1]:.2f} mm, {bend[-1]:.1f}°)")

    ax1.set_xlabel("Độ dịch chuyển gân  Δl [mm]\n(< 0: kéo → uốn   |   > 0: đẩy → thẳng)",
                   fontsize=11)
    ax1.set_ylabel("Góc uốn đầu mút [độ]", fontsize=11)
    ax1.set_title("Vòng Trễ: Góc uốn vs. Dịch chuyển gân", fontsize=12)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.35, ls="--")
    ax1.axhline(0, color="k", lw=0.6, ls=":")
    ax1.axvline(0, color="k", lw=0.6, ls=":")

    # Panel phải: Lỗ màu theo thời gian
    ax2 = axes[1]
    sc = ax2.scatter(disp, bend, c=t_norm, cmap="plasma", s=8, alpha=0.8)
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label("Thời gian chuẩn hóa (0 → 1)", fontsize=10)
    ax2.set_xlabel("Độ dịch chuyển gân  Δl [mm]", fontsize=11)
    ax2.set_ylabel("Góc uốn đầu mút [độ]", fontsize=11)
    ax2.set_title("Quỹ đạo theo Thời gian (màu = thời gian)", fontsize=12)
    ax2.grid(True, alpha=0.35, ls="--")

    plt.tight_layout()
    if save:
        out = Path(csv_path).with_suffix(".hysteresis.png")
        fig1.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[HysteresisPlot] Đã lưu: {out}")

    # ── Figure 2: Chuỗi thời gian ─────────────────────────────────────────
    fig2 = plt.figure(figsize=(14, 9))
    gs   = gridspec.GridSpec(3, 2, figure=fig2, hspace=0.45, wspace=0.35)
    fig2.suptitle("Chuỗi Thời gian Mô phỏng", fontsize=13, fontweight="bold")

    # [0,0] Tendon displacement
    ax_d = fig2.add_subplot(gs[0, 0])
    ax_d.plot(t, disp, "b-", lw=1.5, label="Δl [mm]")
    ax_d.set_ylabel("Dịch chuyển gân [mm]")
    ax_d.set_title("Dịch chuyển gân")
    ax_d.legend(); ax_d.grid(True, alpha=0.3); ax_d.axhline(0, color="k", lw=0.5)

    # [0,1] Tendon length
    ax_l = fig2.add_subplot(gs[0, 1])
    ax_l.plot(t, tlen, "c-", lw=1.5, label="L_tendon [mm]")
    ax_l.set_ylabel("Chiều dài gân [mm]")
    ax_l.set_title("Chiều dài tuyệt đối gân")
    ax_l.legend(); ax_l.grid(True, alpha=0.3)

    # [1,0] Bending angle + Tip-Z
    ax_b = fig2.add_subplot(gs[1, 0])
    ax_b.plot(t, bend,  "r-", lw=1.8, label="Góc uốn [°]")
    ax_b.plot(t, pitch, "m--", lw=1.0, alpha=0.7, label="Tip pitch [°]")
    ax_b.set_ylabel("Góc [độ]")
    ax_b.set_title("Góc uốn đầu mút")
    ax_b.legend(); ax_b.grid(True, alpha=0.3)

    # [1,1] Tip Z position
    ax_z = fig2.add_subplot(gs[1, 1])
    ax_z.plot(t, tip_z, "g-", lw=1.5, label="Tip Z [mm]")
    ax_z.set_ylabel("Vị trí Z tip [mm]")
    ax_z.set_title("Vị trí dọc trục của tip")
    ax_z.legend(); ax_z.grid(True, alpha=0.3)

    # [2,0] Control inputs
    ax_c = fig2.add_subplot(gs[2, 0])
    ax_c.plot(t, ctrl,  "darkorange", lw=1.8, label="Push-Pull [N]")
    ax_c.set_xlabel("Thời gian [s]")
    ax_c.set_ylabel("Lực kéo-đẩy [N]")
    ax_c.set_title("Tín hiệu điều khiển Push-Pull")
    ax_c.legend(); ax_c.grid(True, alpha=0.3); ax_c.axhline(0, color="k", lw=0.5)

    # [2,1] Twist control + Roll angle
    ax_tw = fig2.add_subplot(gs[2, 1])
    ax_tw.plot(t, twist, "purple", lw=1.8, label="Twist ctrl [N·m]")
    ax_tw2 = ax_tw.twinx()
    ax_tw2.plot(t, roll, "olive", lw=1.0, ls="--", alpha=0.7, label="Tip roll [°]")
    ax_tw2.set_ylabel("Góc xoắn tip [°]", color="olive")
    ax_tw.set_xlabel("Thời gian [s]")
    ax_tw.set_ylabel("Mô-men xoắn [N·m]")
    ax_tw.set_title("Xoắn: Điều khiển & Phản hồi")
    ax_tw.legend(loc="upper left"); ax_tw2.legend(loc="upper right")
    ax_tw.grid(True, alpha=0.3)

    if save:
        out2 = Path(csv_path).with_suffix(".timeseries.png")
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        print(f"[HysteresisPlot] Đã lưu: {out2}")

    # ── Tóm tắt thống kê ──────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("  Tóm tắt thống kê")
    print("═" * 50)
    print(f"  Thời gian mô phỏng : {t[-1]:.3f} s ({len(df):,} bước log)")
    print(f"  Dịch chuyển gân    : [{disp.min():.3f}, {disp.max():.3f}] mm")
    print(f"  Góc uốn max        : {bend.max():.2f}°")
    print(f"  Góc xoắn ctrl max  : {abs(twist).max():.4f} N·m")
    print(f"  Tip Z range        : [{tip_z.min():.2f}, {tip_z.max():.2f}] mm")

    if np.any(load_mask) and np.any(unload_mask):
        # Ước tính độ rộng vòng trễ
        load_bend   = bend[load_mask]
        unload_bend = bend[unload_mask]
        hysteresis_width = abs(np.mean(load_bend) - np.mean(unload_bend))
        print(f"  Độ rộng vòng trễ  : ~{hysteresis_width:.2f}°  "
              "(trung bình loading vs unloading)")
    print("═" * 50 + "\n")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phân tích trễ cơ học từ log mô phỏng continuum robot"
    )
    parser.add_argument(
        "--csv", type=str, default="simulation_log.csv",
        help="Đường dẫn file CSV (mặc định: simulation_log.csv)"
    )
    parser.add_argument(
        "--smooth", action="store_true",
        help="Áp dụng lọc Savitzky-Golay cho tín hiệu góc và dịch chuyển"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Lưu tất cả biểu đồ ra file PNG (cùng thư mục với CSV)"
    )
    args = parser.parse_args()
    plot_hysteresis(args.csv, smooth=args.smooth, save=args.save)

