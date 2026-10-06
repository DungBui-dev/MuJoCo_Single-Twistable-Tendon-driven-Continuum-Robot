"""
utils/kinematics.py  ── v1.0.2
================================
Dong hoc thuan (FK) va Jacobian cho Single Twistable Tendon-Driven Continuum Robot.

v1.0.2: Viet lai theo dung toan hoc tu MATLAB cua bai bao.

Nguon tham chieu chinh:
  [MATLAB] fk.m           - Forward kinematics, 3 thanh phan notch transform
  [MATLAB] pcc.m          - Curvature tu vat ly vat lieu (NiTi mechanics)
  [MATLAB] compJacobian.m - Numeric Jacobian (central difference, 6x4)
  [MATLAB] loadRobotParams.m - Geometric parameters

Kien truc FK (theo fk.m):
  T_robot[1] = T_base = [I | [0, 0, h + baseMotion]]
  for i = 2..n:
    if i mod 2 == 0 (notch):
      T_notch = T_Y_bend * T_Z_twist * T_X_bend
      T_robot[i] = T_robot[i-1] * T_notch
    else (uncut):
      T_robot[i] = T_robot[i-1] * [I | [0,0,h]]
  T_tip = T_robot[n] * T_uncut

Moi T_notch gom 3 thanh phan:
  T_X_bend  - uon quanh X (Push-Pull tendon, curvature tu pcc.m)
  T_Z_twist - xoan quanh Z + dich tam y_c (eccentric inner channel)
  T_Y_bend  - uon cam ung boi xoan (coupling Ky)
"""

from __future__ import annotations

import numpy as np
import mujoco

# ─────────────────────────────────────────────────────────────────────────────
# Utility: floating-point noise cleanup
# ─────────────────────────────────────────────────────────────────────────────

_CLEAN_EPS = 1e-9   # nguong de zero-out noise (~1 nm precision)
_ROUND_DEC = 6      # so chu so thap phan khi round


def _clean(arr: np.ndarray) -> np.ndarray:
    """
    Lam sach nhieu floating-point truoc khi tra ve UI/CSV.
    Gia tri < _CLEAN_EPS se bi eplaced bang 0 chinh xac.
    """
    out = np.where(np.abs(arr) < _CLEAN_EPS, 0.0, arr)
    return np.round(out, _ROUND_DEC)


def _clean_scalar(x: float) -> float:
    if abs(x) < _CLEAN_EPS:
        return 0.0
    return round(float(x), _ROUND_DEC)


# ─────────────────────────────────────────────────────────────────────────────
# Rotation matrices (khop voi MATLAB rotx/roty/rotz trong fk.m)
# ─────────────────────────────────────────────────────────────────────────────

_EPS = np.finfo(float).eps   # MATLAB eps tuong duong


def _rotx(theta: float) -> np.ndarray:
    """Quay quanh X (radian). Tuong duong rotx() trong fk.m."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0,   c,  -s],
                  [0.0,   s,   c]])
    R[np.abs(R) < _EPS] = 0.0
    return R


def _roty(theta: float) -> np.ndarray:
    """Quay quanh Y (radian). Tuong duong roty() trong fk.m."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[  c, 0.0,  s],
                  [0.0, 1.0, 0.0],
                  [ -s, 0.0,  c]])
    R[np.abs(R) < _EPS] = 0.0
    return R


def _rotz(theta: float) -> np.ndarray:
    """Quay quanh Z (radian). Tuong duong rotz() trong fk.m."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[  c, -s, 0.0],
                  [  s,  c, 0.0],
                  [0.0, 0.0, 1.0]])
    R[np.abs(R) < _EPS] = 0.0
    return R


def _make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Tao ma tran SE(3) 4x4 tu R(3x3) va t(3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


# ─────────────────────────────────────────────────────────────────────────────
# SoroTwistFK: Forward Kinematics chinh xac theo fk.m + pcc.m
# ─────────────────────────────────────────────────────────────────────────────

class SoroTwistFK:
    """
    Forward Kinematics cua Single Twistable Tendon-Driven Continuum Robot.
    Triển khai Python chinh xac tu MATLAB fk.m + pcc.m cua bai bao.

    Thong so hinh hoc (Robot 2: OD=3.5mm, ID=1.8mm — theo pcc.m/loadRobotParams.m):
      r_o = 1.75 mm  (ban kinh ngoai)
      r_i = 0.90 mm  (ban kinh trong, ID=1.8mm)
      g   = 2.5  mm  (chieu sau cat ranh)
      p   = 0.4375 mm (offset kenh lech tam)
      h   = 1.5  mm  (chieu cao mot notch)
      d_t = 1.04 mm  (khoang cach tendon tu centroid)

    Bien dieu khien q = [pushpull_mm, twist_rad, base_mm]:
      q[0]: do dich chuyen gan (push < 0, pull > 0) [mm]
      q[1]: goc xoan [rad]
      q[2]: dich chuyen de [mm] (mac dinh 0)

    Su dung:
      fk_obj = SoroTwistFK()
      T_list = fk_obj.fk(q=[1.0, -1.5708, 0.0])  # 1mm pull, -90deg twist
      T_tip = T_list[-1]   # ma tran SE(3) 4x4 cua dau mut
      pos_mm = T_tip[:3, 3]  # vi tri [mm]
    """

    # ── Thong so hinh hoc (mm) — theo loadRobotParams.m + pcc.m ──────────
    r_o = 3.5 / 2          # [mm] outer radius
    r_i = 1.8 / 2          # [mm] inner radius  (ID=1.8mm theo pcc.m)
    g   = 2.5              # [mm] cutting depth
    p   = 0.4375           # [mm] eccentric channel offset
    h   = 1.5              # [mm] notch height (uncut height)
    d_t = 1.04             # [mm] tendon offset from centroid

    # ── Thong so vat lieu (SI: Pa, m) — theo pcc.m ────────────────────────
    E_t   = 50e9           # [Pa] Young's modulus tendon (NiTi wire)
    E_s   = 6e9            # [Pa] Young's modulus structure (NiTi tube)
    k_push = 0.65          # He so giam curvature khi day (push, delta_l < 0)
    r_t   = 0.15e-3        # [m]  ban kinh day gan
    L_ten = 255e-3         # [m]  chieu dai gan (tip → anchoring point)

    def __init__(self):
        """Tinh toan cac thong so hinh hoc mot lan."""
        r_o, r_i, g, p = self.r_o, self.r_i, self.g, self.p

        # --- Eccentric channel centroid (fk.m line 21) ---
        self.y_c = -p * np.pi * r_i**2 / (np.pi * (r_o**2 - r_i**2))

        # --- Neutral Bending Plane geometry (pcc.m lines 37-46) ---
        phi_o = 2 * np.arccos((g - r_o) / r_o)
        phi_i = 2 * np.arccos((g - r_o - p) / r_i)

        A_o = 0.5 * r_o**2 * (phi_o - np.sin(phi_o))
        A_i = 0.5 * r_i**2 * (phi_i - np.sin(phi_i))

        y_o = ((2 * r_o**3) / (3 * A_o)) * (np.sin(0.5 * phi_o))**3
        y_i = ((2 * r_i**3) / (3 * A_i)) * (np.sin(0.5 * phi_i))**3 + p

        self.y_bar = (y_o * A_o - y_i * A_i) / (A_o - A_i)

        # --- Area Moment of Inertia (pcc.m lines 49-52) ---
        # Convert mm to m for SI calculation
        r_o_m = r_o * 1e-3
        r_i_m = r_i * 1e-3
        p_m   = p   * 1e-3
        y_c_m = self.y_c * 1e-3
        y_bar_m = self.y_bar * 1e-3

        I_s = (
            (np.pi / 4 * r_o_m**4 + np.pi * r_o_m**2 * y_c_m**2)
            - (np.pi / 4 * r_i_m**4 + np.pi * r_i_m**2 * (p_m - y_c_m)**2)
        )
        I_s += np.pi * (r_o_m**2 - r_i_m**2) * (y_bar_m - y_c_m)**2
        self.I_s = I_s

        # Tendon cross-section
        self.A_t = np.pi * self.r_t**2

    # ── pcc.m: tinh curvature tu do dich chuyen gan ────────────────────────

    def pcc_curvature(self, delta_l_mm: float) -> float:
        """
        Tinh curvature kappa [1/mm] tai centroid frame tu do dich chuyen gan.
        Port chinh xac tu pcc.m.

        Tham so:
          delta_l_mm: do dich chuyen gan cuc bo [mm] (= displ*2/n)
                      > 0: pull (uon cong)
                      < 0: push (thang lai)
        """
        delta_l = delta_l_mm * 1e-3   # mm → m

        d_t_m    = self.d_t   * 1e-3
        y_bar_m  = self.y_bar * 1e-3

        # Bending moment (pcc.m lines 55-59)
        if delta_l >= 0:
            M = (d_t_m + y_bar_m) * (self.E_t * self.A_t * delta_l) / self.L_ten
        else:
            M = self.k_push * (d_t_m + y_bar_m) * (
                self.E_t * self.A_t * delta_l) / self.L_ten

        # Curvature at NBP, then convert to centroid (pcc.m lines 62-64)
        kappa_nbp = M / (self.E_s * self.I_s)   # [1/m]
        kappa_c   = kappa_nbp / (1 - kappa_nbp * y_bar_m)  # centroid

        return kappa_c * 1e-3   # [1/m] → [1/mm]

    # ── fk.m: T_notch transform tai mot notch ─────────────────────────────

    def _notch_tf(
        self,
        i: int,
        n: int,
        twist: float,
        delta_l: float,
        Ky: float,
        Kx: float,
    ) -> np.ndarray:
        """
        Tinh ma tran bien doi SE(3) cho notch thu i.
        Port chinh xac tu notch_tf() trong fk.m.

        Tham so:
          i       : chi so notch (1-indexed, chi notch chan)
          n       : tong so notch
          twist   : goc xoan toan bo [rad]
          delta_l : do dich chuyen gan cuc bo [mm] (= displ*2/n)
          Ky      : he so coupling xoan→uon Y (mac dinh 0.02)
          Kx      : he so coupling xoan→uon X (mac dinh 0.005)

        Tra ve:
          T_notch : ma tran SE(3) 4x4 [don vi mm cho translation]
        """
        h   = self.h
        y_c = self.y_c
        y_bar = self.y_bar

        # --- Twist angle within each notch (fk.m lines 62-63) ---
        local_twist     = twist * 2 / n
        sum_local_twist = twist * i / n

        # ── T_Z_twist: Xoan quanh truc Z + dich tam y_c (fk.m lines 67-72) ──
        Z_twist_Px = -y_c * np.sin(local_twist)
        Z_twist_Py =  y_c * (np.cos(local_twist) - 1)

        T_Z = np.eye(4)
        T_Z[:3, :3] = _rotz(-local_twist)
        T_Z[:3,  3] = [Z_twist_Px, Z_twist_Py, 0.0]

        # ── T_Y_bend: Uon cam ung boi xoan quanh truc Y (fk.m lines 77-83) ──
        Y_bend_angle = np.sign(twist) * Ky * sum_local_twist
        Y_bend_kappa = Y_bend_angle / h

        if abs(Y_bend_kappa) < 1e-9:
            Y_bend_Px = _EPS
        else:
            Y_bend_Px = (-1 + np.cos(Y_bend_angle)) / (Y_bend_kappa + _EPS)

        T_Y = np.eye(4)
        T_Y[:3, :3] = _roty(Y_bend_angle)
        T_Y[:3,  3] = [Y_bend_Px, 0.0, 0.0]

        # ── T_X_bend: Uon chinh quanh truc X (fk.m lines 87-97) ──────────────
        X_kappa = self.pcc_curvature(delta_l)                      # base curvature
        X_kappa += np.sign(twist) * Kx * sum_local_twist           # + twist coupling

        X_s = h / (1 + X_kappa * y_bar)

        if abs(X_kappa) < 1e-12:
            # Gioi han khi kappa → 0: duong thang
            X_bend_Py = 0.0
            X_bend_Pz = X_s
        else:
            X_bend_Py = (-1 + np.cos(X_s * X_kappa)) / X_kappa
            X_bend_Pz =  np.sin(X_s * X_kappa) / X_kappa

        T_X = np.eye(4)
        T_X[:3, :3] = _rotx(X_s * X_kappa)
        T_X[:3,  3] = [0.0, X_bend_Py, X_bend_Pz]

        # ── Ghep cac thanh phan: T_notch = T_Y * T_Z * T_X (fk.m line 100) ──
        T_notch = T_Y @ T_Z @ T_X
        return T_notch

    # ── Giao dien chinh: fk() ─────────────────────────────────────────────

    def fk(
        self,
        q: list | np.ndarray,
        n: int = 32,
        Ky: float = 0.02,
        Kx: float = 0.005,
    ) -> list[np.ndarray]:
        """
        Dong hoc thuan PPT theo fk.m chinh xac.

        Tham so:
          q   : [pushpull_mm, twist_rad, base_mm]
                pushpull > 0 = pull (uon), < 0 = push (thang)
          n   : so notch (mac dinh 32, Robot 2 dung 16 nhung FK theo paper 32)
          Ky  : coupling gain xoan→uon Y (mac dinh 0.02, tu visRobot.m)
          Kx  : coupling gain xoan→uon X (mac dinh 0.005, tu fk.m)

        Tra ve:
          T_list : danh sach n+1 ma tran SE(3) 4x4 [mm translation]
                   T_list[0] = T_base
                   T_list[-1] = T_tip = T_robot[n] * T_uncut
        """
        q = np.asarray(q, dtype=float)
        twist    = q[1]
        base     = q[2] if len(q) > 2 else 0.0
        displ    = max(_EPS, q[0])           # [mm] tong do dich chuyen
        delta_l  = displ * 2.0 / n          # [mm] cuc bo moi notch

        h = self.h
        T_uncut = _make_T(np.eye(3), np.array([0.0, 0.0, h]))

        # --- T_base (fk.m line 38) ---
        T_base = _make_T(np.eye(3), np.array([0.0, 0.0, h + base]))

        T_list: list[np.ndarray] = [None] * (n + 2)
        T_list[1] = T_base

        for i in range(2, n + 1):
            if i % 2 == 0:
                # Notch (cut section)
                T_notch = self._notch_tf(i, n, twist, delta_l, Ky, Kx)
                T_list[i] = T_list[i - 1] @ T_notch
                # Chuan bi T_uncut cho section tiep theo
                T_uncut_local = _make_T(np.eye(3), np.array([0.0, 0.0, h]))
            else:
                # Uncut section
                T_list[i] = T_list[i - 1] @ T_uncut

        # T_tip = T_robot[n] * T_uncut (fk.m line 56)
        T_list[n + 1] = T_list[n] @ _make_T(np.eye(3), np.array([0.0, 0.0, h]))

        # Xoa phan tu None dau danh sach
        return [T for T in T_list if T is not None]

    def get_tip_transform(self, q, n=32, Ky=0.02, Kx=0.005) -> np.ndarray:
        """Lay T_tip (4x4) truc tiep."""
        return self.fk(q, n, Ky, Kx)[-1]

    def get_tip_pos_mm(self, q, n=32, Ky=0.02, Kx=0.005) -> np.ndarray:
        """Lay vi tri dau mut [mm], da lam sach floating-point noise."""
        T = self.get_tip_transform(q, n, Ky, Kx)
        return _clean(T[:3, 3])

    def numeric_jacobian(
        self,
        q: np.ndarray,
        n: int = 32,
        Ky: float = 0.02,
        Kx: float = 0.005,
        perturb: float = 1e-6,
    ) -> np.ndarray:
        """
        Jacobian so (central difference) 6x3.
        Port chinh xac tu compJacobian.m.

        Tra ve J (6, 3):
          J[0:3, :] - Jacobian tinh tien (mm/mm hoac mm/rad)
          J[3:6, :] - Jacobian quay (rad/mm hoac rad/rad)

        q = [pushpull_mm, twist_rad, base_mm]
        """
        q = np.asarray(q, dtype=float)
        n_q = len(q)
        J = np.zeros((6, n_q))

        T0 = self.get_tip_transform(q, n, Ky, Kx)
        R0 = T0[:3, :3]

        for i in range(n_q):
            q_plus  = q.copy(); q_plus[i]  += perturb
            q_minus = q.copy(); q_minus[i] -= perturb

            T_p = self.get_tip_transform(q_plus,  n, Ky, Kx)
            T_m = self.get_tip_transform(q_minus, n, Ky, Kx)

            # Linear velocity (compJacobian.m line 36)
            J[0:3, i] = (T_p[:3, 3] - T_m[:3, 3]) / (2 * perturb)

            # Angular velocity (compJacobian.m lines 39-41)
            delta_R = (T_p[:3, :3] - T_m[:3, :3]) / (2 * perturb)
            S = delta_R @ R0.T   # Skew-symmetric matrix
            J[3, i] = S[2, 1]    # omega_x
            J[4, i] = S[0, 2]    # omega_y
            J[5, i] = S[1, 0]    # omega_z

        return J


# ─────────────────────────────────────────────────────────────────────────────
# Singleton FK instance (dung chung trong toan project)
# ─────────────────────────────────────────────────────────────────────────────

_FK = SoroTwistFK()


def get_sorotwist_fk() -> SoroTwistFK:
    """Lay instance SoroTwistFK singleton."""
    return _FK


# ─────────────────────────────────────────────────────────────────────────────
# 1. MuJoCo Native FK (giu nguyen tu v1.0.1, them _clean)
# ─────────────────────────────────────────────────────────────────────────────

def get_tip_frame_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str = "tip_site",
) -> dict:
    """
    Lay pose cua tip tu MuJoCo FK noi bo (chinh xac nhat).
    v1.0.2: ap dung _clean() de loai bo floating-point noise.

    Tra ve dict:
      pos    : (3,)  vi tri tip [m] trong world frame
      rotmat : (3,3) ma tran quay
      quat   : (4,)  quaternion [w, x, y, z]
    """
    try:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise KeyError(f"Site '{site_name}' not found")
        pos    = _clean(data.site_xpos[site_id].copy())
        rotmat = _clean(data.site_xmat[site_id].reshape(3, 3).copy())
        quat   = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotmat.flatten())
        return {"pos": pos, "rotmat": rotmat, "quat": quat}
    except Exception:
        return {
            "pos":    np.zeros(3),
            "rotmat": np.eye(3),
            "quat":   np.array([1., 0., 0., 0.]),
        }


def get_tip_euler_xyz(rotmat: np.ndarray) -> np.ndarray:
    """
    Trich goc Euler ZYX (roll_x, pitch_y, yaw_z) tu ma tran quay [do].
    v1.0.2: ap dung _clean() cho ket qua.
    """
    R = rotmat
    pitch = np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return _clean(np.array([roll, pitch, yaw]))


# ─────────────────────────────────────────────────────────────────────────────
# 2. PCC FK (giu nguyen tu v1.0.1, them _clean)
# ─────────────────────────────────────────────────────────────────────────────

def _Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0,  0], [0, c, -s], [0, s,  c]])


def _Rz(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[ c, -s, 0], [ s,  c, 0], [ 0,  0, 1]])


def compute_pcc_fk(
    bend_angles_rad: np.ndarray,
    twist_angles_rad: np.ndarray,
    segment_length: float,
) -> dict:
    """
    Dong hoc thuan PCC don gian (Piecewise Constant Curvature).
    Giu tu v1.0.1 de dung cho validation / so sanh voi MuJoCo.
    v1.0.2: them _clean() cho output.
    """
    T_total = np.eye(4)
    for theta, phi in zip(bend_angles_rad, twist_angles_rad):
        R_seg = _Rx(theta) @ _Rz(phi)
        t_seg = np.array([0.0, 0.0, segment_length])
        T_seg = np.eye(4)
        T_seg[:3, :3] = R_seg
        T_seg[:3,  3] = t_seg
        T_total = T_total @ T_seg

    pos    = _clean(T_total[:3, 3])
    rotmat = _clean(T_total[:3, :3])
    quat   = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotmat.flatten())
    return {"pos": pos, "rotmat": rotmat, "quat": quat, "T": T_total}


def validate_fk(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    segment_length: float,
    verbose: bool = False,
) -> dict:
    """So sanh FK cua MuJoCo voi PCC analytical."""
    mj_frame = get_tip_frame_from_mujoco(model, data)
    mj_pos   = mj_frame["pos"]
    mj_quat  = mj_frame["quat"]

    n_jnt  = model.njnt
    bends  = []
    twists = []
    for i in range(n_jnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name.startswith("bend_x_"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            bends.append(data.qpos[model.jnt_qposadr[jid]])
        elif name and name.startswith("twist_z_"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            twists.append(data.qpos[model.jnt_qposadr[jid]])

    pcc_result = compute_pcc_fk(np.array(bends), np.array(twists), segment_length)
    pcc_pos    = pcc_result["pos"]
    pcc_quat   = pcc_result["quat"]

    pos_err   = np.linalg.norm(mj_pos - pcc_pos) * 1000
    dot       = np.clip(abs(np.dot(mj_quat, pcc_quat)), 0, 1)
    angle_err = np.degrees(2 * np.arccos(dot))

    if verbose:
        print(f"  MuJoCo  tip: [{mj_pos[0]*1e3:.2f}, {mj_pos[1]*1e3:.2f}, {mj_pos[2]*1e3:.2f}] mm")
        print(f"  PCC     tip: [{pcc_pos[0]*1e3:.2f}, {pcc_pos[1]*1e3:.2f}, {pcc_pos[2]*1e3:.2f}] mm")
        print(f"  Pos err: {pos_err:.3f} mm | Angle err: {angle_err:.3f} deg")

    return {
        "pos_error_mm":    pos_err,
        "angle_error_deg": angle_err,
        "mujoco_pos":      mj_pos,
        "pcc_pos":         pcc_pos,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Jacobian (MuJoCo native, giu nguyen)
# ─────────────────────────────────────────────────────────────────────────────

def compute_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str = "tip_site",
) -> tuple[np.ndarray, np.ndarray]:
    """Jacobian tinh tien va quay cua site tip (MuJoCo mj_jacSite)."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    if site_id >= 0:
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp, jacr


def get_full_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str = "tip_site",
) -> np.ndarray:
    """Jacobian 6xnv day du (tinh tien + quay)."""
    jacp, jacr = compute_jacobian(model, data, site_name)
    return np.vstack([jacp, jacr])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Utility geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_tip_bending_angle_deg(
    tip_pos: np.ndarray,
    base_pos: np.ndarray | None = None,
) -> float:
    """Goc uon tong the cua dau mut [do]. v1.0.2: _clean applied."""
    if base_pos is None:
        base_pos = np.zeros(3)
    v = _clean(tip_pos) - _clean(base_pos)
    length = np.linalg.norm(v)
    if length < 1e-10:
        return 0.0
    z_comp = np.clip(v[2] / length, -1.0, 1.0)
    return float(np.degrees(np.arccos(z_comp)))


def compute_tip_bending_plane_angle_deg(tip_pos: np.ndarray) -> float:
    """Goc cua mat phang uon so voi truc X (trong mat phang XY) [do]."""
    xy = _clean(tip_pos)[:2]
    if np.linalg.norm(xy) < 1e-10:
        return 0.0
    return float(np.degrees(np.arctan2(xy[1], xy[0])))
