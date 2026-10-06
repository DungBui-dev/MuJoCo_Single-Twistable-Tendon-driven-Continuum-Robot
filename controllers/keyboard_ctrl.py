import threading
import numpy as np
import mujoco

# WASD step sizes
CTRL_STEP_PP_DEFAULT = 0.2    # push-pull step [N]
CTRL_STEP_TW_DEFAULT = 0.05   # twist step [N.m/motor]
CTRL_MAX_PP          = 5.0    # push-pull max [N]
CTRL_MAX_TW          = 0.5    # twist max [N.m]
STEP_MIN             = 0.01   # min adjustable step
STEP_MAX             = 1.0    # max adjustable step

class KeyboardController:
    """
    Thu thap phim ban phim qua key_callback (GLFW thread) va ap dung
    vao ctrl trong simulation thread chinh (sim loop).

    Cach hoat dong:
      - key_callback() duoc goi tu GLFW thread: chi ghi vao queue
      - flush_pending() duoc goi tu sim loop: xu ly queue va update ctrl
      - Cach nay tranh xung dot Slider / lag khi nhan phim nhanh

    Reset dong bo:
      - Khi nhan R: dat _reset_flag = True
      - flush_pending() se ep qpos/ctrl/qvel ve 0 ngay lap tuc
    """

    def __init__(self, step_pp: float = CTRL_STEP_PP_DEFAULT,
                       step_tw: float = CTRL_STEP_TW_DEFAULT):
        self.ctrl      = np.zeros(2)   # [push_pull_N, twist_Nm]
        self.step_pp   = step_pp
        self.step_tw   = step_tw

        self._lock        = threading.Lock()
        self._key_queue   = []        # danh sach keycode cho xu ly
        self._reset_flag  = False     # flag reset toan bo

        # GLFW key codes
        self._KEY_MAP = {
            87: "W",  119: "W",   # W/w = pull
            83: "S",  115: "S",   # S/s = push
            65: "A",   97: "A",   # A/a = twist left
            68: "D",  100: "D",   # D/d = twist right
            82: "R",  114: "R",   # R/r = reset
            81: "Q",  113: "Q",   # Q/q = release pp
            69: "E",  101: "E",   # E/e = release tw
            91: "[",              # [ = step decrease
            93: "]",              # ] = step increase
        }

    def key_callback(self, keycode: int):
        """Duoc goi tu GLFW thread -- CHI GHI vao queue, khong xu ly."""
        if keycode in self._KEY_MAP:
            with self._lock:
                self._key_queue.append(keycode)

    def flush_pending(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Goi moi sim step: xu ly tat ca phim pending va cap nhat ctrl.
        Dam bao reset dong bo ngay lap tuc.
        """
        with self._lock:
            keys = list(self._key_queue)
            self._key_queue.clear()
            do_reset = self._reset_flag
            self._reset_flag = False

        if do_reset:
            self._do_reset(model, data)
            return

        for keycode in keys:
            action = self._KEY_MAP.get(keycode, "")
            self._apply_action(action)

        # Luon cap nhat ctrl (kể ca khi khong co phim moi)
        data.ctrl[0] = self.ctrl[0]
        if len(data.ctrl) > 1:
            data.ctrl[1:] = self.ctrl[1]   # broadcast twist to all m motors

    def _apply_action(self, action: str):
        """Xu ly mot action cu cu the."""
        pp, tw = self.ctrl[0], self.ctrl[1]
        spp, stw = self.step_pp, self.step_tw

        if action == "W":
            pp = max(-CTRL_MAX_PP, pp - spp)
        elif action == "S":
            pp = min(CTRL_MAX_PP, pp + spp)
        elif action == "A":
            tw = max(-CTRL_MAX_TW, tw - stw)
        elif action == "D":
            tw = min(CTRL_MAX_TW, tw + stw)
        elif action == "R":
            pp, tw = 0.0, 0.0
            with self._lock:
                self._reset_flag = True
        elif action == "Q":
            sign = np.sign(pp) if abs(pp) > 1e-9 else 0.0
            pp = max(0.0, abs(pp) - spp) * sign
        elif action == "E":
            sign = np.sign(tw) if abs(tw) > 1e-9 else 0.0
            tw = max(0.0, abs(tw) - stw) * sign
        elif action == "[":
            self.step_pp = max(STEP_MIN, round(self.step_pp * 0.5, 3))
            self.step_tw = max(STEP_MIN, round(self.step_tw * 0.5, 3))
            print(f"  [Step] PP={self.step_pp:.3f}  TW={self.step_tw:.3f}")
        elif action == "]":
            self.step_pp = min(STEP_MAX, round(self.step_pp * 2.0, 3))
            self.step_tw = min(STEP_MAX, round(self.step_tw * 2.0, 3))
            print(f"  [Step] PP={self.step_pp:.3f}  TW={self.step_tw:.3f}")

        self.ctrl[0] = pp
        self.ctrl[1] = tw
        if action in ("W", "S", "A", "D", "Q", "E", "R"):
            print(
                f"  [Ctrl] PP={self.ctrl[0]:+.2f}N  TW={self.ctrl[1]:+.2f}Nm"
                f"  (key={action})"
            )

    def _do_reset(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Reset dong bo toan bo: ep qpos=0, ctrl=0, qvel=0.
        Duoc goi trong sim loop (main thread) -- an toan voi MuJoCo.
        """
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        self.ctrl[:] = 0.0
        data.ctrl[:] = 0.0
        print("  [RESET] All states cleared -> qpos=0, ctrl=0, vel=0")

