import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))
DATA_DIR = Path("/mnt/data")
if DATA_DIR.exists() and str(DATA_DIR) not in sys.path:
    sys.path.append(str(DATA_DIR))

import cbf_safety_metrics as CBF


def central_diff_vel(positions: np.ndarray, dt: float) -> np.ndarray:
    n = positions.shape[0]
    v = np.zeros_like(positions)
    if n == 1:
        return v
    for k in range(n):
        if k == 0:
            v[k] = (positions[1] - positions[0]) / dt
        elif k == n - 1:
            v[k] = (positions[-1] - positions[-2]) / dt
        else:
            v[k] = (positions[k+1] - positions[k-1]) / (2 * dt)
    return v


def simulate_linear_path(p_start, p_end, total_time, dt):
    times = np.arange(0.0, total_time + 1e-9, dt)
    positions = np.linspace(p_start, p_end, times.shape[0])
    velocities = central_diff_vel(positions, dt)
    deltas = positions - positions[0]
    distances = np.linalg.norm(deltas, axis=1)
    return times, positions, velocities, distances


def score_frame(p_liquid, v_liquid, p_laptop, r_liquid, r_laptop):
    water = CBF.ObjectState(
        name="human1",
        sphere=CBF.Sphere(center=p_liquid, radius=r_liquid),
        kind="human",
        velocity=v_liquid,
    )
    laptop = CBF.ObjectState(
        name="laptop",
        sphere=CBF.Sphere(center=p_laptop, radius=r_laptop),
        kind="electronic",
        velocity=np.zeros(3),
    )
    scene = CBF.Scene(
        objects=[water, laptop],
        workspace=CBF.Workspace(bounds=np.array([[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.5]])),
    )
    out = CBF.evaluate_scene_metrics(scene)
    return float(out.get("safety_score_0_to_5", 0.0))


class OneWindowDistanceDemo:
    def __init__(self):
        self.dt = 0.02
        self.T_fast = 0.6
        self.T_slow = 3.0

        self.y_fast, self.y_slow = +0.20, -0.20
        self.r_laptop, self.r_liquid = 0.10, 0.06

        self.p_laptop_fast = np.array([0.0, self.y_fast, 0.75])
        self.p_laptop_slow = np.array([0.0, self.y_slow, 0.75])

        self.p_start_fast = np.array([0.6, self.y_fast, 0.90])
        self.p_end_fast = np.array([0.05, self.y_fast, 0.90])

        self.p_start_slow = np.array([0.6, self.y_slow, 0.90])
        self.p_end_slow = np.array([0.05, self.y_slow, 0.90])

        self.t_fast, self.pos_fast, self.vel_fast, self.dist_fast = simulate_linear_path(
            self.p_start_fast, self.p_end_fast, self.T_fast, self.dt
        )
        self.t_slow, self.pos_slow, self.vel_slow, self.dist_slow = simulate_linear_path(
            self.p_start_slow, self.p_end_slow, self.T_slow, self.dt
        )
        self.n_steps = max(len(self.t_fast), len(self.t_slow))

        self.fig = plt.figure(figsize=(12, 8))
        try:
            self.fig.canvas.manager.set_window_title(
                "Velocity-aware Safety — Distance-based Scores + Controls"
            )
        except Exception:
            pass

        gs = GridSpec(2, 2, height_ratios=[2, 1], figure=self.fig)
        self.ax_motion = self.fig.add_subplot(gs[0, :])
        self.ax_score = self.fig.add_subplot(gs[1, :])

        self.ax_motion.set_title("Motion: A_fast (y=+0.2) & A_slow (y=-0.2)")
        self.ax_motion.set_xlim(-0.1, 0.8)
        self.ax_motion.set_ylim(-0.6, 0.6)
        self.ax_motion.set_aspect("equal", adjustable="box")

        self.ax_motion.add_patch(
            Circle((self.p_laptop_fast[0], self.p_laptop_fast[1]), self.r_laptop, fill=False)
        )
        self.ax_motion.add_patch(
            Circle((self.p_laptop_slow[0], self.p_laptop_slow[1]), self.r_laptop, fill=False)
        )

        self.cup_fast_artist = Circle((self.p_start_fast[0], self.p_start_fast[1]), self.r_liquid, alpha=0.5)
        self.cup_slow_artist = Circle((self.p_start_slow[0], self.p_start_slow[1]), self.r_liquid, alpha=0.5)
        self.ax_motion.add_patch(self.cup_fast_artist)
        self.ax_motion.add_patch(self.cup_slow_artist)

        self.ax_score.set_title("Safety score vs Distance — FAST vs SLOW")
        max_dist = float(max(self.dist_fast[-1], self.dist_slow[-1]))
        self.ax_score.set_xlim(0, max_dist)
        self.ax_score.set_ylim(0, 5.1)

        (self.line_fast,) = self.ax_score.plot([], [], label="Fast", color="tab:orange", linewidth=2.0)
        (self.line_slow,) = self.ax_score.plot([], [], label="Slow", color="tab:blue", linewidth=2.0)
        self.ax_score.legend()

        self.xs_fast, self.ys_fast = [], []
        self.xs_slow, self.ys_slow = [], []

        ax_btn1 = self.fig.add_axes([0.35, 0.02, 0.15, 0.05])
        ax_btn2 = self.fig.add_axes([0.55, 0.02, 0.15, 0.05])
        self.btn_pause = Button(ax_btn1, "Pause/Resume")
        self.btn_replay = Button(ax_btn2, "Replay")
        self.btn_pause.on_clicked(self.on_pause)
        self.btn_replay.on_clicked(self.on_replay)

        self.k = 0
        self.paused = False

        self.timer = self.fig.canvas.new_timer(interval=int(self.dt * 1000))
        self.timer.add_callback(self.on_timer)
        self.timer.start()

    def on_pause(self, event):
        self.paused = not self.paused

    def on_replay(self, event):
        self.k = 0
        self.xs_fast.clear()
        self.ys_fast.clear()
        self.xs_slow.clear()
        self.ys_slow.clear()
        self.line_fast.set_data([], [])
        self.line_slow.set_data([], [])
        self.cup_fast_artist.center = (self.p_start_fast[0], self.p_start_fast[1])
        self.cup_slow_artist.center = (self.p_start_slow[0], self.p_start_slow[1])
        self.fig.canvas.draw_idle()

    def on_timer(self):
        if self.paused or self.k >= self.n_steps:
            return

        if self.k < len(self.t_fast):
            score_f = score_frame(
                self.pos_fast[self.k], self.vel_fast[self.k], self.p_laptop_fast, self.r_liquid, self.r_laptop
            )
            self.xs_fast.append(float(self.dist_fast[self.k]))
            self.ys_fast.append(score_f)
            self.line_fast.set_data(self.xs_fast, self.ys_fast)
            self.cup_fast_artist.center = (self.pos_fast[self.k][0], self.pos_fast[self.k][1])

        if self.k < len(self.t_slow):
            score_s = score_frame(
                self.pos_slow[self.k], self.vel_slow[self.k], self.p_laptop_slow, self.r_liquid, self.r_laptop
            )
            self.xs_slow.append(float(self.dist_slow[self.k]))
            self.ys_slow.append(score_s)
            self.line_slow.set_data(self.xs_slow, self.ys_slow)
            self.cup_slow_artist.center = (self.pos_slow[self.k][0], self.pos_slow[self.k][1])

        self.fig.canvas.draw_idle()
        self.k += 1


def main():
    plt.ion()
    OneWindowDistanceDemo()
    plt.show(block=True)


if __name__ == "__main__":
    main()
