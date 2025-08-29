"""
Visualize safety score as a water cup moves closer to a laptop (liquid→electronics hazard).
This script imports `cbf_safety_metrics.py` from the same folder.
It animates the composite risk and prints the current distance + safety grade.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from cbf_safety_metrics import (
    Scene, RobotState, LinkState, ObjectState, Workspace, JointLimits, Sphere,
    evaluate_scene_metrics
)

# -------------------------
# Scene setup (static robot, moving cup)
# -------------------------

# Robot with two links (kept modest and stationary for this test)
links = [
    LinkState("ee", Sphere(np.array([0.0, 0.0, 0.5]), 0.05), np.array([0.0, 0.0, 0.0])),
    LinkState("wrist", Sphere(np.array([-0.2, 0.0, 0.5]), 0.06), np.array([0.0, 0.0, 0.0])),
]
robot = RobotState(links=links, q=np.array([0.0, 0.5]), qd=np.array([0.0, 0.0]))

# Laptop fixed; cup will move toward it on a straight line in XY plane (same Z)
laptop = ObjectState("laptop", Sphere(np.array([0.5, 0.2, 0.75]), 0.10), kind="electronic")
# Start the cup far away along +X and move it toward laptop's X
cup_start = np.array([0.95, 0.2, 0.75])
cup_end   = np.array([0.50, 0.2, 0.75])  # center-to-center coincident at end (max violation)
water_cup = ObjectState("water_cup", Sphere(cup_start.copy(), 0.06), kind="liquid")

# Optional bystanders (human, knife) to keep metrics realistic but not dominating this test
human = ObjectState("human_1", Sphere(np.array([ 0.1,  0.0, 0.5 ]), 0.15), kind="human")
knife  = ObjectState("knife",   Sphere(np.array([ 0.1, -0.2, 0.5 ]), 0.02), kind="sharp")

workspace = Workspace(bounds=np.array([[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.5]]))
limits    = JointLimits(q_min=np.array([-1.5, -0.5]), q_max=np.array([1.5, 1.2]))

scene = Scene(robot=robot, objects=[human, laptop, water_cup, knife], workspace=workspace, joint_limits=limits)

# -------------------------
# Animation + scoring
# -------------------------

steps = 100  # number of animation steps
xs = []      # frame index (or use distance)
ys = []      # safety grade (1–5)

fig, ax = plt.subplots()
ax.set_title("Safety grade as cup approaches laptop (higher = safer)")
ax.set_xlabel("Frame")
ax.set_ylabel("Safety grade (1–5)")
ax.set_ylim(1.0, 5.0)
line, = ax.plot([], [], lw=2)
text_box = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

# Precompute vector for cup motion
motion_vec = (cup_end - cup_start)

# Helper: compute center distance between cup and laptop

def center_distance(a: Sphere, b: Sphere) -> float:
    d = a.center - b.center
    return float(np.sqrt(np.dot(d, d)))

# Animation update function

def update(frame_idx):
    t = frame_idx / (steps - 1)
    # Move cup linearly toward laptop
    water_cup.sphere.center[:] = cup_start + t * motion_vec

    # Evaluate safety metrics
    out = evaluate_scene_metrics(scene)
    grade = out["safety_grade_1_to_5"]
    dist  = center_distance(water_cup.sphere, laptop.sphere)

    xs.append(frame_idx)
    ys.append(grade)

    # Update line and text
    line.set_data(xs, ys)
    ax.set_xlim(0, max(1, frame_idx))
    text_box.set_text(
        f"Frame: {frame_idx}\n"
        f"Cup-Laptop center distance: {dist:.3f} m\n"
        f"Composite risk: {out['composite_risk']:.3f}\n"
        f"Safety grade: {grade:.2f}"
    )

    # Print a simple console log too
    print(f"frame={frame_idx:03d} dist={dist:.3f} grade={grade:.2f} risk={out['composite_risk']:.3f}")
    return line, text_box

ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=False, repeat=False)
plt.show()
