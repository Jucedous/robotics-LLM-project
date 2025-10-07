from __future__ import annotations
import math
from typing import Tuple

Vec3 = Tuple[float, float, float]

def _dist(a: Vec3, b: Vec3) -> float:
    return math.dist(a, b)

def _planar_dist(a: Vec3, b: Vec3) -> float:
    ax, ay, _ = a
    bx, by, _ = b
    return math.hypot(ax - bx, ay - by)

def distance_barrier(a: Vec3, b: Vec3, d_min: float) -> float:
    return _dist(a, b) - d_min

def above_barrier(a: Vec3, b: Vec3, z_min: float, r_align: float = 0.2) -> float:
    ax, ay, az = a
    bx, by, bz = b
    planar = _planar_dist(a, b)
    if planar > r_align:
        return (az - bz) - z_min + (planar - r_align)
    return (az - bz) - z_min


def closing_speed_barrier(pos_a: Vec3, vel_a: Vec3, pos_b: Vec3, v_max: float) -> float:
    ax, ay, az = pos_a
    bx, by, bz = pos_b
    vx, vy, vz = vel_a
    dx, dy, dz = (bx - ax), (by - ay), (bz - az)
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < 1e-6:
        return v_max
    nx, ny, nz = dx/dist, dy/dist, dz/dist
    approach_speed = vx*nx + vy*ny + vz*nz
    return v_max - max(0.0, approach_speed)



def barrier_to_risk(h: float, scale: float = 0.1) -> float:
    return 1.0 / (1.0 + math.exp(h/scale))