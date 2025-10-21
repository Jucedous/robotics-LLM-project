from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch

NODE_F = 9
EDGE_F = 6

def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@dataclass
class SceneCfg:
    n_nodes: int = 2
    box_xy: float = 1.0
    z_range: Tuple[float,float] = (0.0, 0.6)
    r_xy: float = 0.25
    z_over: float = 0.10
    p_liquid_cup: float = 1.0
    p_noise: float = 0.0

def sample_scene(cfg: SceneCfg):
    N = cfg.n_nodes
    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.zeros((N, 3), dtype=np.float32)
    pos[1, 0] = np.random.uniform(-0.1, 0.1)
    pos[1, 1] = np.random.uniform(-0.1, 0.1)
    pos[1, 2] = np.random.uniform(0.02, 0.05)
    pos[0, 0] = np.random.uniform(-cfg.box_xy, cfg.box_xy)
    pos[0, 1] = np.random.uniform(-cfg.box_xy, cfg.box_xy)
    pos[0, 2] = np.random.uniform(cfg.z_range[0], cfg.z_range[1])
    is_liquid = 1.0 if (np.random.rand() < cfg.p_liquid_cup) else 0.0
    is_electronic = 1.0
    nodes = np.zeros((N, NODE_F), dtype=np.float32)
    nodes[0, 0:3] = pos[0]
    nodes[0, 3:6] = vel[0]
    nodes[0, 6] = is_liquid
    nodes[0, 7] = 0.0
    nodes[0, 8] = 0.0
    nodes[1, 0:3] = pos[1]
    nodes[1, 3:6] = vel[1]
    nodes[1, 6] = 0.0
    nodes[1, 7] = is_electronic
    nodes[1, 8] = 0.0
    edges = np.zeros((N, N, EDGE_F), dtype=np.float32)
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx, dy, dz = pos[j] - pos[i]
            dist_xy = math.hypot(dx, dy)
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            is_above = 1.0 if dz > 0 else 0.0
            edges[i, j, :] = [dx, dy, dz, dist_xy, dist, is_above]
            adj[i, j] = 1.0
    dx, dy, dz = pos[1] - pos[0]
    dist_xy = math.hypot(dx, dy)
    hazard = (is_liquid > 0.5) and (dist_xy < cfg.r_xy) and ((pos[0,2] - pos[1,2]) > cfg.z_over)
    y = 1.0 if hazard else 0.0
    if np.random.rand() < cfg.p_noise:
        y = 1.0 - y
    return (
        torch.from_numpy(nodes),
        torch.from_numpy(edges),
        torch.from_numpy(adj),
        torch.tensor([y], dtype=torch.float32),
    )

def make_batch(cfg: SceneCfg, batch_size: int):
    node_list, edge_list, adj_list, y_list = [], [], [], []
    for _ in range(batch_size):
        n, e, a, y = sample_scene(cfg)
        node_list.append(n)
        edge_list.append(e)
        adj_list.append(a)
        y_list.append(y)
    nodes = torch.stack(node_list, dim=0)
    edges = torch.stack(edge_list, dim=0)
    adjs  = torch.stack(adj_list, dim=0)
    ys    = torch.cat(y_list, dim=0)
    return nodes, edges, adjs, ys
