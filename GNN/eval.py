import math
import torch
from data import NODE_F, EDGE_F

def _to_device(t, device):
    return t.to(device) if isinstance(t, torch.Tensor) else t

def eval_one(model, cup_xyz, laptop_xyz):
    device = next(model.parameters()).device
    nodes = torch.zeros((2, NODE_F), dtype=torch.float32, device=device)
    nodes[0, 0:3] = torch.tensor(cup_xyz, dtype=torch.float32, device=device)
    nodes[0, 6] = 1.0
    nodes[1, 0:3] = torch.tensor(laptop_xyz, dtype=torch.float32, device=device)
    nodes[1, 7] = 1.0
    edges = torch.zeros((2, 2, EDGE_F), dtype=torch.float32, device=device)
    adj = torch.zeros((2, 2), dtype=torch.float32, device=device)
    for i in range(2):
        for j in range(2):
            if i == j:
                continue
            dx = nodes[j, 0] - nodes[i, 0]
            dy = nodes[j, 1] - nodes[i, 1]
            dz = nodes[j, 2] - nodes[i, 2]
            dist_xy = torch.sqrt(dx*dx + dy*dy)
            dist = torch.sqrt(dx*dx + dy*dy + dz*dz)
            is_above = torch.tensor(1.0 if dz.item() > 0 else 0.0, device=device)
            edges[i, j, :] = torch.stack([dx, dy, dz, dist_xy, dist, is_above])
            adj[i, j] = 1.0
    with torch.no_grad():
        out = model(nodes, edges, adj)
        return {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}

def quick_sanity_checks(model):
    tests = [
        ((0.00, 0.00, 0.30), (0.02, 0.01, 0.05)),
        ((0.40, 0.40, 0.30), (0.00, 0.00, 0.05)),
        ((0.00, 0.00, 0.07), (0.00, 0.00, 0.05)),
        ((0.10, 0.10, 0.35), (-0.05, -0.05, 0.04)),
    ]
    for cup, lap in tests:
        out = eval_one(model, cup, lap)
        print(f"cup={cup} laptop={lap} -> hazard={out['hazard_score']:.3f}, alpha_xy={out['alpha_xy']:.3f}, alpha_z={out['alpha_z']:.3f}")
