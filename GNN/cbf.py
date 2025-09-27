import torch

def cbf_weighted_distance(cup_xyz, laptop_xyz, alpha_xy, alpha_z, r_xy=0.25, z_over=0.10):
    cup_xyz = torch.tensor(cup_xyz, dtype=torch.float32)
    lap_xyz = torch.tensor(laptop_xyz, dtype=torch.float32)
    dx, dy = cup_xyz[0]-lap_xyz[0], cup_xyz[1]-lap_xyz[1]
    dz = cup_xyz[2]-lap_xyz[2]
    dxy = torch.sqrt(dx*dx + dy*dy) + 1e-6
    over = torch.relu(dz - z_over)
    margin = (dxy - r_xy) - (alpha_xy * torch.exp(-5.0*dxy)) * (alpha_z * torch.tanh(5.0*over))
    return margin
