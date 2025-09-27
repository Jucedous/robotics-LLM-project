# python3 predict.py --ckpt checkpoints/hazard_gnn.pt --cup 0.00 0.00 0.30 --laptop 0.02 0.01 0.05

import argparse
import torch
from model import HazardGNN
from eval import eval_one

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default='checkpoints/hazard_gnn.pt')
    ap.add_argument('--cup', type=float, nargs=3, default=[0.0,0.0,0.30])
    ap.add_argument('--laptop', type=float, nargs=3, default=[0.02,0.01,0.05])
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    hidden = ckpt.get('hidden', 64)
    model = HazardGNN(hidden=hidden, msg_hidden=hidden)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    out = eval_one(model, tuple(args.cup), tuple(args.laptop))
    print(f"hazard={out['hazard_score']:.3f}, alpha_xy={out['alpha_xy']:.3f}, alpha_z={out['alpha_z']:.3f}")