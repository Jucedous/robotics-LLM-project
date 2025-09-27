import os
import csv
import torch
import torch.nn as nn
from data import SceneCfg, make_batch, seed_all
from model import HazardGNN

def train(epochs=1000, batch=64, hidden=64, lr=3e-3, device='cpu', seed=0, log_every=100, save_path=None, log_csv=None):
    device = torch.device(device)
    seed_all(seed)
    cfg = SceneCfg(n_nodes=2, r_xy=0.25, z_over=0.10, p_liquid_cup=1.0)
    model = HazardGNN(hidden=hidden, msg_hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    bce = nn.BCELoss()

    csv_writer = None
    csv_fh = None
    if log_csv:
        os.makedirs(os.path.dirname(log_csv) or '.', exist_ok=True)
        csv_fh = open(log_csv, 'w', newline='')
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(['step','loss','acc','alpha_xy','alpha_z'])

    for step in range(1, epochs + 1):
        nodes, edges, adjs, ys = make_batch(cfg, batch)
        nodes, edges, adjs, ys = nodes.to(device), edges.to(device), adjs.to(device), ys.to(device)
        out = model(nodes, edges, adjs)
        loss_cls = bce(out['hazard_score'], ys)
        loss_alpha = 1e-3 * (out['alpha_xy'].mean() + out['alpha_z'].mean())
        loss = loss_cls + loss_alpha
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % max(1, log_every) == 0:
            with torch.no_grad():
                pred = (out['hazard_score'] > 0.5).float()
                acc = (pred == ys).float().mean().item()
                ax = out['alpha_xy'].mean().item()
                az = out['alpha_z'].mean().item()
                print(f"step {step:5d} | loss {loss.item():.4f} | acc {acc:.3f} | alpha_xy {ax:.3f} | alpha_z {az:.3f}")
                if csv_writer:
                    csv_writer.writerow([step, f"{loss.item():.6f}", f"{acc:.6f}", f"{ax:.6f}", f"{az:.6f}"])
                    csv_fh.flush()

    if csv_fh:
        csv_fh.close()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save({'model_state': model.state_dict(), 'cfg': cfg.__dict__, 'hidden': hidden}, save_path)
        print(f"saved checkpoint -> {save_path}")

    return model
