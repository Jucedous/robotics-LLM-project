from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import NODE_F, EDGE_F

class MLP(nn.Module):
    def __init__(self, in_ch, hid, out_ch, act=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, hid), act(),
            nn.Linear(hid, hid), act(),
            nn.Linear(hid, out_ch),
        )
    def forward(self, x):
        return self.net(x)

class HazardGNN(nn.Module):
    def __init__(self, node_f=NODE_F, edge_f=EDGE_F, hidden=64, msg_hidden=64):
        super().__init__()
        self.node_enc = MLP(node_f, hidden, hidden)
        self.edge_mlp = MLP(2*hidden + edge_f, msg_hidden, hidden)
        self.node_upd = MLP(2*hidden, hidden, hidden)
        self.readout = MLP(hidden, hidden, 1)
        self.head_alpha = MLP(hidden, hidden, 2)

    def message_passing(self, X, E, A):
        B, N, H = X.shape
        Xi = X.unsqueeze(2).expand(B, N, N, H)
        Xj = X.unsqueeze(1).expand(B, N, N, H)
        msg_in = torch.cat([Xi, Xj, E], dim=-1)
        msgs = self.edge_mlp(msg_in)
        msgs = msgs * A.unsqueeze(-1)
        agg = msgs.sum(dim=2)
        Xupd = self.node_upd(torch.cat([X, agg], dim=-1))
        return Xupd

    def forward(self, nodes, edges, adj):
        if nodes.dim() == 2:
            nodes = nodes.unsqueeze(0)
            edges = edges.unsqueeze(0)
            adj = adj.unsqueeze(0)
        X0 = self.node_enc(nodes)
        X1 = self.message_passing(X0, edges, adj)
        X2 = self.message_passing(X1, edges, adj)
        G = X2.mean(dim=1)
        hazard_logits = self.readout(G)
        hazard_score = torch.sigmoid(hazard_logits)
        alphas = F.softplus(self.head_alpha(G))
        alpha_xy = alphas[:, 0:1]
        alpha_z  = alphas[:, 1:2]
        return {
            'hazard_score': hazard_score.squeeze(-1),
            'alpha_xy': alpha_xy.squeeze(-1),
            'alpha_z':  alpha_z.squeeze(-1),
        }
