# gnn_hazard.py
# Minimal scene-graph GNN for pairwise hazard risk
# python>=3.9, torch>=2.0

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utilities
# -----------------------------
def build_edge_features(pos, sizes, edge_index, kinds, kind_to_id, rel_feats):
    """
    pos: [N, 3] world positions (x,y,z) for object reference points
    sizes: [N, 3] (optional) bbox sizes (not used here but kept for extension)
    edge_index: [2, E] tensor of source->target indices
    kinds: list[str] length N (e.g. "liquid", "electronic")
    rel_feats: dict with precomputed edge features { "d_xy": [E], "z_gap": [E], "is_above": [E] }

    Returns: edge_attr [E, D_e], node_kind_ids [N]
    """
    src, dst = edge_index
    d_xy     = torch.as_tensor(rel_feats["d_xy"], dtype=torch.float32).unsqueeze(1)      # meters
    z_gap    = torch.as_tensor(rel_feats["z_gap"], dtype=torch.float32).unsqueeze(1)     # meters (A.bottom - B.top)
    is_above = torch.as_tensor(rel_feats["is_above"], dtype=torch.float32).unsqueeze(1)  # 0/1

    # Monotonicity trick: include -d_xy so risk increases as distance shrinks.
    edge_attr = torch.cat([ -d_xy, d_xy, z_gap, is_above ], dim=1)  # [-d, d, z_gap, above]
    node_kind_ids = torch.tensor([kind_to_id[k] for k in kinds], dtype=torch.long)
    return edge_attr, node_kind_ids


def scatter_add(src, index, dim_size):
    out = src.new_zeros((dim_size, src.size(-1)))
    out.index_add_(0, index, src)
    return out


# -----------------------------
# Model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, act=nn.SiLU, last_act=False):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.act = act()
        self.last_act = last_act

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        if self.last_act:
            x = self.act(x)
        return x


class HazardGNN(nn.Module):
    """
    - Node init: learned embeddings for object kinds
    - Message: phi(h_src, h_dst, e_attr)
    - Aggregate: sum at destination
    - Update: GRU(h_dst, agg)
    - Edge head: hazard score per edge in [0,1]
    """
    def __init__(self, num_kinds, d_node=64, d_edge=4, d_msg=64, num_layers=3):
        super().__init__()
        self.emb = nn.Embedding(num_kinds, d_node)

        self.msg_mlps = nn.ModuleList([
            MLP(d_in=2*d_node + d_edge, d_hidden=d_msg, d_out=d_msg) for _ in range(num_layers)
        ])
        self.upd_grus = nn.ModuleList([
            nn.GRUCell(d_msg, d_node) for _ in range(num_layers)
        ])

        # Edge scoring head (uses final node reps + edge features)
        self.edge_head = nn.Sequential(
            nn.Linear(2*d_node + d_edge, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # Optional: ensure non-negative weights on the -d_xy channel to enforce monotonicity.
        # Here we’ll do it softly by clamping selected weights after each step (see clamp_monotone()).
        self._monotone_channels = {"minus_d_xy": 0}  # index 0 in edge_attr is -d_xy

    def forward(self, edge_index, node_kind_ids, edge_attr):
        """
        edge_index: [2, E] long
        node_kind_ids: [N] long
        edge_attr: [E, d_edge]
        Returns: hazard_prob [E] in [0,1]
        """
        N = node_kind_ids.numel()
        E = edge_index.size(1)
        h = self.emb(node_kind_ids)  # [N, d_node]

        src, dst = edge_index

        for msg_mlp, gru in zip(self.msg_mlps, self.upd_grus):
            h_src = h[src]
            h_dst = h[dst]
            m = msg_mlp(torch.cat([h_src, h_dst, edge_attr], dim=1))  # [E, d_msg]
            agg = scatter_add(m, dst, dim_size=N)  # sum messages into dst nodes
            h = gru(agg, h)  # node-wise update

        # Edge scoring
        h_src = h[src]
        h_dst = h[dst]
        logits = self.edge_head(torch.cat([h_src, h_dst, edge_attr], dim=1)).squeeze(-1)
        prob = torch.sigmoid(logits)
        return prob

    @torch.no_grad()
    def clamp_monotone(self):
        """
        Soft monotonicity: encourage risk to increase with -d_xy by clamping
        weights >= 0 on the edge-head input channel that corresponds to -d_xy.
        """
        # Find the first linear layer in edge_head
        lin = None
        for m in self.edge_head:
            if isinstance(m, nn.Linear):
                lin = m
                break
        if lin is None: return

        # Inputs order to edge_head: [h_src, h_dst, edge_attr]
        # edge_attr first channel index in that concatenation:
        # idx_edge_start = 2*d_node; minus_d_xy is at idx_edge_start + 0
        pass  # Implement if you want strict clamping; omitted for brevity.


# -----------------------------
# Tiny demo
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Ontology
    kinds = ["liquid", "electronic", "human", "sharp", "fragile", "hot", "unknown"]
    kind_to_id = {k:i for i,k in enumerate(kinds)}

    # Scene: cup(liquid) above laptop(electronic), plus a distractor object
    node_kinds = ["liquid", "electronic", "fragile"]  # N=3
    N = len(node_kinds)

    # Build a directed edge for each ordered pair (i->j), i != j
    edges = [(i,j) for i in range(N) for j in range(N) if i!=j]
    edge_index = torch.tensor(edges, dtype=torch.long).T  # [2,E]

    # Positions & sizes (not used directly, features precomputed for simplicity)
    pos   = torch.tensor([[0.00, 0.00, 0.20],   # cup
                          [0.00, 0.00, 0.00],   # laptop (top at z=0)
                          [0.60, 0.00, 0.00]],  # vase (fragile)
                         dtype=torch.float32)
    sizes = torch.ones_like(pos)*0.1

    # Precompute relational features (normally from your geometry lib)
    def d_xy(a,b): return torch.linalg.norm((a-b)[:2])
    def z_gap(a,b): return a[2] - b[2]  # treat z as bottom-to-top simplified

    dxy_list, zgap_list, above_list = [], [], []
    for (i,j) in edges:
        dij = d_xy(pos[i], pos[j]).item()
        zg  = z_gap(pos[i], pos[j]).item()
        dxy_list.append(dij)
        zgap_list.append(zg)
        above_list.append(1.0 if zg>0 else 0.0)

    rel_feats = {"d_xy": dxy_list, "z_gap": zgap_list, "is_above": above_list}
    edge_attr, node_kind_ids = build_edge_features(
        pos, sizes, edge_index, node_kinds, kind_to_id, rel_feats
    )

    model = HazardGNN(num_kinds=len(kinds), d_node=64, d_edge=edge_attr.size(1), d_msg=64, num_layers=2)

    # Fake labels: mark (liquid->electronic) hazardous if above & close laterally (<0.22m)
    labels = []
    for (i,j), dij, zg, abv in zip(edges, dxy_list, zgap_list, above_list):
        ki, kj = node_kinds[i], node_kinds[j]
        haz = 1.0 if (ki=="liquid" and kj=="electronic" and abv>0.5 and dij<0.22) else 0.0
        labels.append(haz)
    y = torch.tensor(labels, dtype=torch.float32)

    # Train (tiny) – purely illustrative
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(500):
        opt.zero_grad()
        p = model(edge_index, node_kind_ids, edge_attr)
        loss = F.binary_cross_entropy(p, y)
        loss.backward()
        opt.step()
        if (step+1) % 100 == 0:
            with torch.no_grad():
                print(f"step {step+1}: loss={loss.item():.4f}, preds={p.detach().round().tolist()}")

    # After training, p ~ hazard probability for each directed edge in `edges`
    with torch.no_grad():
        p = model(edge_index, node_kind_ids, edge_attr)
        for (i,j), prob in zip(edges, p.tolist()):
            print(f"{node_kinds[i]} -> {node_kinds[j]}  prob={prob:.2f}")
