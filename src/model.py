"""
Fraud Detection GNN
===================
Architecture (from model structure):

  1. FeatureEncoder        — h_i^(0) = σ(x_i W_1)
  2. GlobalMessagePassing  — DFS random walk (p>1, q<1) → positional
                             encoding → Transformer self-attention → FFN
                             z_i = softmax(α q̃_i^T K̃_{N(i)}) Ṽ_{N(i)}^T
  3. LocalMessagePassing   — L-layer GAT
  4. RelationAggregation   — learnable weighted aggregation across R relation types
  5. FraudDetectionGNN     — main model: concat(h_global, h_local) → MLP → logits
                             + class-balanced loss for imbalanced fraud data

Advanced mode: local MP → global MP, so distant node info also carries
               local topology of the distant node.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature Encoder  (Initial Embedding)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEncoder(nn.Module):
    """Maps raw node features to a hidden embedding space.

    h_i^(0) = σ(x_i W_1)

    Projecting fraudsters and normal users into a shared space helps
    expose attribute differences that are otherwise masked by similar
    raw feature distributions.
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [N, in_dim]  →  [N, hidden_dim]"""
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DFS Random Walk  (Node2Vec-style biased walk)
# ─────────────────────────────────────────────────────────────────────────────

class DFSRandomWalk:
    """Node2Vec biased random walk that favours depth-first exploration.

    Transition probability:
        P(v_i=r | v_{i-1}=s, v_{i-2}=t)  ∝  1/p  if d(t,r)=0  (return)
                                               1    if d(t,r)=1  (BFS step)
                                               1/q  if d(t,r)=2  (DFS step)

    Setting p > 1 and q < 1 biases the walker towards unexplored nodes
    (DFS), capturing long-range and bridge-based structural patterns.
    """

    def __init__(
        self,
        walk_length: int = 10,
        num_walks: int = 5,
        p: float = 2.0,
        q: float = 0.5,
    ):
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q

    @staticmethod
    def build_adj(edge_index: torch.Tensor) -> Dict[int, List[int]]:
        """Build adjacency list from edge_index once; caller should cache it."""
        adj: Dict[int, List[int]] = {}
        for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            adj.setdefault(u, []).append(v)
        return adj

    def sample(self, adj: Dict[int, List[int]], start: int) -> List[List[int]]:
        """Return `num_walks` DFS-biased walks starting from `start`.

        adj    : pre-built adjacency list (call build_adj once per edge_index)
        Returns: list of node-id lists (including start)
        """
        walks: List[List[int]] = []
        for _ in range(self.num_walks):
            walk = [start]
            for _ in range(self.walk_length - 1):
                curr = walk[-1]
                nbrs = adj.get(curr, [])
                if not nbrs:
                    break
                if len(walk) == 1:
                    walk.append(random.choice(nbrs))
                    continue

                prev = walk[-2]
                prev_set = set(adj.get(prev, []))
                weights = [
                    1.0 / self.p if r == prev
                    else (1.0 if r in prev_set else 1.0 / self.q)
                    for r in nbrs
                ]
                total = sum(weights)
                probs = [w / total for w in weights]
                walk.append(random.choices(nbrs, weights=probs, k=1)[0])
            walks.append(walk)
        return walks


# ─────────────────────────────────────────────────────────────────────────────
# 3. Global-aware Message Passing  (Transformer over path-based neighbourhood)
# ─────────────────────────────────────────────────────────────────────────────

class GlobalMessagePassing(nn.Module):
    """Captures global structural information via DFS random walk + Transformer.

    For each target node i:
      1. Collect path-based neighbourhood via DFS walks (precomputed offline).
      2. Add sinusoidal positional encoding based on walk-sequence position.
      3. Self-attention: node i as Query; walk neighbours as Key/Value.
      4. FFN aggregation.

    z_i = softmax(α q̃_i^T K̃_{N(i)}) Ṽ_{N(i)}^T

    Performance:
      Call precompute(edge_index, num_nodes) once before training.
      This stores walk neighbourhoods as padded tensors [N, K], allowing
      the forward pass to run as a single batched Transformer call with
      no Python-level node loops.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        walk_length: int = 10,
        num_walks: int = 5,
        p: float = 2.0,
        q: float = 0.5,
        max_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.walker   = DFSRandomWalk(walk_length, num_walks, p, q)
        self.max_nbrs = walk_length * num_walks  # upper bound on unique neighbours

        # Sinusoidal positional encoding (position = step in walk sequence)
        pe = torch.zeros(max_len, hidden_dim)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

        self.alpha = nn.Parameter(torch.ones(1))
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Precomputed walk tables (set by precompute())
        self._nbr_nodes: Optional[torch.Tensor] = None   # [N, K]  node ids, -1=pad
        self._nbr_pos:   Optional[torch.Tensor] = None   # [N, K]  walk positions
        self._nbr_mask:  Optional[torch.Tensor] = None   # [N, K]  bool, True=valid

    # ── Offline precomputation ────────────────────────────────────────────────

    def precompute(self, edge_index: torch.Tensor, num_nodes: int) -> None:
        """Precompute DFS walk neighbourhoods for ALL nodes.

        Must be called once before training (and again if the graph changes).
        Stores results as padded CPU tensors; they are moved to the model's
        device automatically in forward().

        edge_index : [2, E] (CPU or GPU — converted internally)
        num_nodes  : total number of nodes N
        """
        print(f"    [GlobalMP] precomputing walks for {num_nodes} nodes "
              f"(walk_len={self.walker.walk_length}, num_walks={self.walker.num_walks}) ...")

        adj = DFSRandomWalk.build_adj(edge_index.cpu())
        K   = self.max_nbrs

        nbr_nodes = torch.full((num_nodes, K), -1,    dtype=torch.long)
        nbr_pos   = torch.zeros((num_nodes, K),       dtype=torch.long)
        nbr_mask  = torch.zeros((num_nodes, K),       dtype=torch.bool)

        for node in range(num_nodes):
            walks = self.walker.sample(adj, node)
            seen: Dict[int, int] = {}
            for walk in walks:
                for step, v in enumerate(walk[1:], start=1):
                    if v not in seen:
                        seen[v] = step
            nbrs = list(seen.keys())
            poss = [seen[v] for v in nbrs]
            k    = min(len(nbrs), K)
            if k > 0:
                nbr_nodes[node, :k] = torch.tensor(nbrs[:k], dtype=torch.long)
                nbr_pos  [node, :k] = torch.tensor(poss[:k], dtype=torch.long)
                nbr_mask [node, :k] = True

        self._nbr_nodes = nbr_nodes   # [N, K]
        self._nbr_pos   = nbr_pos     # [N, K]
        self._nbr_mask  = nbr_mask    # [N, K]
        print(f"    [GlobalMP] precomputation done.")

    # ── Forward (batched, no Python node loop) ────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        x          : [N, d]  current node embeddings
        edge_index : [2, E]  (only used if precompute() was not called)
        target_idx : [M]     nodes to compute global embedding for
        Returns    : [M, d]
        """
        if self._nbr_nodes is not None:
            return self._forward_precomputed(x, target_idx)
        return self._forward_online(x, edge_index, target_idx)

    def _forward_precomputed(
        self, x: torch.Tensor, target_idx: torch.Tensor
    ) -> torch.Tensor:
        """Batched Transformer forward using precomputed walk tables."""
        device = x.device
        M = target_idx.size(0)
        idx_cpu = target_idx.cpu()  # precomputed tables live on CPU

        nbr_nodes = self._nbr_nodes[idx_cpu].to(device)   # [M, K]
        nbr_pos   = self._nbr_pos  [idx_cpu].to(device)   # [M, K]
        nbr_mask  = self._nbr_mask [idx_cpu].to(device)   # [M, K]

        K = nbr_nodes.size(1)

        # Gather neighbour embeddings; pad positions use node 0 (masked out anyway)
        safe_idx  = nbr_nodes.clamp(min=0)                   # [M, K]
        nbr_emb   = x[safe_idx]                              # [M, K, d]

        # Add positional encoding
        pos_clamped = nbr_pos.clamp(max=self.pe.size(0) - 1) # [M, K]
        nbr_emb     = nbr_emb + self.pe[pos_clamped]         # [M, K, d]

        # Zero out padding positions
        nbr_emb = nbr_emb * nbr_mask.unsqueeze(-1).float()   # [M, K, d]

        # Self-attention: query = target node, key/value = walk neighbours
        query = self.q_proj(x[target_idx]).unsqueeze(1)      # [M, 1, d]
        key   = self.k_proj(nbr_emb)                         # [M, K, d]
        value = self.v_proj(nbr_emb)                         # [M, K, d]

        # key_padding_mask: True = ignore (pad positions)
        key_pad_mask = ~nbr_mask                             # [M, K]

        # Nodes with NO valid neighbours would make softmax degenerate → NaN.
        # Force at least one position to be "valid" for those rows so that
        # softmax has something to attend to; the value is zeroed anyway.
        all_masked = key_pad_mask.all(dim=1)                 # [M]
        key_pad_mask[all_masked, 0] = False                  # unmask slot 0

        attn_out, _ = self.attn(
            query, key, value, key_padding_mask=key_pad_mask
        )                                                    # [M, 1, d]
        attn_out = attn_out.squeeze(1)                       # [M, d]

        # For nodes that had no neighbours, ignore the attention output
        attn_out = attn_out * (~all_masked).float().unsqueeze(1)

        h = self.norm1(x[target_idx] + attn_out)
        h = self.norm2(h + self.ffn(h))
        return h                                             # [M, d]

    def _forward_online(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback: per-node Python loop (slow, for small graphs / testing)."""
        device = x.device
        adj  = DFSRandomWalk.build_adj(edge_index.cpu())
        outs: List[torch.Tensor] = []

        for i in target_idx.tolist():
            walks = self.walker.sample(adj, i)
            seen: Dict[int, int] = {}
            for walk in walks:
                for step, v in enumerate(walk[1:], start=1):
                    if v not in seen:
                        seen[v] = step

            if not seen:
                outs.append(x[i])
                continue

            nbrs = list(seen.keys())
            poss = list(seen.values())
            pos_t    = torch.tensor(poss, dtype=torch.long, device=device).clamp(max=self.pe.size(0) - 1)
            nbr_emb  = x[nbrs] + self.pe[pos_t]
            query    = self.q_proj(x[i]).unsqueeze(0).unsqueeze(0)
            key      = self.k_proj(nbr_emb).unsqueeze(0)
            value    = self.v_proj(nbr_emb).unsqueeze(0)
            out, _   = self.attn(query, key, value)
            h = self.norm1(x[i].unsqueeze(0) + out.squeeze(0))
            h = self.norm2(h + self.ffn(h))
            outs.append(h.squeeze(0))

        return torch.stack(outs)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Local Message Passing  (GAT, L layers)
# ─────────────────────────────────────────────────────────────────────────────

class LocalMessagePassing(nn.Module):
    """L-layer GAT for local neighbourhood aggregation.

    Most normal-user neighbours are also normal users, so local
    propagation is a strong signal for benign classification.
    After each layer the residual connection + LayerNorm stabilises training.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_degree: Optional[int] = None,
    ):
        """
        max_degree : if set, randomly sample at most this many neighbours per
                     node before each GAT layer.  Keeps GPU memory bounded for
                     relations with very large edge counts (e.g. Amazon 's').
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        head_dim = hidden_dim // num_heads
        self.layers = nn.ModuleList(
            [
                GATConv(hidden_dim, head_dim, heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout   = dropout
        self.max_degree = max_degree

    @staticmethod
    def _sample_edges(edge_index: torch.Tensor, max_degree: int) -> torch.Tensor:
        """For each destination node keep at most `max_degree` random incoming edges.

        Fully vectorised — no Python-level loops over edges.
        """
        device = edge_index.device
        num_edges = edge_index.size(1)

        # 1. Randomly shuffle edges so within each dst group order is random
        perm = torch.randperm(num_edges, device=device)
        shuffled_dst = edge_index[1, perm]

        # 2. Stable-sort by dst: within each dst group the random shuffle is kept
        order = torch.argsort(shuffled_dst, stable=True)
        sorted_dst = shuffled_dst[order]

        # 3. Compute within-group rank via cumsum trick
        new_group = torch.cat([
            torch.ones(1, dtype=torch.bool, device=device),
            sorted_dst[1:] != sorted_dst[:-1],
        ])
        group_id     = new_group.cumsum(0) - 1          # group index per edge
        group_starts = new_group.nonzero(as_tuple=True)[0]  # start pos per group
        within_rank  = torch.arange(num_edges, device=device) - group_starts[group_id]

        # 4. Keep edges whose within-group rank < max_degree
        keep = within_rank < max_degree
        return edge_index[:, perm[order[keep]]]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """x : [N, d]  →  [N, d]"""
        h = x
        for conv, norm in zip(self.layers, self.norms):
            ei = edge_index
            if self.max_degree is not None and self.training:
                ei = self._sample_edges(ei, self.max_degree)
            h = norm(
                h + F.dropout(conv(h, ei), p=self.dropout, training=self.training)
            )
        return h


# ─────────────────────────────────────────────────────────────────────────────
# 5. Relation-wise Weighted Aggregation
# ─────────────────────────────────────────────────────────────────────────────

class RelationAggregation(nn.Module):
    """Soft weighted aggregation of R relation-specific embeddings.

    Different relation types have different sensitivities to global vs.
    local structure; learnable attention weights capture this automatically.

    mean_agg=True uses simple mean (ablation: Mean-Rel).
    """

    def __init__(self, hidden_dim: int, mean_agg: bool = False):
        super().__init__()
        self.mean_agg = mean_agg
        if not mean_agg:
            self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, embs: List[torch.Tensor]) -> torch.Tensor:
        """
        embs : list of R tensors, each [M, d]
        Returns [M, d]
        """
        stacked = torch.stack(embs, dim=1)               # [M, R, d]
        if self.mean_agg:
            return stacked.mean(dim=1)                   # [M, d]
        weights = F.softmax(self.score(stacked), dim=1)  # [M, R, 1]
        return (weights * stacked).sum(dim=1)            # [M, d]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main Model
# ─────────────────────────────────────────────────────────────────────────────

class FraudDetectionGNN(nn.Module):
    """Heterogeneous graph fraud detection model.

    Full pipeline per relation type r:
        x  →  FeatureEncoder  →  h^(0)
        h^(0)  →  GlobalMP_r  →  h_global_r    (path-based, Transformer)
        h^(0)  →  LocalMP_r   →  h_local_r     (GAT, L layers)

    Aggregation:
        {h_global_r}  →  RelationAgg  →  h_global
        {h_local_r}   →  RelationAgg  →  h_local
        concat(h_global, h_local)  →  MLP  →  logits

    Advanced mode (advanced=True):
        h^(0)  →  LocalMP_r  →  h_local_r  →  GlobalMP_r  →  h_global_r
        Lets global MP receive locally-enriched embeddings, so distant
        nodes bring their own local topology into the target's context.

    Loss: class-balanced cross-entropy adjusts for fraudster/normal imbalance
        logits' = logits + log P(y),  where P(y) = empirical class freq.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        num_relations: int = 3,
        num_local_layers: int = 2,
        num_heads: int = 4,
        walk_length: int = 10,
        num_walks: int = 5,
        p: float = 2.0,
        q: float = 0.5,
        dropout: float = 0.1,
        advanced: bool = False,
        max_degree: Optional[int] = 50,
        # ── Ablation flags ────────────────────────────────────────────────────
        use_feature_encoder: bool = True,   # False → single linear projection
        use_global_mp: bool = True,         # False → zero global embeddings
        use_local_mp: bool = True,          # False → zero local embeddings
        mean_relation_agg: bool = False,    # True  → simple mean over relations
    ):
        """
        max_degree          : max neighbours sampled per node in LocalMP during
                              training. Default 50 keeps GPU memory bounded.
        use_feature_encoder : ablation — replace 2-layer MLP with linear proj
        use_global_mp       : ablation — disable global (DFS+Transformer) branch
        use_local_mp        : ablation — disable local (GAT) branch
        mean_relation_agg   : ablation — use simple mean instead of learned weights
        """
        super().__init__()
        self.num_relations       = num_relations
        self.advanced            = advanced
        self.use_global_mp       = use_global_mp
        self.use_local_mp        = use_local_mp

        # 1. Initial embedding
        if use_feature_encoder:
            self.encoder = FeatureEncoder(in_dim, hidden_dim)
        else:
            # Ablation: single linear projection (no nonlinearity)
            self.encoder = nn.Linear(in_dim, hidden_dim)

        # 2. Global MP — one per relation type (skipped if ablated)
        self.global_mp = nn.ModuleList(
            [
                GlobalMessagePassing(
                    hidden_dim, num_heads, walk_length, num_walks, p, q, dropout=dropout
                )
                for _ in range(num_relations)
            ]
        ) if use_global_mp else nn.ModuleList()

        # 3. Local MP — one per relation type (skipped if ablated)
        self.local_mp = nn.ModuleList(
            [
                LocalMessagePassing(hidden_dim, num_local_layers, num_heads, dropout,
                                    max_degree=max_degree)
                for _ in range(num_relations)
            ]
        ) if use_local_mp else nn.ModuleList()

        # 4. Relation aggregation (attention or mean)
        self.global_rel_agg = RelationAggregation(hidden_dim, mean_agg=mean_relation_agg)
        self.local_rel_agg  = RelationAggregation(hidden_dim, mean_agg=mean_relation_agg)

        # 5. Final classifier: concat(h_global, h_local) → MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_indices: List[torch.Tensor],
        target_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x           : [N, in_dim]       node feature matrix
        edge_indices: list of R tensors [2, E_r], one per relation type
        target_idx  : [M] node indices to classify (default: all nodes)
        Returns     : logits  [M, num_classes]
        """
        if target_idx is None:
            target_idx = torch.arange(x.size(0), device=x.device)

        M = target_idx.size(0)
        d = self.classifier[0].in_features // 2  # hidden_dim

        # 1. Initial embedding
        h0 = self.encoder(x)  # [N, d]

        global_embs: List[torch.Tensor] = []
        local_embs:  List[torch.Tensor] = []

        for r, ei in enumerate(edge_indices):
            # Global branch
            if self.use_global_mp:
                g_mp = self.global_mp[r]
                if self.advanced and self.use_local_mp:
                    src = self.local_mp[r](h0, ei)
                    global_h = g_mp(src, ei, target_idx)
                else:
                    global_h = g_mp(h0, ei, target_idx)
            else:
                global_h = torch.zeros(M, d, device=x.device)

            # Local branch
            if self.use_local_mp:
                l_mp = self.local_mp[r]
                local_h = l_mp(h0, ei)[target_idx]
            else:
                local_h = torch.zeros(M, d, device=x.device)

            global_embs.append(global_h)
            local_embs.append(local_h)

        # 4. Weighted aggregation across relation types
        h_global = self.global_rel_agg(global_embs)  # [M, d]
        h_local  = self.local_rel_agg(local_embs)    # [M, d]

        # 5. Concat + MLP
        h = torch.cat([h_global, h_local], dim=-1)   # [M, 2d]
        return self.classifier(h)                     # [M, num_classes]

    # ── Walk precomputation ───────────────────────────────────────────────────

    def precompute_walks(
        self, edge_indices: List[torch.Tensor], num_nodes: int
    ) -> None:
        """Precompute DFS walk neighbourhoods for each relation's GlobalMP.

        No-op when use_global_mp=False (ablation).
        """
        if not self.use_global_mp:
            return
        print("[model] Precomputing walk neighbourhoods ...")
        for r, (g_mp, ei) in enumerate(zip(self.global_mp, edge_indices)):
            print(f"  relation {r+1}/{len(edge_indices)}")
            g_mp.precompute(ei, num_nodes)
        print("[model] Precomputation complete.")

    # ── Loss ─────────────────────────────────────────────────────────────────

    @staticmethod
    def balanced_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_priors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Class-balanced cross-entropy loss.

        Adjusts logits by log P(y) (empirical class frequencies) to correct
        for the severe imbalance between fraudsters and normal users:

            argmax_y P(y|x) = argmax_y (C_y(x;Θ) + log P(y))

        Args:
            logits        : [M, num_classes] raw model output
            labels        : [M] ground-truth class indices
            class_priors  : [num_classes] pre-computed class frequencies;
                            estimated from `labels` if not provided
        """
        if class_priors is None:
            counts = torch.bincount(labels, minlength=logits.size(-1)).float()
            class_priors = counts / counts.sum()

        log_prior = torch.log(class_priors.clamp(min=1e-8)).to(logits.device)
        adjusted_logits = logits + log_prior.unsqueeze(0)
        return F.cross_entropy(adjusted_logits, labels)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)

    N          = 100    # nodes
    in_dim     = 32     # raw feature dim
    hidden_dim = 64
    R          = 3      # relation types

    x = torch.randn(N, in_dim)

    # Random heterogeneous edges for each relation type
    edge_indices = [
        torch.randint(0, N, (2, 200)) for _ in range(R)
    ]

    target_idx = torch.arange(N)
    labels = torch.randint(0, 2, (N,))  # 0 = normal, 1 = fraudster

    model = FraudDetectionGNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
        num_relations=R,
        num_local_layers=2,
        num_heads=4,
        walk_length=6,
        num_walks=3,
        p=2.0,
        q=0.5,
        dropout=0.1,
        advanced=False,
    )

    logits = model(x, edge_indices, target_idx)
    loss   = FraudDetectionGNN.balanced_loss(logits, labels)

    print(f"logits shape : {logits.shape}")   # [100, 2]
    print(f"loss         : {loss.item():.4f}")
    print("Smoke test passed.")
