"""
utils.py — data loading helpers for fraud detection datasets.

Supported datasets and their .dgl graph structures:

  yelp   — node type 'r',       relations: u / t / s         (homo excluded)
  amazon — node type 'r',       relations: p / s / v         (homo excluded)
  comp   — node type 'company', relations: invest_bc2bc /
                                           provide_bc2bc /
                                           sale_bc2bc          (homo excluded)
"""

import os
import subprocess
import sys
from typing import List, Tuple

import torch
import dgl


# Relation types per dataset (homo excluded — it overlaps with the others)
DATASET_RELATIONS = {
    "yelp":   ["u", "t", "s"],
    "amazon": ["p", "s", "v"],
    "comp":   ["invest_bc2bc", "provide_bc2bc", "sale_bc2bc"],
}

# Node type to read data from (most datasets use 'r'; comp uses 'company')
DATASET_NODE_TYPE = {
    "yelp":   "r",
    "amazon": "r",
    "comp":   "company",
}


def load_dataset(
    dataset: str,
    data_dir: str = "D:/毕设/data/",
    preprocess_script: str = "D:/毕设/src/data_preprocess.py",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[
    torch.Tensor,          # features  [N, F]
    torch.Tensor,          # labels    [N]
    torch.Tensor,          # train_mask[N]
    torch.Tensor,          # valid_mask[N]
    torch.Tensor,          # test_mask [N]
    List[torch.Tensor],    # edge_indices — list of R [2, E_r] tensors
]:
    """Load a preprocessed DGL graph and return tensors ready for the model.

    If train_ratio + val_ratio < 1.0, the remainder is used as test set.
    When train_ratio=0.7 and val_ratio=0.15 the split is 70/15/15.
    Set train_ratio=0 to use the original masks from the .dgl file.

    Args:
        dataset      : 'yelp', 'amazon', or 'comp'
        data_dir     : directory containing the .dgl files
        train_ratio  : fraction of nodes for training  (0 = use original mask)
        val_ratio    : fraction of nodes for validation
        seed         : random seed for reproducible splits

    Returns:
        features, labels, train_mask, valid_mask, test_mask, edge_indices
    """
    assert dataset in DATASET_RELATIONS, \
        f"Unknown dataset '{dataset}'. Choose from {list(DATASET_RELATIONS.keys())}."

    dgl_path = os.path.join(data_dir, f"{dataset}.dgl")

    # ── Auto-generate if missing ──────────────────────────────────────────────
    if not os.path.exists(dgl_path):
        print(f"[utils] {dgl_path} not found — running preprocessing ...")
        ret = subprocess.run(
            [sys.executable, preprocess_script, "--dataset", dataset],
            check=True,
        )
        if ret.returncode != 0 or not os.path.exists(dgl_path):
            raise RuntimeError(f"Preprocessing failed or did not produce {dgl_path}.")

    # ── Load graph ────────────────────────────────────────────────────────────
    graphs, _ = dgl.load_graphs(dgl_path)
    g = graphs[0]

    ntype = DATASET_NODE_TYPE[dataset]
    features = g.nodes[ntype].data["feature"].float()
    labels   = g.nodes[ntype].data["label"].long()
    N        = features.size(0)

    # ── Build masks ───────────────────────────────────────────────────────────
    if train_ratio > 0:
        # Re-split by the requested ratio (stratified by label)
        import numpy as np
        rng = np.random.default_rng(seed)
        idx = np.arange(N)
        rng.shuffle(idx)
        n_train = int(N * train_ratio)
        n_val   = int(N * val_ratio)

        train_idx = idx[:n_train]
        val_idx   = idx[n_train:n_train + n_val]
        test_idx  = idx[n_train + n_val:]

        train_mask = torch.zeros(N, dtype=torch.bool)
        valid_mask = torch.zeros(N, dtype=torch.bool)
        test_mask  = torch.zeros(N, dtype=torch.bool)
        train_mask[train_idx] = True
        valid_mask[val_idx]   = True
        test_mask[test_idx]   = True

        split_info = f"{train_ratio*100:.0f}/{val_ratio*100:.0f}/{(1-train_ratio-val_ratio)*100:.0f} split"
    else:
        # Use original masks from the .dgl file
        train_mask = g.nodes[ntype].data["train_mask"]
        valid_mask = g.nodes[ntype].data["valid_mask"]
        test_mask  = g.nodes[ntype].data["test_mask"]
        split_info = "original split"

    # ── Build edge_index tensors (PyG format [2, E]) ──────────────────────────
    edge_indices: List[torch.Tensor] = []
    for rel in DATASET_RELATIONS[dataset]:
        src, dst = g.edges(etype=rel)
        edge_indices.append(torch.stack([src, dst], dim=0).long())

    print(
        f"[utils] Loaded '{dataset}' ({split_info}): "
        f"{N} nodes, {features.size(1)} features, "
        f"{int(labels.sum())} fraudsters / {int((labels == 0).sum())} normal | "
        f"train={int(train_mask.sum())} val={int(valid_mask.sum())} test={int(test_mask.sum())}"
    )
    for rel, ei in zip(DATASET_RELATIONS[dataset], edge_indices):
        print(f"  relation '{rel}': {ei.size(1)} edges")

    return features, labels, train_mask, valid_mask, test_mask, edge_indices
