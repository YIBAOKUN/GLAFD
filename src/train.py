"""
train.py — training and evaluation script for FraudDetectionGNN.

Usage:
    # Train on YelpChi (default)
    python src/train.py --dataset yelp

    # Train on Amazon with advanced mode
    python src/train.py --dataset amazon --advanced

    # Full options
    python src/train.py --dataset yelp --hidden_dim 64 --epochs 100 --lr 1e-3
"""

import argparse
import os
import sys

import torch
import numpy as np
from typing import Dict, List
from sklearn.metrics import f1_score, roc_auc_score, recall_score, confusion_matrix

# Allow running from project root or from src/
sys.path.insert(0, os.path.dirname(__file__))

from model import FraudDetectionGNN
from utils import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: FraudDetectionGNN,
    features: torch.Tensor,
    edge_indices: List[torch.Tensor],
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute Recall, F1-macro, AUC and GMean on nodes selected by `mask`.

    GMean = sqrt(recall_normal × recall_fraud)
          = geometric mean of per-class recall, robust to class imbalance.

    Returns: dict with keys 'recall', 'f1_macro', 'auc', 'gmean'
    """
    model.eval()
    target_idx = mask.nonzero(as_tuple=True)[0]
    logits = model(features, edge_indices, target_idx)          # [M, 2]
    probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()   # fraud prob
    preds  = logits.argmax(dim=1).cpu().numpy()
    true   = labels[target_idx].cpu().numpy()

    # Recall of the fraud class (class 1)
    recall   = recall_score(true, preds, pos_label=1, zero_division=0)
    f1_macro = f1_score(true, preds, average="macro", zero_division=0)
    auc      = roc_auc_score(true, probs)

    # GMean: geometric mean of recall for each class
    cm  = confusion_matrix(true, preds, labels=[0, 1])
    tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0] + 1e-8)  # recall of fraud
    tnr = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-8)  # recall of normal
    gmean = float(np.sqrt(tpr * tnr))

    return {"recall": recall, "f1_macro": f1_macro, "auc": auc, "gmean": gmean}


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    # ── Load data ─────────────────────────────────────────────────────────────
    features, labels, train_mask, valid_mask, test_mask, edge_indices = load_dataset(
        args.dataset, data_dir=args.data_dir,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[train] Using device: {device}")

    features     = features.to(device)
    labels       = labels.to(device)
    edge_indices = [ei.to(device) for ei in edge_indices]
    train_mask   = train_mask.to(device)
    valid_mask   = valid_mask.to(device)
    test_mask    = test_mask.to(device)

    train_mask  = train_mask.bool()
    valid_mask  = valid_mask.bool()
    test_mask   = test_mask.bool()
    train_idx   = train_mask.nonzero(as_tuple=True)[0]

    # ── Class priors for balanced loss (from training set) ────────────────────
    train_labels  = labels[train_mask.bool()]
    counts        = torch.bincount(train_labels, minlength=2).float()
    class_priors  = (counts / counts.sum()).to(device)
    print(f"[train] Class priors — normal: {class_priors[0]:.3f}, fraud: {class_priors[1]:.3f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    in_dim        = features.size(1)
    num_relations = len(edge_indices)

    model = FraudDetectionGNN(
        in_dim        = in_dim,
        hidden_dim    = args.hidden_dim,
        num_classes   = 2,
        num_relations = num_relations,
        num_local_layers = args.num_layers,
        num_heads     = args.num_heads,
        walk_length   = args.walk_length,
        num_walks     = args.num_walks,
        p             = args.p,
        q             = args.q,
        dropout       = args.dropout,
        advanced      = args.advanced,
        max_degree    = args.max_degree if args.max_degree > 0 else None,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {total_params:,}")

    # Precompute walk neighbourhoods (one-time, CPU, before training)
    model.precompute_walks(
        [ei.cpu() for ei in edge_indices], features.size(0)
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_auc  = 0.0
    best_epoch    = 0
    ckpt_path     = os.path.join(args.save_dir, f"best_{args.dataset}.pt")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'Epoch':>6}  {'Loss':>8}  {'Recall':>8}  {'F1-mac':>8}  {'AUC':>8}  {'GMean':>8}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(features, edge_indices, train_idx)
        loss   = FraudDetectionGNN.balanced_loss(logits, labels[train_idx], class_priors)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m = evaluate(model, features, edge_indices, labels, valid_mask)
            scheduler.step(m["auc"])

            marker = " *" if m["auc"] > best_val_auc else ""
            print(
                f"{epoch:>6}  {loss.item():>8.4f}  "
                f"{m['recall']:>8.4f}  {m['f1_macro']:>8.4f}  "
                f"{m['auc']:>8.4f}  {m['gmean']:>8.4f}{marker}"
            )

            if m["auc"] > best_val_auc:
                best_val_auc = m["auc"]
                best_epoch   = epoch
                torch.save(model.state_dict(), ckpt_path)

    # ── Test with best checkpoint ─────────────────────────────────────────────
    print(f"\n[train] Best epoch: {best_epoch}  (Val AUC {best_val_auc:.4f})")
    print(f"[train] Loading checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    m = evaluate(model, features, edge_indices, labels, test_mask)
    result_lines = [
        "",
        "-" * 40,
        f"Dataset  : {args.dataset}",
        f"Best epoch: {best_epoch}  (Val AUC {best_val_auc:.4f})",
        f"  Recall   : {m['recall']:.4f}",
        f"  F1-macro : {m['f1_macro']:.4f}",
        f"  AUC      : {m['auc']:.4f}",
        f"  GMean    : {m['gmean']:.4f}",
        "-" * 40,
        "",
    ]
    for line in result_lines:
        print(line)

    # Save to results file
    result_path = os.path.join(args.save_dir, f"{args.dataset}_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        # Write hyperparameters
        f.write(f"Dataset      : {args.dataset}\n")
        f.write(f"Split        : {args.train_ratio*100:.0f}/{args.val_ratio*100:.0f}"
                f"/{(1-args.train_ratio-args.val_ratio)*100:.0f}\n")
        f.write(f"hidden_dim   : {args.hidden_dim}\n")
        f.write(f"num_layers   : {args.num_layers}\n")
        f.write(f"num_heads    : {args.num_heads}\n")
        f.write(f"walk_length  : {args.walk_length}\n")
        f.write(f"num_walks    : {args.num_walks}\n")
        f.write(f"p / q        : {args.p} / {args.q}\n")
        f.write(f"max_degree   : {args.max_degree}\n")
        f.write(f"dropout      : {args.dropout}\n")
        f.write(f"advanced     : {args.advanced}\n")
        f.write(f"epochs       : {args.epochs}\n")
        f.write(f"lr           : {args.lr}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best epoch   : {best_epoch}  (Val AUC {best_val_auc:.4f})\n")
        f.write("-" * 40 + "\n")
        f.write(f"Recall       : {m['recall']:.4f}\n")
        f.write(f"F1-macro     : {m['f1_macro']:.4f}\n")
        f.write(f"AUC          : {m['auc']:.4f}\n")
        f.write(f"GMean        : {m['gmean']:.4f}\n")
    print(f"[train] Results saved to {result_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FraudDetectionGNN")

    # Data
    p.add_argument("--dataset",  type=str, default="yelp",
                   choices=["yelp", "amazon", "comp"], help="Dataset to use")
    p.add_argument("--data_dir",    type=str,   default="D:/毕设/data/",
                   help="Directory containing .dgl files")
    p.add_argument("--train_ratio", type=float, default=0,
                   help="Fraction of nodes for training (0 = use original split)")
    p.add_argument("--val_ratio",   type=float, default=0.15,
                   help="Fraction of nodes for validation")
    p.add_argument("--save_dir", type=str, default="D:/毕设/results/",
                   help="Directory to save best model checkpoint and results")

    # Model architecture
    p.add_argument("--hidden_dim",  type=int,   default=64)
    p.add_argument("--num_layers",  type=int,   default=2,
                   help="Number of GAT layers for local MP")
    p.add_argument("--num_heads",   type=int,   default=4,
                   help="Attention heads (hidden_dim must be divisible by this)")
    p.add_argument("--walk_length", type=int,   default=10,
                   help="DFS random walk length")
    p.add_argument("--num_walks",   type=int,   default=5,
                   help="Number of DFS walks per node")
    p.add_argument("--p",           type=float, default=2.0,
                   help="Node2Vec return parameter (>1 for DFS bias)")
    p.add_argument("--q",           type=float, default=0.5,
                   help="Node2Vec in-out parameter (<1 for DFS bias)")
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--max_degree",  type=int,   default=50,
                   help="Max neighbours per node in LocalMP (0 = no limit)")
    p.add_argument("--advanced",    action="store_true",
                   help="Advanced mode: local MP -> global MP")

    # Training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--eval_every",   type=int,   default=5,
                   help="Evaluate on validation set every N epochs")
    p.add_argument("--cpu",          action="store_true",
                   help="Force CPU even if CUDA is available")
    p.add_argument("--no_save",      action="store_true",
                   help="Skip saving results to file")
    p.add_argument("--seed",         type=int,   default=42,
                   help="Random seed for reproducibility")

    return p.parse_args()


if __name__ == "__main__":
    train(get_args())
