"""
ablation.py — run all ablation variants on both datasets and save results.

Variants:
  Full          complete model (baseline)
  w/o FE        FeatureEncoder replaced with single linear projection
  w/o Global    global (DFS+Transformer) branch disabled
  w/o Local     local (GAT) branch disabled
  Mean-Rel      relation aggregation uses simple mean instead of attention
  w/o Balanced  standard cross-entropy instead of class-balanced loss
  Advanced      local MP → global MP (local-enriched global)

Usage:
    python src/ablation.py
    python src/ablation.py --datasets yelp          # single dataset
    python src/ablation.py --epochs 50              # faster run
"""

import argparse
import os
import sys
import time
from typing import Dict, List

import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, recall_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from model import FraudDetectionGNN
from utils import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Ablation variant definitions
# Each entry: (display_name, model_kwargs_overrides, use_balanced_loss)
# ─────────────────────────────────────────────────────────────────────────────

VARIANTS = [
    ("Full",          {},                                                    True),
    ("w/o FE",        {"use_feature_encoder": False},                        True),
    ("w/o Global",    {"use_global_mp":  False},                             True),
    ("w/o Local",     {"use_local_mp":   False},                             True),
    ("Mean-Rel",      {"mean_relation_agg": True},                           True),
    ("w/o Balanced",  {},                                                    False),
    ("Advanced",      {"advanced": True},                                    True),
]


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
    model.eval()
    target_idx = mask.nonzero(as_tuple=True)[0]
    logits = model(features, edge_indices, target_idx)
    probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds  = logits.argmax(dim=1).cpu().numpy()
    true   = labels[target_idx].cpu().numpy()

    recall   = recall_score(true, preds, pos_label=1, zero_division=0)
    f1_macro = f1_score(true, preds, average="macro", zero_division=0)
    auc      = roc_auc_score(true, probs)
    cm  = confusion_matrix(true, preds, labels=[0, 1])
    tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0] + 1e-8)
    tnr = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-8)
    gmean = float(np.sqrt(tpr * tnr))

    return {"recall": recall, "f1_macro": f1_macro, "auc": auc, "gmean": gmean}


# ─────────────────────────────────────────────────────────────────────────────
# Single variant training run
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(
    variant_name: str,
    model_kwargs: dict,
    use_balanced_loss: bool,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    test_mask: torch.Tensor,
    edge_indices: List[torch.Tensor],
    base_cfg: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:

    features_d     = features.to(device)
    labels_d       = labels.to(device)
    edge_indices_d = [ei.to(device) for ei in edge_indices]
    train_mask_d   = train_mask.to(device)
    valid_mask_d   = valid_mask.to(device)
    test_mask_d    = test_mask.to(device)
    train_idx      = train_mask_d.nonzero(as_tuple=True)[0]

    counts       = torch.bincount(labels_d[train_mask_d], minlength=2).float()
    class_priors = (counts / counts.sum()).to(device)

    # Build model with base config + variant overrides
    model = FraudDetectionGNN(
        in_dim           = features.size(1),
        hidden_dim       = base_cfg.hidden_dim,
        num_classes      = 2,
        num_relations    = len(edge_indices),
        num_local_layers = base_cfg.num_layers,
        num_heads        = base_cfg.num_heads,
        walk_length      = base_cfg.walk_length,
        num_walks        = base_cfg.num_walks,
        p                = base_cfg.p,
        q                = base_cfg.q,
        dropout          = base_cfg.dropout,
        max_degree       = base_cfg.max_degree,
        **model_kwargs,
    ).to(device)

    model.precompute_walks(edge_indices_d, features.size(0))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=base_cfg.lr, weight_decay=base_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, verbose=False
    )

    best_val_auc = 0.0
    best_state   = None

    for epoch in range(1, base_cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features_d, edge_indices_d, train_idx)

        if use_balanced_loss:
            loss = FraudDetectionGNN.balanced_loss(logits, labels_d[train_idx], class_priors)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels_d[train_idx])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % base_cfg.eval_every == 0 or epoch == base_cfg.epochs:
            m = evaluate(model, features_d, edge_indices_d, labels_d, valid_mask_d)
            scheduler.step(m["auc"])
            if m["auc"] > best_val_auc:
                best_val_auc = m["auc"]
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best and evaluate on test
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_m = evaluate(model, features_d, edge_indices_d, labels_d, test_mask_d)
    return test_m


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}\n")

    os.makedirs(args.save_dir, exist_ok=True)
    summary_path = os.path.join(args.save_dir, "ablation_results.txt")

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}  # dataset→variant→metrics

    # Per-dataset lr/epochs overrides
    dataset_lr     = {"yelp": args.yelp_lr,     "amazon": args.amazon_lr,     "comp": args.comp_lr}
    dataset_epochs = {"yelp": args.yelp_epochs, "amazon": args.amazon_epochs, "comp": args.comp_epochs}

    for dataset in args.datasets:
        print(f"{'='*60}")
        print(f"  Dataset: {dataset.upper()}")
        print(f"{'='*60}")

        features, labels, train_mask, valid_mask, test_mask, edge_indices = \
            load_dataset(dataset, data_dir=args.data_dir, train_ratio=0)

        # Build a per-dataset config by overriding lr and epochs
        import copy
        ds_cfg = copy.copy(args)
        ds_cfg.lr     = dataset_lr[dataset]     if dataset_lr[dataset]     is not None else args.lr
        ds_cfg.epochs = dataset_epochs[dataset] if dataset_epochs[dataset] is not None else args.epochs

        print(f"  lr={ds_cfg.lr}  epochs={ds_cfg.epochs}")

        all_results[dataset] = {}

        for name, overrides, use_bl in VARIANTS:
            print(f"\n  ── Variant: {name} ──")
            t0 = time.time()

            metrics = run_variant(
                name, overrides, use_bl,
                features, labels, train_mask, valid_mask, test_mask, edge_indices,
                ds_cfg, device,
            )
            elapsed = time.time() - t0
            all_results[dataset][name] = metrics

            print(
                f"     Recall={metrics['recall']:.4f}  "
                f"F1-mac={metrics['f1_macro']:.4f}  "
                f"AUC={metrics['auc']:.4f}  "
                f"GMean={metrics['gmean']:.4f}  "
                f"({elapsed/60:.1f} min)"
            )

    # ── Print & save summary table ────────────────────────────────────────────
    metrics_order = ["recall", "f1_macro", "auc", "gmean"]
    col_w = 10

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("ABLATION STUDY — SUMMARY")
    lines.append("=" * 70)

    for dataset in args.datasets:
        lines.append(f"\nDataset: {dataset.upper()}")
        header = f"{'Variant':<18}" + "".join(f"{m:>{col_w}}" for m in ["Recall", "F1-macro", "AUC", "GMean"])
        lines.append(header)
        lines.append("-" * len(header))
        for name, _, _ in VARIANTS:
            m = all_results[dataset][name]
            row = f"{name:<18}" + "".join(f"{m[k]:>{col_w}.4f}" for k in metrics_order)
            lines.append(row)

    lines.append("\n" + "=" * 70)

    for line in lines:
        print(line)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Hidden dim   : {args.hidden_dim}\n")
        f.write(f"Walk length  : {args.walk_length}\n")
        f.write(f"Num walks    : {args.num_walks}\n")
        f.write(f"p / q        : {args.p} / {args.q}\n")
        for ds in args.datasets:
            lr_val = dataset_lr[ds] if dataset_lr[ds] is not None else args.lr
            ep_val = dataset_epochs[ds] if dataset_epochs[ds] is not None else args.epochs
            f.write(f"{ds:<8} lr={lr_val}  epochs={ep_val}\n")
        f.write("\n")
        for line in lines:
            f.write(line + "\n")

    print(f"\nResults saved to {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation study for FraudDetectionGNN")
    p.add_argument("--datasets",     nargs="+", default=["yelp", "amazon"],
                   choices=["yelp", "amazon", "comp"])
    p.add_argument("--data_dir",     type=str,   default="D:/毕设/data/")
    p.add_argument("--save_dir",     type=str,   default="D:/毕设/results/")
    p.add_argument("--hidden_dim",   type=int,   default=64)
    p.add_argument("--num_layers",   type=int,   default=2)
    p.add_argument("--num_heads",    type=int,   default=4)
    p.add_argument("--walk_length",  type=int,   default=10)
    p.add_argument("--num_walks",    type=int,   default=5)
    p.add_argument("--p",            type=float, default=2.0)
    p.add_argument("--q",            type=float, default=0.5)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--max_degree",   type=int,   default=50)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=5e-4)
    p.add_argument("--eval_every",    type=int,   default=5)
    p.add_argument("--cpu",           action="store_true")
    # Per-dataset overrides (None = fall back to global --lr / --epochs)
    p.add_argument("--yelp_lr",       type=float, default=None)
    p.add_argument("--yelp_epochs",   type=int,   default=None)
    p.add_argument("--amazon_lr",     type=float, default=None)
    p.add_argument("--amazon_epochs", type=int,   default=None)
    p.add_argument("--comp_lr",       type=float, default=None)
    p.add_argument("--comp_epochs",   type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(get_args())
