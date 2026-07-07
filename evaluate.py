"""
evaluate.py — Evaluate a trained model on the held-out test set.

Usage
-----
    python evaluate.py --model autoencoder --checkpoint checkpoints/autoencoder_best.pth
    python evaluate.py --model vgg         --checkpoint checkpoints/vgg_best.pth

What it does
------------
1. Loads the checkpoint (which contains vocab, inv_vocab, epoch, val_acc).
2. Recreates the identical test split used during training (same seed=42).
3. Runs the model in eval mode over the test DataLoader.
4. Prints a per-ingredient classification report for the top-N most common
   ingredients and overall micro/macro F1.

Notes
-----
* The checkpoint is expected to have been saved by train.py in the format:
      torch.save({
          'epoch':       int,
          'val_acc':     float,
          'model_state': OrderedDict,
          'head_state':  OrderedDict,   # autoencoder only
          'vocab':       Dict[str, int],
          'inv_vocab':   Dict[int, str],
      }, path)

* Import fix: "from dataset import ..." must resolve to YOUR local
  dataset.py, NOT the pip-installed 'dataset' package.  This is guaranteed
  as long as you run:
      python evaluate.py            (not  python -m something.evaluate)
  from the project root, and the project root is first on sys.path.
  The sys.path insertion below makes this bullet-proof.
"""

import os
import sys
import json
import argparse

# ── Make sure the project root is always importable ──────────────────────────
# This prevents the "from dataset import FoodDataset" error caused by the
# pip-installed 'dataset' package shadowing your local dataset.py.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Local imports — now guaranteed to resolve to project root
from dataset import FoodDataset, get_val_transform
from models  import FoodAutoencoder, RecipeHead, MiniVGG, MiniGoogleNet, \
                    MiniAlexNet, CNN2FC


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained Inverse-Cooking model on the test set."
    )
    p.add_argument(
        "--model", type=str, required=True,
        choices=["autoencoder", "vgg", "googlenet", "alexnet", "cnn2fc"],
        help="Which model architecture to evaluate.",
    )
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the .pth checkpoint file saved by train.py.",
    )
    p.add_argument(
        "--data_dir",   type=str, default="data",
        help="Root data directory (must contain layer1.json and images/).",
    )
    p.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for the test DataLoader.",
    )
    p.add_argument(
        "--workers",    type=int, default=4,
        help="Number of DataLoader worker processes.",
    )
    p.add_argument(
        "--threshold",  type=float, default=0.5,
        help="Sigmoid threshold for multi-label prediction (default: 0.5).",
    )
    p.add_argument(
        "--top_n",      type=int, default=20,
        help="Number of most-common ingredients to include in the report.",
    )
    return p.parse_args()


# ── Model factory ────────────────────────────────────────────────────────────

def build_model(
    model_name: str,
    num_classes: int,
    device: torch.device,
    ckpt: dict,
) -> tuple:
    """
    Returns (model, head_or_None).
    For the autoencoder both model and head are loaded.
    For CNN models head is None.
    """
    is_ae = (model_name == "autoencoder")

    if is_ae:
        model = FoodAutoencoder()
        head  = RecipeHead(num_ingredients=num_classes)
        model.load_state_dict(ckpt["model_state"])
        head.load_state_dict(ckpt["head_state"])
        return model.to(device).eval(), head.to(device).eval()

    arch_map = {
        "vgg":       MiniVGG,
        "googlenet": MiniGoogleNet,
        "alexnet":   MiniAlexNet,
        "cnn2fc":    CNN2FC,
    }
    model = arch_map[model_name](num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), None


# ── Evaluation loop ──────────────────────────────────────────────────────────

def run_eval(
    model:       nn.Module,
    head:        nn.Module | None,
    loader:      DataLoader,
    device:      torch.device,
    threshold:   float,
    is_ae:       bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference; return (all_preds, all_labels) as numpy bool arrays."""
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)

            if is_ae:
                _, z   = model(imgs)
                logits = head(z)
            else:
                logits = model(imgs)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(np.int8)

            all_preds.append(preds)
            all_labels.append(labels.numpy().astype(np.int8))

    return np.vstack(all_preds), np.vstack(all_labels)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 1. Load checkpoint ──────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train a model first with:  python train.py --model <name>"
        )

    ckpt = torch.load(args.checkpoint, map_location=device)
    vocab     = ckpt["vocab"]       # Dict[str, int]
    inv_vocab = ckpt["inv_vocab"]   # Dict[int, str]
    num_classes = len(vocab)

    print(f"[Checkpoint] {args.checkpoint}")
    print(f"  Epoch   : {ckpt.get('epoch', '?')}")
    print(f"  Val acc : {ckpt.get('val_acc', 0):.2%}")
    print(f"  Classes : {num_classes}")

    # 2. Load + split recipes (same seed as train.py) ─────────────────────────
    layer1_path = os.path.join(args.data_dir, "layer1.json")
    if not os.path.exists(layer1_path):
        raise FileNotFoundError(
            f"layer1.json not found at {layer1_path}.\n"
            "Download Recipe1M+ from https://im2recipe.csail.mit.edu/ "
            "and place it in the data/ directory."
        )

    with open(layer1_path) as f:
        recipes = json.load(f)

    # Mirror the exact split from train.py
    _, temp_r = train_test_split(recipes, test_size=0.20, random_state=42)
    _, test_r = train_test_split(temp_r,  test_size=0.50, random_state=42)
    print(f"[Test set] {len(test_r):,} recipes")

    img_dir = os.path.join(args.data_dir, "images")
    test_ds = FoodDataset(test_r, img_dir, vocab, get_val_transform())

    if len(test_ds) == 0:
        print("\n[Warning] Test dataset is empty — no images found in",
              img_dir, "\nSkipping evaluation.")
        return

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Build model ──────────────────────────────────────────────────────────
    is_ae = (args.model == "autoencoder")
    model, head = build_model(args.model, num_classes, device, ckpt)
    print(f"[Model] {args.model} loaded  (is_autoencoder={is_ae})")

    # 4. Run inference ────────────────────────────────────────────────────────
    print(f"\nRunning evaluation  (threshold={args.threshold}) ...")
    all_preds, all_labels = run_eval(
        model, head, test_loader, device, args.threshold, is_ae
    )

    # 5. Reports ──────────────────────────────────────────────────────────────
    # -- Micro / Macro F1 across ALL ingredients --
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    prec     = precision_score(all_labels, all_preds, average="micro", zero_division=0)
    rec      = recall_score(   all_labels, all_preds, average="micro", zero_division=0)

    print("\n" + "=" * 60)
    print(f"  Micro  F1 : {micro_f1:.4f}")
    print(f"  Macro  F1 : {macro_f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print("=" * 60)

    # -- Per-ingredient report for top-N most frequent in the test labels --
    freq_per_class = all_labels.sum(axis=0)            # (num_classes,)
    top_idx = freq_per_class.argsort()[-args.top_n:][::-1]

    # inv_vocab keys may be stored as int or str depending on json.dump round-trip
    def _name(i: int) -> str:
        return inv_vocab.get(i, inv_vocab.get(str(i), f"idx_{i}"))

    top_names = [_name(i) for i in top_idx]

    print(f"\nPer-ingredient report — top {args.top_n} most frequent ingredients:")
    print(classification_report(
        all_labels[:, top_idx],
        all_preds[:, top_idx],
        target_names=top_names,
        zero_division=0,
    ))


if __name__ == "__main__":
    main()