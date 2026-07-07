"""
train.py — Train any model in the Inverse Cooking project.

Usage
-----
    # Autoencoder + ingredient head
    python train.py --model autoencoder --epochs 50 --lr 1e-3

    # Classification CNNs (multi-label ingredient prediction)
    python train.py --model vgg       --epochs 50
    python train.py --model googlenet --epochs 40
    python train.py --model alexnet   --epochs 40
    python train.py --model cnn2fc    --epochs 30

Checkpoint format (compatible with evaluate.py)
-----------------------------------------------
    {
        'epoch':       int,
        'val_acc':     float,           # micro-F1 used as "accuracy" here
        'model_state': OrderedDict,
        'head_state':  OrderedDict,     # autoencoder only; absent for CNNs
        'vocab':       Dict[str, int],
        'inv_vocab':   Dict[int, str],
    }
"""

import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

from dataset import FoodDataset, get_train_transform, get_val_transform, build_vocab
from models  import FoodAutoencoder, RecipeHead, MiniVGG, MiniGoogleNet, \
                    MiniAlexNet, CNN2FC


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",      type=str, default="autoencoder",
                   choices=["autoencoder", "vgg", "googlenet", "alexnet", "cnn2fc"])
    p.add_argument("--data_dir",   type=str, default="data")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--latent_dim", type=int, default=256,
                   help="Autoencoder latent dimension (ignored for CNNs).")
    p.add_argument("--min_freq",   type=int, default=10,
                   help="Minimum ingredient frequency to include in vocab.")
    p.add_argument("--servings",   type=int, default=2,
                   help="Default serving count used in predict.py (stored in ckpt).")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_loaders(
    train_r, val_r, img_dir, vocab, batch_size, workers
) -> tuple[DataLoader, DataLoader]:
    train_ds = FoodDataset(train_r, img_dir, vocab, get_train_transform())
    val_ds   = FoodDataset(val_r,   img_dir, vocab, get_val_transform())

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin)
    return train_loader, val_loader


def save_checkpoint(
    path:        str,
    epoch:       int,
    val_acc:     float,
    model:       nn.Module,
    vocab:       dict,
    inv_vocab:   dict,
    head:        nn.Module | None = None,
) -> None:
    ckpt = {
        "epoch":       epoch,
        "val_acc":     val_acc,
        "model_state": model.state_dict(),
        "vocab":       vocab,
        "inv_vocab":   inv_vocab,
    }
    if head is not None:
        ckpt["head_state"] = head.state_dict()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(ckpt, path)


# ── Training loops ────────────────────────────────────────────────────────────

def train_one_epoch_ae(
    ae, head, loader, optimizer, ae_criterion, head_criterion, device
) -> float:
    ae.train(); head.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        recon, z = ae(imgs)
        loss_recon = ae_criterion(recon, imgs)
        loss_ingr  = head_criterion(head(z), labels)
        loss = loss_recon + loss_ingr
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def train_one_epoch_cnn(
    model, loader, optimizer, criterion, device
) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, head, loader, device, threshold=0.5, is_ae=False) -> float:
    model.eval()
    if head:
        head.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        if is_ae:
            _, z   = model(imgs)
            logits = head(z)
        else:
            logits = model(imgs)
        preds = (torch.sigmoid(logits).cpu().numpy() > threshold).astype(int)
        all_preds.append(preds)
        all_labels.append(labels.numpy().astype(int))
    return f1_score(
        np.vstack(all_labels), np.vstack(all_preds),
        average="micro", zero_division=0
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}  |  Model: {args.model}")

    # 1. Load recipes + build vocab
    layer1_path = os.path.join(args.data_dir, "layer1.json")
    if not os.path.exists(layer1_path):
        raise FileNotFoundError(
            f"{layer1_path} not found.\n"
            "Download Recipe1M+ from https://im2recipe.csail.mit.edu/"
        )
    with open(layer1_path) as f:
        recipes = json.load(f)

    vocab, inv_vocab = build_vocab(recipes, min_freq=args.min_freq)
    num_classes = len(vocab)
    print(f"[Vocab] {num_classes} ingredients (min_freq={args.min_freq})")

    # 2. Split: 80 % train | 10 % val | 10 % test  (mirror in evaluate.py)
    train_r, temp_r = train_test_split(recipes, test_size=0.20, random_state=42)
    val_r,   _      = train_test_split(temp_r,  test_size=0.50, random_state=42)
    print(f"[Split] train={len(train_r):,}  val={len(val_r):,}")

    img_dir = os.path.join(args.data_dir, "images")
    train_loader, val_loader = make_loaders(
        train_r, val_r, img_dir, vocab, args.batch_size, args.workers
    )

    # 3. Build model(s)
    is_ae = (args.model == "autoencoder")
    if is_ae:
        model = FoodAutoencoder(latent_dim=args.latent_dim).to(device)
        head  = RecipeHead(latent_dim=args.latent_dim,
                           num_ingredients=num_classes).to(device)
        params     = list(model.parameters()) + list(head.parameters())
        ae_crit    = nn.MSELoss()
        head_crit  = nn.BCEWithLogitsLoss()
    else:
        arch_map = {
            "vgg":       MiniVGG,
            "googlenet": MiniGoogleNet,
            "alexnet":   MiniAlexNet,
            "cnn2fc":    CNN2FC,
        }
        model = arch_map[args.model](num_classes=num_classes).to(device)
        head  = None
        params    = model.parameters()
        head_crit = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, verbose=True
    )

    ckpt_path = os.path.join("checkpoints", f"{args.model}_best.pth")
    best_val  = 0.0

    # 4. Training loop
    for epoch in range(1, args.epochs + 1):
        if is_ae:
            loss = train_one_epoch_ae(
                model, head, train_loader, optimizer,
                ae_crit, head_crit, device
            )
        else:
            loss = train_one_epoch_cnn(
                model, train_loader, optimizer, head_crit, device
            )

        val_acc = validate(model, head, val_loader, device, is_ae=is_ae)
        scheduler.step(val_acc)

        flag = ""
        if val_acc > best_val:
            best_val = val_acc
            save_checkpoint(ckpt_path, epoch, val_acc,
                            model, vocab, inv_vocab, head)
            flag = "  ← saved"

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss {loss:.4f} | val-microF1 {val_acc:.4f}{flag}")

    print(f"\nTraining complete. Best val micro-F1: {best_val:.4f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()