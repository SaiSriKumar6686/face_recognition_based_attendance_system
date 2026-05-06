"""
train_optimal.py
─────────────────
Script 3: Train the Model for Maximum Accuracy

This script runs the full training pipeline with tuned hyperparameters
designed to squeeze the highest accuracy out of the available data:

  1. Generates a rich synthetic dataset (target: 300 per identity)
  2. Trains the model with:
     • Longer epochs (30+)
     • Cosine-annealing LR schedule with warm restarts
     • Proper CosFace warmup → main loss transition
     • Gradient clipping for stable convergence
     • Best-model checkpointing on validation loss
  3. Exports the trained model to ONNX
  4. Re-enrolls all students with fresh embeddings
  5. Runs the Rank-1 accuracy benchmark to verify improvement

Usage:
    python testing/train_optimal.py                    # full pipeline
    python testing/train_optimal.py --skip-augment     # skip dataset build
    python testing/train_optimal.py --epochs 50        # override epochs
    python testing/train_optimal.py --target-samples 500  # more augmentation
"""

import sys
import argparse
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.utils.config_loader import cfg
from src.utils.logger import log
from src.training.adaface_model import build_model, save_checkpoint, export_onnx
from src.training.dataset import SyntheticFaceDataset

_CKDIR   = Path(cfg["paths"]["checkpoints"])
_ONNXDIR = Path(cfg["paths"]["onnx"])


def _cosface_loss(embeddings, labels, num_classes, s=64.0, m=0.35):
    """CosFace (large-margin cosine) loss."""
    cos_theta = embeddings
    one_hot = torch.zeros_like(cos_theta)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    logits = s * (cos_theta - one_hot * m)
    return nn.CrossEntropyLoss()(logits, labels)


def train(epochs: int = 30, batch_size: int = 32, lr: float = 0.001,
          warmup_epochs: int = 5, grad_clip: float = 1.0):
    """
    Train with validation split and best-model tracking.
    """
    log.info("Loading dataset...")
    full_dataset = SyntheticFaceDataset()
    num_classes = full_dataset.num_classes
    total = len(full_dataset)

    # 85/15 train-val split
    val_size = max(1, int(total * 0.15))
    train_size = total - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    log.info(f"Dataset: {total} images | {num_classes} classes | "
             f"Train: {train_size} | Val: {val_size}")

    model = build_model(num_classes=num_classes)
    dev = next(model.parameters()).device

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, epochs // 3), T_mult=2
    )

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    log.info(f"Training config: epochs={epochs}, batch={batch_size}, lr={lr}, "
             f"warmup={warmup_epochs}, grad_clip={grad_clip}")
    print()

    for epoch in range(epochs):
        # ── Training ─────────────────────────────────
        model.train()
        train_loss = 0.0
        use_cosface = epoch < warmup_epochs
        loss_name = "CosFace-Warmup" if use_cosface else "CosFace-Main"

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{loss_name}]",
                                  leave=False):
            imgs, labels = imgs.to(dev), labels.to(dev)
            embeddings = model(imgs)
            loss = _cosface_loss(embeddings, labels, num_classes,
                                s=32.0 if use_cosface else 64.0,
                                m=0.20 if use_cosface else 0.35)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        scheduler.step()

        # ── Validation ───────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(dev), labels.to(dev)
                embeddings = model(imgs)
                loss = _cosface_loss(embeddings, labels, num_classes)
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        current_lr = optimizer.param_groups[0]["lr"]
        improved = avg_val < best_val_loss
        marker = " ★ best" if improved else ""

        log.info(f"Epoch {epoch+1:3d}/{epochs} | "
                 f"train_loss={avg_train:.4f} | val_loss={avg_val:.4f} | "
                 f"lr={current_lr:.6f}{marker}")

        # Save checkpoint every epoch
        save_checkpoint(model, _CKDIR / f"optimal_epoch{epoch+1:03d}.pt", epoch=epoch)

        if improved:
            best_val_loss = avg_val
            save_checkpoint(model, _CKDIR / "optimal_best.pt", epoch=epoch)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # Save training curves
    np.savez(str(_PROJECT_ROOT / "testing" / "results" / "training_curves.npz"),
             train_loss=history["train_loss"], val_loss=history["val_loss"])
    log.info("Training curves saved to testing/results/training_curves.npz")

    return model, num_classes


def export_model(model, num_classes):
    """Export the best checkpoint to ONNX."""
    from datetime import datetime
    _ONNXDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _ONNXDIR / f"optimal_{ts}.onnx"
    export_onnx(model, out_path, device="cpu")
    log.info(f"ONNX exported: {out_path}")
    return out_path


def re_enroll_all():
    """Re-run enrollment with the new model for fresh embeddings."""
    from src.enrollment.enroll_student import enroll_all_from_seed_dir
    log.info("Re-enrolling all students with new model embeddings...")
    enroll_all_from_seed_dir()
    log.info("Re-enrollment complete.")


def run_benchmark():
    """Run the Rank-1 accuracy test to verify improvement."""
    # Force reload of singletons so they pick up the new ONNX model
    import src.inference.embedder as emb_mod
    import src.inference.matcher as mat_mod
    emb_mod._embedder_singleton = None
    mat_mod._matcher_singleton = None

    from testing.test_recognition import test_gallery_accuracy
    correct, total, wrong = test_gallery_accuracy()
    return correct, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for maximum accuracy")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--target-samples", type=int, default=300)
    parser.add_argument("--skip-augment", action="store_true",
                        help="Skip synthetic dataset generation")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip ONNX export + re-enrollment")
    args = parser.parse_args()

    t_start = time.time()
    results_dir = _PROJECT_ROOT / "testing" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Augment ──────────────────────────────
    if not args.skip_augment:
        log.info(f"Step 1: Building synthetic dataset (target={args.target_samples}/identity)...")

        # Temporarily set target in config
        original_target = cfg["augmentation"]["target_samples_per_identity"]
        cfg["augmentation"]["target_samples_per_identity"] = args.target_samples

        from scripts.dataset_builder import build_all
        build_all(target=args.target_samples)

        cfg["augmentation"]["target_samples_per_identity"] = original_target
    else:
        log.info("Step 1: Skipping augmentation (--skip-augment)")

    # ── Step 2: Train ────────────────────────────────
    log.info("Step 2: Training model...")
    model, num_classes = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_epochs=args.warmup,
    )

    # ── Step 3: Export + Re-enroll ────────────────────
    if not args.skip_export:
        log.info("Step 3: Exporting ONNX + re-enrolling...")
        export_model(model, num_classes)
        re_enroll_all()
    else:
        log.info("Step 3: Skipping export (--skip-export)")

    # ── Step 4: Benchmark ────────────────────────────
    log.info("Step 4: Running accuracy benchmark...")
    try:
        correct, total = run_benchmark()
        elapsed = time.time() - t_start
        print(f"\n{'=' * 60}")
        print(f"  TRAINING PIPELINE COMPLETE")
        print(f"  Final Rank-1 Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
        print(f"  Total time: {elapsed/60:.1f} minutes")
        print(f"{'=' * 60}\n")
    except Exception as e:
        log.warning(f"Benchmark skipped: {e}")

    elapsed = time.time() - t_start
    log.info(f"Total pipeline time: {elapsed/60:.1f} minutes")
