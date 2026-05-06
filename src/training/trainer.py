"""
trainer.py
──────────
Main training loop.

Handles:
•  Phase 1 bootstrap training on synthetic dataset
•  Phase 3 incremental fine-tuning on mixed dataset (triggered by continual_learning)
•  Cosface loss warmup → AdaFace loss switch
•  EWC penalty integration (imported from continual_learning module)
•  Checkpoint saving every epoch + best-model tracking
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config_loader import cfg
from src.utils.logger import log
from src.training.adaface_model import build_model, save_checkpoint, freeze_except_last_n
from src.training.dataset import SyntheticFaceDataset, MixedFaceDataset

_TCFG  = cfg["training"]
_CKDIR = Path(cfg["paths"]["checkpoints"])


def _cosface_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                  num_classes: int, s: float = 64.0, m: float = 0.35) -> torch.Tensor:
    """Simple CosFace loss for warmup phase before switching to AdaFace loss."""
    cos_theta = embeddings  # assume embeddings are already L2-normalised
    one_hot = torch.zeros_like(cos_theta)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    cos_theta = s * (cos_theta - one_hot * m)
    return nn.CrossEntropyLoss()(cos_theta, labels)


def train_bootstrap(resume_checkpoint: str = None) -> None:
    """
    Phase 1 — Train from scratch on synthetic dataset.
    Call once after running scripts/dataset_builder.py.
    """
    dataset = SyntheticFaceDataset()
    loader  = DataLoader(
        dataset,
        batch_size=_TCFG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model    = build_model(num_classes=dataset.num_classes)
    dev      = next(model.parameters()).device
    optimizer = torch.optim.Adam(
        model.parameters(), lr=_TCFG["learning_rate"], weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_TCFG["epochs"]
    )

    start_epoch = 0
    if resume_checkpoint:
        from src.training.adaface_model import load_checkpoint
        start_epoch = load_checkpoint(model, resume_checkpoint) + 1

    best_loss = float("inf")

    for epoch in range(start_epoch, _TCFG["epochs"]):
        model.train()
        total_loss = 0.0
        use_cosface = epoch < _TCFG["warmup_epochs"]
        loss_name   = "CosFace" if use_cosface else "AdaFace"

        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1} [{loss_name}]"):
            imgs, labels = imgs.to(dev), labels.to(dev)
            embeddings = model(imgs)

            # loss selection
            if use_cosface or not _TCFG.get("cosface_warmup"):
                loss = _cosface_loss(embeddings, labels, dataset.num_classes)
            else:
                # TODO: plug in AdaFace loss once third_party/adaface is installed
                # from head import AdaFaceLoss
                # loss = adaface_loss(embeddings, norms, labels)
                loss = _cosface_loss(embeddings, labels, dataset.num_classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()
        log.info(f"Epoch {epoch+1}/{_TCFG['epochs']} | loss={avg_loss:.4f}")

        # save checkpoint
        ckpt_path = _CKDIR / f"bootstrap_epoch{epoch+1:03d}.pt"
        save_checkpoint(model, ckpt_path, epoch=epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, _CKDIR / "bootstrap_best.pt", epoch=epoch)
            log.info(f"  ↳ New best model saved (loss={best_loss:.4f})")

    log.info("Bootstrap training complete.")


def fine_tune_incremental(ewc_penalty=None) -> None:
    """
    Phase 3 — Incremental fine-tune on mixed dataset.
    Only last `fine_tune_layers` layers are trained.
    Optional EWC penalty passed in from continual_learning module.
    """
    dataset = MixedFaceDataset()
    loader  = DataLoader(
        dataset,
        batch_size=_TCFG["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model = build_model(num_classes=dataset.num_classes)

    # load best bootstrap checkpoint as starting point
    best_ckpt = _CKDIR / "bootstrap_best.pt"
    if best_ckpt.exists():
        from src.training.adaface_model import load_checkpoint
        load_checkpoint(model, best_ckpt)
    else:
        log.warning("No bootstrap checkpoint found — fine-tuning from scratch.")

    # freeze all but last N layers
    freeze_except_last_n(model, n=cfg["continual_learning"]["fine_tune_layers"])

    dev       = next(model.parameters()).device
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=_TCFG["learning_rate"] * 0.1,
    )

    ft_epochs = max(5, _TCFG["epochs"] // 4)
    for epoch in range(ft_epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(loader, desc=f"FineTune epoch {epoch+1}"):
            imgs, labels = imgs.to(dev), labels.to(dev)
            embeddings   = model(imgs)
            loss         = _cosface_loss(embeddings, labels, dataset.num_classes)

            # add EWC penalty if provided
            if ewc_penalty is not None:
                loss = loss + ewc_penalty(model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        log.info(f"FineTune epoch {epoch+1}/{ft_epochs} | loss={avg_loss:.4f}")

    save_checkpoint(model, _CKDIR / "finetune_latest.pt", epoch=ft_epochs)
    log.info("Incremental fine-tuning complete.")


if __name__ == "__main__":
    train_bootstrap()
