"""
ewc.py  —  Elastic Weight Consolidation
─────────────────────────────────────────
EWC prevents catastrophic forgetting when incrementally fine-tuning
the face recognition model on new student data.

How it works
────────────
After training on the existing dataset, we compute the Fisher Information
Matrix (FIM) diagonal for each parameter.  During subsequent fine-tuning,
a penalty proportional to FIM × (θ - θ*)² is added to the loss, making
the optimiser reluctant to change parameters important for existing identities.

References
──────────
Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural
networks." PNAS.

Usage
-----
    from src.continual_learning.ewc import EWC
    ewc = EWC(model, dataset_loader, device)
    # ... fine-tune ...
    loss = task_loss + ewc.penalty(model)
"""

import copy
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config_loader import cfg
from src.utils.logger import log

_EWC_LAMBDA = cfg["continual_learning"]["ewc_lambda"]


class EWC:
    """
    Elastic Weight Consolidation regulariser.

    Parameters
    ----------
    model    : the trained nn.Module (before incremental fine-tuning starts).
    loader   : DataLoader for the existing training set (used to compute FIM).
    device   : 'cuda' or 'cpu'.
    lam      : EWC regularisation strength (default from config).
    n_batches: number of batches to use for FIM estimation (fewer = faster).
    """

    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: str = "cpu",
        lam: float = _EWC_LAMBDA,
        n_batches: int = 50,
    ):
        self.lam    = lam
        self.device = device

        # θ*  — optimal parameters before fine-tuning
        self.params_star: dict[str, torch.Tensor] = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        # diagonal FIM
        self.fisher: dict[str, torch.Tensor] = self._compute_fisher(
            model, loader, device, n_batches
        )
        log.info(f"EWC: Fisher diagonal computed over {n_batches} batches.")

    # ── Fisher estimation ────────────────────────────────────────────

    @staticmethod
    def _compute_fisher(
        model: nn.Module,
        loader: DataLoader,
        device: str,
        n_batches: int,
    ) -> dict[str, torch.Tensor]:
        model.eval()
        fisher = {
            n: torch.zeros_like(p.data)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="EWC Fisher")):
            if batch_idx >= n_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            model.zero_grad()
            output = model(imgs)
            # use log-likelihood as the loss proxy
            loss = nn.CrossEntropyLoss()(output, labels)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.clone().pow(2)

        # average
        for n in fisher:
            fisher[n] /= n_batches

        return fisher

    # ── Penalty ──────────────────────────────────────────────────────

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC penalty for the current model weights.

        Returns a scalar tensor to add to the task loss.
        """
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in model.named_parameters():
            if n in self.fisher:
                diff   = p - self.params_star[n].to(p.device)
                fisher = self.fisher[n].to(p.device)
                penalty += (fisher * diff.pow(2)).sum()
        return (self.lam / 2.0) * penalty
