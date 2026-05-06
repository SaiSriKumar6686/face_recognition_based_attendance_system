"""
train_bootstrap.py
───────────────────
Kick off Phase 1 bootstrap training on the synthetic dataset.

Run AFTER dataset_builder.py has populated data/synthetic/.

Usage
─────
    python scripts/train_bootstrap.py
    python scripts/train_bootstrap.py --resume models/checkpoints/bootstrap_epoch010.pt
"""

import argparse
from src.training.trainer import train_bootstrap
from src.utils.logger import log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    log.info("Starting Phase 1 bootstrap training…")
    train_bootstrap(resume_checkpoint=args.resume)
