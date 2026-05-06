"""
logger.py
─────────
Centralised loguru logger with rotation.
Usage:
    from src.utils.logger import log
    log.info("Starting augmentation pipeline")
"""

from pathlib import Path
from loguru import logger as log
from src.utils.config_loader import cfg

_log_dir = Path(cfg["paths"]["logs"])
_log_dir.mkdir(parents=True, exist_ok=True)

log.add(
    _log_dir / "system.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} — {message}",
)

__all__ = ["log"]
