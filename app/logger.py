from __future__ import annotations

import sys
from pathlib import Path
from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru to log to console and to a rotating file under logs/.

    - Console: colored, human-friendly
    - File: logs/tradebot.log (rotation: 10 MB, retention: 7 days)
    """
    logger.remove()
    # Console sink
    logger.add(
        sys.stdout,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=False,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # File sink
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "tradebot.log"
    logger.add(
        str(log_file),
        level=level,
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
