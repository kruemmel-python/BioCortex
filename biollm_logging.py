"""Zentrales Logging-Setup für BioCortex."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(run_name: str = "biocortex-train", *, level: int = logging.INFO) -> logging.Logger:
    """Erzeugt und konfiguriert ein Logger-Objekt für BioCortex."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{run_name}_{timestamp}.log"

    logger = logging.getLogger("BioCortex")
    if getattr(logger, "_bio_logger_configured", False):
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger._bio_logger_configured = True  # type: ignore[attr-defined]
    logger.log_path = log_file  # type: ignore[attr-defined]
    logger.run_id = f"{run_name}_{timestamp}"  # type: ignore[attr-defined]

    logger.info("=== BioCortex Logging gestartet ===")
    logger.info("Log-Datei: %s", log_file)
    return logger
