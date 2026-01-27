"""Centralized logging configuration for llm_integration.

All modules should import logger from here:
    from ..logging_config import logger
    # or
    from llm_integration.logging_config import logger
"""

import logging
import os
from pathlib import Path

# Determine log directory
LOG_DIR = Path(os.getenv("RESULTS_DIR", "results/llm_iterations"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Single log file for all llm_integration modules
LOG_FILE = LOG_DIR / "llm.log"

# Configure the root logger for llm_integration
logger = logging.getLogger("llm_integration")
logger.setLevel(logging.INFO)  # Only INFO and above (no DEBUG spam)

# Prevent duplicate handlers if module is imported multiple times
if not logger.handlers:
    # File handler - concise format
    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
