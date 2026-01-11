"""Utility functions and project-wide configuration ."""

from pathlib import Path
import os
import random
from typing import Optional

import numpy as np


# Resolve project root as the parent of the src/ directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RANDOM_SEED = 42

# Ensure key directories exist
for _dir in (FIGURES_DIR, OUTPUTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_figure(name: str) -> Path:
    """Save current matplotlib figure to the figures/ folder and return the path.

    """

    import matplotlib.pyplot as plt 

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = FIGURES_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    return filepath


def get_output_path(filename: str) -> Path:
    """Return a path inside the outputs/ directory for the given filename."""

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR / filename
