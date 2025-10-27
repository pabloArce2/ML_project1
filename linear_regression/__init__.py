"""
Utilities for fitting ridge regression models on the SAheart dataset.

Run the experiment entrypoint in ``run.py`` to reproduce the 10-fold cross-validation
experiments across multiple target configurations and regularisation strengths.
"""

from .run import main  # re-export the CLI entrypoint for convenience

__all__ = ["main"]
