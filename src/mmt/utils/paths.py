"""
Path utilities for the multi-modal-transformer project.

This module provides small helpers to locate important filesystem roots
without hard-coding absolute paths.

Assumed layout (relative to this file):

    repo_root/
      src/
        mmt/
          utils/
            paths.py   <-- this file
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
