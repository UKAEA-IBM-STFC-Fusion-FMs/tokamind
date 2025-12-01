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

`get_repo_root()` walks up from `paths.py` to return `repo_root`.
"""

from pathlib import Path

def get_repo_root() -> Path:
    # .../multi-modal-transformer/src/mmt/utils/paths.py → repo root
    return Path(__file__).resolve().parents[3]