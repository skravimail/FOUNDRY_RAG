"""Repo-relative data path helpers."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"


def data_dir(module_folder: str) -> Path:
    path = DATA_ROOT / module_folder
    if not path.is_dir():
        raise FileNotFoundError(f"Missing data directory: {path}")
    return path


def data_file(module_folder: str, filename: str) -> Path:
    path = data_dir(module_folder) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing data file: {path}")
    return path
