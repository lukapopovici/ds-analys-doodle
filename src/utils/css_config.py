"""Helpers to manage CSS theme configs stored under `src/configs/css`.

Functions:
- list_css_configs() -> List of available config names (stems)
- load_css(name) -> CSS text for a given config name (stem)
- save_uploaded_css(name, content) -> saves uploaded content as a named css file

"""
from __future__ import annotations

from pathlib import Path
from typing import List
import re

CSS_DIR = Path(__file__).resolve().parent.parent / "configs" / "css"
CSS_DIR.mkdir(parents=True, exist_ok=True)


def list_css_configs() -> List[str]:
    """Return list of available css config names (without .css extension)."""
    return sorted([p.stem for p in CSS_DIR.glob("*.css")])


def load_css(name: str) -> str:
    """Load CSS text by name (stem). Raises FileNotFoundError if missing."""
    candidate = next((p for p in CSS_DIR.glob("*.css") if p.stem == name), None)
    if candidate is None:
        raise FileNotFoundError(f"CSS config '{name}' not found in {CSS_DIR}")
    return candidate.read_text(encoding="utf-8")


def save_uploaded_css(name: str, content: str) -> Path:
    """Save uploaded CSS content to the configs directory using a sanitized name.

    Returns the Path to the saved file.
    """
    if not name:
        raise ValueError("Name must be provided to save CSS config")
    # sanitize name
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    if not safe.lower().endswith(".css"):
        safe = f"{safe}.css"

    path = CSS_DIR / safe
    path.write_text(content, encoding="utf-8")
    return path


__all__ = ["CSS_DIR", "list_css_configs", "load_css", "save_uploaded_css"]
