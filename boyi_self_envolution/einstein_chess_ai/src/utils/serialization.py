"""JSON/pickle serialization helpers for game records and replay buffers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save JSON with UTF-8 encoding."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON data."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_pickle(data: Any, path: str | Path) -> None:
    """Save pickle data."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump(data, fh)


def load_pickle(path: str | Path) -> Any:
    """Load pickle data."""
    with Path(path).open("rb") as fh:
        return pickle.load(fh)

