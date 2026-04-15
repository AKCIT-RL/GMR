"""JSON-serializable report helpers."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Convert nested dict/list with numpy types to JSON-safe structures."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (float, int, np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (float, complex)):
        return float(obj)
    return str(obj)


def save_report(path: str, report: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(report), f, indent=2)
