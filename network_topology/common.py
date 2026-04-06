from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Callable


def normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def to_builtin(value: Any, *, max_depth: int = 10, _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()
    if max_depth < 0:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    object_id = id(value)
    if object_id in _seen:
        return "<recursive-ref>"

    if dataclasses.is_dataclass(value):
        _seen.add(object_id)
        result = {
            field.name: to_builtin(getattr(value, field.name), max_depth=max_depth - 1, _seen=_seen)
            for field in dataclasses.fields(value)
        }
        _seen.remove(object_id)
        return result

    if isinstance(value, Mapping):
        _seen.add(object_id)
        result = {
            str(key): to_builtin(item, max_depth=max_depth - 1, _seen=_seen)
            for key, item in value.items()
        }
        _seen.remove(object_id)
        return result

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        _seen.add(object_id)
        result = [to_builtin(item, max_depth=max_depth - 1, _seen=_seen) for item in value]
        _seen.remove(object_id)
        return result

    if hasattr(value, "model_dump") and callable(value.model_dump):
        return to_builtin(value.model_dump(), max_depth=max_depth - 1, _seen=_seen)

    if hasattr(value, "dict") and callable(value.dict):
        return to_builtin(value.dict(), max_depth=max_depth - 1, _seen=_seen)

    if hasattr(value, "__dict__"):
        _seen.add(object_id)
        result = {
            key: to_builtin(item, max_depth=max_depth - 1, _seen=_seen)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
        _seen.remove(object_id)
        return result

    return repr(value)


def deep_find_first(
    data: Any,
    candidate_keys: Sequence[str],
    predicate: Callable[[Any], bool] | None = None,
) -> Any:
    normalized_candidates = {normalize_key(key) for key in candidate_keys}

    def _search(node: Any) -> Any:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if normalize_key(key) in normalized_candidates and (predicate is None or predicate(value)):
                    return value
            for value in node.values():
                match = _search(value)
                if match is not None:
                    return match
            return None
        if isinstance(node, list):
            for value in node:
                match = _search(value)
                if match is not None:
                    return match
        return None

    return _search(data)


def stable_signature(data: Any) -> str:
    serialized = json.dumps(to_builtin(data), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
