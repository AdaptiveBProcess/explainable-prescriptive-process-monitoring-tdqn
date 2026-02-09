from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import load_yaml


@dataclass
class Config:
    raw: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        return cls(raw=load_yaml(path))



