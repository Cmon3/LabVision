"""Dataset loaders for the benchmark (PLAN.md T5).

Only stdlib. Never reads real data here — callers pass a root dir.
Conventions (documented so T8 can rely on them):
- BBBC005: keep the in-focus ``F1`` subset (configurable via ``plane``);
  ground-truth count is parsed from the filename, trying in order:
  ``count300`` / ``count_300``, ``300cells``, ``_c300``, trailing
  ``_300`` before the extension. No match -> ValueError.
- AGAR: list images; ground truth comes from an optional caller-supplied
  lookup mapping filename (or stem) -> count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

IMAGE_SUFFIXES = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg"})

_COUNT_PATTERNS = (
    re.compile(r"count[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)[_-]?cells?", re.IGNORECASE),
    re.compile(r"_c(\d+)"),
    re.compile(r"_(\d+)$"),
)


@dataclass(frozen=True)
class Sample:
    path: Path
    count: int | None


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def list_image_files(root: str | Path, recursive: bool = False) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")
    entries = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in entries if is_image_file(p))


def matches_plane(name: str, plane: str = "F1") -> bool:
    return re.search(r"[_-]" + re.escape(plane) + r"(?![0-9])", name) is not None


def parse_bbbc005_count(name: str) -> int:
    stem = Path(name).stem
    for pattern in _COUNT_PATTERNS:
        m = pattern.search(stem)
        if m:
            return int(m.group(1))
    raise ValueError(f"no ground-truth count in filename: {name!r}")


def list_bbbc005(root: str | Path, plane: str = "F1") -> list[Sample]:
    """List BBBC005 images for one focus plane with filename counts."""
    return [
        Sample(path=p, count=parse_bbbc005_count(p.name))
        for p in list_image_files(root)
        if matches_plane(p.name, plane)
    ]


def lookup_count(name: str, counts: Mapping[str, int] | None) -> int | None:
    if not counts:
        return None
    if name in counts:
        return counts[name]
    return counts.get(Path(name).stem)


def list_agar(
    root: str | Path,
    counts: Mapping[str, int] | None = None,
    recursive: bool = False,
) -> list[Sample]:
    """List AGAR images, attaching counts from an optional lookup."""
    return [Sample(path=p, count=lookup_count(p.name, counts))
            for p in list_image_files(root, recursive)]
