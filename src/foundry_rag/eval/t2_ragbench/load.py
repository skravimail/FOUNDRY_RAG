"""Load T²-RAGBench pilot / oracle JSONL artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from foundry_rag.paths import DATA_ROOT

DEFAULT_PILOT = DATA_ROOT / "t2_ragbench" / "pilot.jsonl"
DEFAULT_ORACLE = DATA_ROOT / "t2_ragbench" / "oracle_contexts.jsonl"


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def load_pilot(path: Path | str | None = None) -> list[dict[str, Any]]:
    return load_jsonl(path or DEFAULT_PILOT)


def load_oracle_by_id(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path or DEFAULT_ORACLE)
    return {row["id"]: row for row in rows}


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
