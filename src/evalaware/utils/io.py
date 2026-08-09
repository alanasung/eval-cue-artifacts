"""JSON and JSONL persistence for append-only run records.

JSONL is the format for anything a long run produces incrementally: LLM call
logs, per-example judgements, cache manifests. It has one property that matters
more than any other here -- a run killed at hour six leaves a file whose first
N lines are still valid, whereas a killed JSON array write leaves a file that
parses as nothing at all.

That property is only real if readers honour it, so :func:`read_jsonl` tolerates
a truncated final line by default. It counts what it skipped and logs a warning
rather than staying quiet, because "my file has 999 rows instead of 1000" is a
fact the reader has to know before averaging over it.

Whole-file JSON writes go through :func:`~..utils.run_manifest.write_json_atomic`
so the same crash cannot corrupt a results file. It is re-exported here so that
callers have one import for all persistence.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .run_manifest import write_json_atomic

__all__ = [
    "append_jsonl",
    "count_jsonl",
    "ensure_dir",
    "iter_jsonl",
    "load_json",
    "read_jsonl",
    "save_json",
    "write_jsonl",
]

logger = logging.getLogger(__name__)

save_json = write_json_atomic


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if absent, and return it.

    Raises:
        NotADirectoryError: If the path exists as a file. Left alone, the caller
            gets a confusing permission-style error on the first write instead.
    """
    target = Path(path)
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(
            f"{target} exists but is a file, not a directory; "
            "remove it or point the config at a different path"
        )
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_json(path: str | Path, *, default: Any = None) -> Any:
    """Read a JSON file, returning ``default`` when it does not exist.

    Args:
        path: File to read.
        default: Returned for a missing file. A missing file is normal on the
            first run; malformed contents never are.

    Raises:
        ValueError: If the file exists but does not parse, with the byte offset
            so the damage can be found without re-reading the whole file.
    """
    target = Path(path)
    if not target.is_file():
        return default
    text = target.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{target} is not valid JSON (at line {exc.lineno}, column {exc.colno}): {exc.msg}; "
            "the file was likely truncated by a killed run, so delete it and re-run the stage"
        ) from exc


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> Path:
    """Append one record. The only write path for incremental logs.

    Each call opens, writes, and closes so that a process killed between calls
    leaves a complete file rather than a buffer that never flushed.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), default=str) + "\n")
    return target


def write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    append: bool = True,
) -> Path:
    """Write many records, appending by default.

    Appending is the default deliberately: overwriting an incremental log is
    almost always a mistake, and making the destructive case explicit means it
    has to be typed out.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), default=str) + "\n")
    return target


def iter_jsonl(path: str | Path, *, tolerate_truncated: bool = True) -> Iterator[dict[str, Any]]:
    """Yield records one at a time, so a large log never has to fit in memory.

    Args:
        path: File to read. A missing file yields nothing, matching the
            "no rows logged yet" case.
        tolerate_truncated: Skip lines that do not parse. Only the final line of
            a JSONL file can be truncated by a crash; a broken line in the
            middle means something else went wrong.

    Raises:
        ValueError: If a line does not parse and ``tolerate_truncated`` is off.
    """
    target = Path(path)
    if not target.is_file():
        return
    with target.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                if not tolerate_truncated:
                    raise ValueError(
                        f"{target} line {lineno} is not valid JSON: {exc.msg}; "
                        "pass tolerate_truncated=True to skip damaged lines"
                    ) from exc
                logger.warning(
                    "skipping unparseable line %d of %s (%s); "
                    "this is expected only for the final line of a killed run",
                    lineno,
                    target,
                    exc.msg,
                )


def read_jsonl(path: str | Path, *, tolerate_truncated: bool = True) -> list[dict[str, Any]]:
    """Read every record into a list. Convenience over :func:`iter_jsonl`."""
    return list(iter_jsonl(path, tolerate_truncated=tolerate_truncated))


def count_jsonl(path: str | Path) -> int:
    """Number of parseable records, without materialising them.

    The cheap way to answer "how far did the run get" before deciding whether
    to resume or restart.
    """
    return sum(1 for _ in iter_jsonl(path))
