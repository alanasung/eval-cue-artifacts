"""Git provenance helpers.

Every result-writing path in this repo calls :func:`git_sha`. A number without a
commit attached cannot be reproduced, and "which version of the code produced
this" is the question that gets asked months later when a result is questioned.

These functions never raise on a missing or broken git installation: running
outside a repository is a legitimate situation (a fresh tarball, a CI cache),
and it should degrade to ``"unknown"`` rather than kill a run that has already
spent an hour computing.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = ["GitState", "git_is_dirty", "git_sha", "git_state"]

_UNKNOWN = "unknown"
_TIMEOUT_S = 5


def _run_git(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            cwd=str(cwd) if cwd else None,
        )
    except (OSError, subprocess.SubprocessError):
        # git missing, not executable, or hung. Provenance is best-effort.
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def git_sha(*, short: bool = False, cwd: Path | None = None) -> str:
    """Return the current commit SHA, or ``"unknown"`` outside a repository."""
    args = ["rev-parse", "--short", "HEAD"] if short else ["rev-parse", "HEAD"]
    return _run_git(args, cwd) or _UNKNOWN


def git_is_dirty(*, cwd: Path | None = None) -> bool:
    """True if the working tree has uncommitted changes.

    A dirty tree means the recorded SHA does not fully describe the code that
    ran, which is worth flagging in the manifest.
    """
    out = _run_git(["status", "--porcelain"], cwd)
    return bool(out) if out is not None else False


@dataclass(frozen=True)
class GitState:
    """Snapshot of repository state at the moment a result was produced."""

    sha: str
    branch: str
    dirty: bool

    @property
    def is_reproducible(self) -> bool:
        """A result is only exactly reproducible from a clean, known commit."""
        return self.sha != _UNKNOWN and not self.dirty

    def to_dict(self) -> dict[str, object]:
        return {
            "sha": self.sha,
            "branch": self.branch,
            "dirty": self.dirty,
            "is_reproducible": self.is_reproducible,
        }


def git_state(*, cwd: Path | None = None) -> GitState:
    """Collect commit, branch, and dirtiness in one call."""
    return GitState(
        sha=git_sha(cwd=cwd),
        branch=_run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd) or _UNKNOWN,
        dirty=git_is_dirty(cwd=cwd),
    )
