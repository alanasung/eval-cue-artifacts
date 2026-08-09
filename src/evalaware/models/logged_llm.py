"""Transparent logging wrapper for any LLM used inside the measurement instrument.

When an LLM grades, monitors, or simulates, it is part of the apparatus rather
than the subject. That makes its behaviour a methods-section fact, and reviewers
ask methods-section questions about it: which model version, what exactly was in
the prompt, how often did it get called, and did its outputs drift over the run.

None of those can be answered afterwards from a summary metric. They can only be
answered from a log written at call time, which is what this wrapper does: one
JSONL line per call holding the model version, the prompt, the response, the
elapsed time, and a UTC timestamp.

The wrapper is deliberately a thin :class:`~..evaluation.protocols.LLMBackend`
itself, so it can be dropped in wherever a bare callable was expected and
nothing downstream needs to know it is being recorded.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..evaluation.protocols import LLMBackend
from ..utils.io import append_jsonl, count_jsonl
from ..utils.run_manifest import utc_now_iso, write_json_atomic

__all__ = ["LLM_LOG_NAME", "LLM_CONFIG_NAME", "LoggedLLM"]

logger = logging.getLogger(__name__)

LLM_LOG_NAME = "llm_log.jsonl"
LLM_CONFIG_NAME = "llm_config.json"


class LoggedLLM:
    """Wrap an ``LLMBackend`` so every call lands in ``run_dir/llm_log.jsonl``.

    Args:
        backend: Any ``str -> str`` callable: an API client, a local model
            closure, or a stub used to test the harness.
        run_dir: Where the log is written. Created if absent.
        model_version: The exact version string the provider reports, not a
            family name. ``gpt-4o`` is not a version; a dated snapshot is.
        max_prompt_chars: Truncate very long prompts in the log. Set to ``None``
            to record them whole, at the cost of a large file.
        extra: Constant fields added to every line, e.g. the temperature or the
            id of the prompt template in use.

    Raises:
        ValueError: If ``model_version`` is blank, since an unversioned log
            cannot answer the question it exists for.
    """

    def __init__(
        self,
        backend: LLMBackend,
        run_dir: str | Path,
        *,
        model_version: str,
        max_prompt_chars: int | None = 4_000,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if not model_version.strip():
            raise ValueError(
                "model_version must be a concrete version string such as "
                "'claude-sonnet-4-5-20250929'; a bare family name cannot be reproduced later"
            )
        self.backend = backend
        self.run_dir = Path(run_dir)
        self.model_version = model_version
        self.max_prompt_chars = max_prompt_chars
        self.extra = dict(extra or {})
        self.log_path = self.run_dir / LLM_LOG_NAME
        self._calls = 0
        self._elapsed = 0.0

    def _clip(self, text: str) -> str:
        if self.max_prompt_chars is None or len(text) <= self.max_prompt_chars:
            return text
        return f"{text[: self.max_prompt_chars]}... [truncated {len(text)} chars]"

    def __call__(self, prompt: str) -> str:
        """Call the backend, append the record, and return the response.

        The record is written even when the backend raises: a failed call is
        part of the call history, and a log that only contains successes
        understates both cost and flakiness.
        """
        started = time.perf_counter()
        response = ""
        error: str | None = None
        try:
            response = self.backend(prompt)
            return response
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = time.perf_counter() - started
            self._calls += 1
            self._elapsed += elapsed
            record: dict[str, Any] = {
                "model_version": self.model_version,
                "prompt": self._clip(prompt),
                "response": self._clip(response),
                "elapsed_seconds": round(elapsed, 4),
                "timestamp": utc_now_iso(),
                **self.extra,
            }
            if error is not None:
                record["error"] = error
            append_jsonl(self.log_path, record)

    @property
    def n_calls(self) -> int:
        """Calls made by this wrapper instance."""
        return self._calls

    @property
    def total_seconds(self) -> float:
        """Wall-clock time spent inside the backend."""
        return round(self._elapsed, 4)

    def log_config(self, settings: Mapping[str, Any] | None = None) -> Path:
        """Write the pinned settings this instrument ran under.

        Separate from the call log because a reader checking "was the judge
        pinned" should not have to read a hundred thousand call records to find
        out. Written atomically, and overwritten on each call so it always
        reflects the settings actually in force.
        """
        payload: dict[str, Any] = {
            "model_version": self.model_version,
            "backend": type(self.backend).__name__,
            "max_prompt_chars": self.max_prompt_chars,
            "logged_at": utc_now_iso(),
            **self.extra,
            **dict(settings or {}),
        }
        return write_json_atomic(self.run_dir / LLM_CONFIG_NAME, payload)

    def summary(self) -> dict[str, object]:
        """Call-count and timing summary, for the run manifest.

        Counts are read back from the file rather than from the in-memory
        counter so that a resumed run reports the whole history, not just the
        part this process contributed.
        """
        return {
            "model_version": self.model_version,
            "calls_this_process": self._calls,
            "calls_logged": count_jsonl(self.log_path),
            "total_seconds": self.total_seconds,
            "log_path": str(self.log_path),
        }
