"""Cross-cutting utilities: provenance, seeding, persistence, logging, validation.

Nothing here knows about models, datasets, or the research question. That is the
constraint that lets every other subpackage depend on this one without creating
a cycle, and it is why the import graph stays a tree.

The three things a reportable number needs -- a seed, a commit, and a resolved
config -- are owned by :mod:`.reproducibility`, :mod:`.git`, and
:mod:`.run_manifest` respectively.
"""

from __future__ import annotations

from .git import GitState, git_is_dirty, git_sha, git_state
from .io import (
    append_jsonl,
    count_jsonl,
    ensure_dir,
    iter_jsonl,
    load_json,
    read_jsonl,
    save_json,
    write_jsonl,
)
from .logging import LOG_FILE_NAME, configure_logging, get_logger, set_level
from .reproducibility import (
    DeterminismReport,
    determinism_report,
    seed_everything,
    set_seed,
    worker_seed,
)
from .run_manifest import (
    ResultPayload,
    RunMetadata,
    create_run_dir,
    save_results,
    save_run_metadata,
    utc_now_iso,
    write_json_atomic,
)
from .validation import (
    require_in_range,
    require_non_empty,
    require_positive,
    require_same_length,
    require_subset,
    validate_layers,
)

__all__ = [
    "LOG_FILE_NAME",
    "DeterminismReport",
    "GitState",
    "ResultPayload",
    "RunMetadata",
    "append_jsonl",
    "configure_logging",
    "count_jsonl",
    "create_run_dir",
    "determinism_report",
    "ensure_dir",
    "get_logger",
    "git_is_dirty",
    "git_sha",
    "git_state",
    "iter_jsonl",
    "load_json",
    "read_jsonl",
    "require_in_range",
    "require_non_empty",
    "require_positive",
    "require_same_length",
    "require_subset",
    "save_json",
    "save_results",
    "save_run_metadata",
    "seed_everything",
    "set_level",
    "set_seed",
    "utc_now_iso",
    "validate_layers",
    "worker_seed",
    "write_json_atomic",
    "write_jsonl",
]
