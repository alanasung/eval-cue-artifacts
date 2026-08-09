"""Print what this machine will actually do with the current config.

"It works on my machine" is usually a device, dtype, or memory difference, and
all three are invisible until a run behaves strangely. This script resolves them
up front and prints them, so the answer to "what did that run use" exists before
the run rather than being reconstructed afterwards.

It is also the first thing to paste into an issue. Everything below is
non-destructive: nothing is downloaded, nothing is written, nothing is cached.

Usage:
    python scripts/doctor.py
    python scripts/doctor.py --preset full
    python scripts/doctor.py --json
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

WATCHED_PACKAGES = (
    "torch", "transformers", "numpy", "pandas", "scikit-learn",
    "hydra-core", "omegaconf", "matplotlib",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report resolved device, dtype, and environment.")
    parser.add_argument("--preset", default=None, help="Resolve against this experiment preset.")
    parser.add_argument(
        "-o", "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Hydra-style override; repeatable.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Machine-readable output.")
    return parser


def package_versions() -> dict[str, str]:
    versions = {}
    for name in WATCHED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def torch_report() -> dict[str, object]:
    """Backend availability straight from torch, before any config is applied."""
    try:
        import torch
    except ImportError:
        return {"available": False}
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_built": bool(getattr(torch.version, "cuda", None)),
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "default_dtype": str(torch.get_default_dtype()).replace("torch.", ""),
        "num_threads": torch.get_num_threads(),
    }


def disk_report(paths: list[Path]) -> dict[str, object]:
    usage = shutil.disk_usage(str(REPO_ROOT))
    return {
        "repo_free_gb": round(usage.free / 1024**3, 1),
        "repo_total_gb": round(usage.total / 1024**3, 1),
        "existing_dirs": {str(path): path.is_dir() for path in paths},
    }


def main(argv: list[str] | None = None) -> int:
    """Resolve and print the environment this repo will run in."""
    args = build_parser().parse_args(argv)

    # Imported here so --help works before `pip install -e .` and costs nothing.
    from evalaware.configs.loader import available_presets, load_config
    from evalaware.models.device import get_device
    from evalaware.utils.git import git_state
    from evalaware.utils.reproducibility import determinism_report

    overrides = list(args.override)
    if args.preset:
        overrides.insert(0, f"experiment={args.preset}")
    cfg = load_config(overrides=overrides)
    device = get_device(cfg)
    state = git_state()

    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "torch": torch_report(),
        "resolved": device.to_dict(),
        "determinism": determinism_report(int(cfg.run.seed)).to_dict(),
        "packages": package_versions(),
        "git": state.to_dict(),
        "config": {
            "preset": args.preset or "(base)",
            "experiment": str(cfg.experiment.name),
            "task_id": str(cfg.experiment.task_id),
            "profile": str(cfg.run.profile),
            "model": str(cfg.model.name),
            "requested_device": str(cfg.model.device),
            "requested_dtype": str(cfg.model.dtype),
            "memory_budget_gb": float(cfg.run.max_memory_gb),
            "available_presets": available_presets(),
        },
        "disk": disk_report(
            [REPO_ROOT / str(cfg.paths.data), REPO_ROOT / str(cfg.paths.cache),
             REPO_ROOT / str(cfg.paths.runs), REPO_ROOT / str(cfg.paths.results)]
        ),
    }

    if args.as_json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    print("== interpreter ==")
    print(f"  python     {report['python']}  ({report['executable']})")
    print(f"  platform   {report['platform']}")
    print("\n== torch ==")
    for key, value in report["torch"].items():
        print(f"  {key:16s} {value}")
    print("\n== resolved for this config ==")
    print(f"  device     {device.device}")
    print(f"  dtype      {device.dtype}")
    print(f"  memory     {device.available_memory_gb:.1f} GiB free "
          f"of {device.total_memory_gb:.1f} GiB"
          f"{' unified' if device.unified_memory else ''}")
    print(f"  budget     {cfg.run.max_memory_gb} GiB (run.max_memory_gb)")
    for note in device.notes:
        print(f"  note       {note}")
    for caveat in report["determinism"]["caveats"]:
        print(f"  caveat     {caveat}")
    print("\n== packages ==")
    for name, version in report["packages"].items():
        print(f"  {name:16s} {version}")
    print("\n== repository ==")
    print(f"  git        {state.sha[:12]} on {state.branch} (dirty={state.dirty})")
    print(f"  presets    {', '.join(report['config']['available_presets'])}")
    print(f"  disk free  {report['disk']['repo_free_gb']} GiB")

    if not report["torch"].get("available"):
        print("\ntorch is not installed; run `make install` before any experiment")
        return 1
    if device.device == "cpu":
        print(
            "\nNo accelerator was resolved. The pilot path will still run but will be "
            "one to two orders of magnitude slower than on MPS."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
