"""Report what is in the artifact cache and whether it can be trusted.

The frozen-backbone workflow depends on the cache being correct, and a cache is
exactly the kind of thing that stays plausible-looking while being wrong. Three
questions get asked here, in order of how much time the answer saves:

* Which namespaces exist, how many records each holds, and how much disk they
  are using -- the question behind "why is my laptop full".
* Which producer version each namespace is pinned to. Two namespaces at
  different versions is fine; one namespace holding records from two versions
  is a corrupted experiment.
* Do the manifest and the files on disk agree? A crash between writing the array
  and appending the manifest line leaves a record that exists but is unlisted,
  which resume logic will silently skip.

Usage:
    python scripts/inspect_cache.py
    python scripts/inspect_cache.py --root .cache/artifacts --verify
    python scripts/inspect_cache.py --namespace pilot --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and verify the artifact cache.")
    parser.add_argument("--root", default=None, help="Cache root. Defaults to paths.cache.")
    parser.add_argument("--namespace", default=None, help="Restrict to one namespace.")
    parser.add_argument(
        "--verify", action="store_true",
        help="Read every record back and report unreadable or unlisted entries.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Machine-readable output.")
    parser.add_argument(
        "-o", "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Hydra-style override used when reading paths.cache; repeatable.",
    )
    return parser


def _human(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def describe_namespace(root: Path, namespace: str, verify: bool) -> dict:
    from evalaware.cache.artifact_cache import ArtifactCache

    version_file = root / namespace / "version.txt"
    version = version_file.read_text(encoding="utf-8").strip() if version_file.is_file() else None
    cache = ArtifactCache(root, namespace, version=version or "unknown", strict_version=False)

    keys = cache.keys()
    records = cache.records()
    listed = {record.id for record in records}
    versions = sorted({record.version for record in records})

    info = {
        "namespace": namespace,
        "pinned_version": version,
        "record_versions": versions,
        "n_files": len(keys),
        "n_manifest_lines": len(records),
        "unlisted": sorted(set(keys) - listed),
        "listed_but_missing": sorted(listed - set(keys)),
        "disk_bytes": cache.disk_usage_bytes(),
        "mixed_versions": len(versions) > 1,
    }
    if verify:
        unreadable = []
        for key in keys:
            try:
                cache.read(key)
            except (KeyError, ValueError, OSError) as exc:
                unreadable.append({"id": key, "error": f"{type(exc).__name__}: {exc}"})
        info["unreadable"] = unreadable
    return info


def main(argv: list[str] | None = None) -> int:
    """Summarize every cache namespace and flag inconsistencies."""
    args = build_parser().parse_args(argv)

    # Imported here so --help works before `pip install -e .` and costs nothing.
    from evalaware.configs.loader import load_config

    if args.root:
        root = Path(args.root)
    else:
        cfg = load_config(overrides=args.override)
        root = REPO_ROOT / str(cfg.paths.cache)

    if not root.is_dir():
        print(f"no cache at {root}; nothing has been cached yet")
        return 0

    namespaces = (
        [args.namespace]
        if args.namespace
        else sorted(path.name for path in root.iterdir() if path.is_dir())
    )
    infos = [describe_namespace(root, name, args.verify) for name in namespaces]

    if args.as_json:
        print(json.dumps({"root": str(root), "namespaces": infos}, indent=2))
    else:
        print(f"cache root: {root}")
        print(f"{'namespace':22s} {'version':10s} {'records':>8s} {'manifest':>9s} {'size':>10s}")
        for info in infos:
            print(
                f"{info['namespace']:22s} {str(info['pinned_version']):10s} "
                f"{info['n_files']:>8d} {info['n_manifest_lines']:>9d} "
                f"{_human(info['disk_bytes']):>10s}"
            )
        for info in infos:
            for label, key in (
                ("unlisted in manifest", "unlisted"),
                ("listed but missing", "listed_but_missing"),
                ("unreadable", "unreadable"),
            ):
                items = info.get(key) or []
                if items:
                    print(f"\n{info['namespace']}: {len(items)} {label}")
                    for item in items[:10]:
                        print(f"  {item}")
            if info["mixed_versions"]:
                print(
                    f"\n{info['namespace']}: manifest holds several producer versions "
                    f"{info['record_versions']}; results built from it mix incomparable "
                    "artifacts and should be discarded"
                )

    problems = sum(
        len(info.get("unlisted") or [])
        + len(info.get("listed_but_missing") or [])
        + len(info.get("unreadable") or [])
        + int(info["mixed_versions"])
        for info in infos
    )
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
