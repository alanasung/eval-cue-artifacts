"""Materialize the item set named by the ``data`` config group.

Datasets are built by a registered builder rather than by a branch in this
script, so adding a source means registering a function, not editing a CLI::

    # src/evalaware/data/__init__.py
    BUILDERS = {"pilot": build_pilot, "full": build_full}

The ``synthetic`` builder lives here rather than in the registry because it must
work on a machine with no network, no credentials, and no registry: it is what
makes `make smoke` runnable on a fresh clone. Its items carry planted structure
and every downstream payload built from them is marked synthetic.

Splits are written into the items themselves rather than into a side file. A
split assignment that lives somewhere else is a split assignment that eventually
disagrees with the data.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --preset pilot --force
    python scripts/build_dataset.py -o data.n_items=128 -o data.seed=3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

TEMPLATE_FAMILIES = ("direct", "framed", "multi_turn")
DIFFICULTIES = ("easy", "medium", "hard")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the configured item set.")
    parser.add_argument("--preset", default=None, help="Experiment preset to read data config from.")
    parser.add_argument(
        "-o", "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Hydra-style override; repeatable.",
    )
    parser.add_argument("--out", default=None, help="Output directory. Defaults to <data>/<name>.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing item file.")
    parser.add_argument("--limit", type=int, default=None, help="Preview only the first N items.")
    return parser


def assign_splits(n: int, fracs: tuple[float, float, float], seed: int, shuffle: bool) -> list[str]:
    """Deterministic split labels. Rounding leftovers go to train, never to test."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n) if shuffle else np.arange(n)
    n_train = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    labels = ["test"] * n
    for position, index in enumerate(order):
        if position < n_train:
            labels[int(index)] = "train"
        elif position < n_train + n_val:
            labels[int(index)] = "val"
    return labels


def build_synthetic(n_items: int, seed: int, splits: list[str], max_prompt_tokens: int) -> list[dict]:
    """Items with planted structure, for plumbing and instrument checks only.

    ``target`` is the continuous quantity, ``gate`` the binary decision derived
    from it, and ``latent_signal`` the planted cause. Difficulty is planted as
    distance from the decision boundary, so a harness that recovers "hard items
    are harder" is recovering something that is genuinely there.
    """
    rng = np.random.default_rng(seed)
    items = []
    for index in range(n_items):
        target = float(rng.uniform(0.0, 1.0))
        margin = abs(target - 0.5)
        difficulty = DIFFICULTIES[0] if margin > 0.33 else (
            DIFFICULTIES[1] if margin > 0.15 else DIFFICULTIES[2]
        )
        family = TEMPLATE_FAMILIES[index % len(TEMPLATE_FAMILIES)]
        items.append(
            {
                "id": f"syn-{index:05d}",
                "split": splits[index],
                "prompt": (
                    f"[{family}] Synthetic item {index}. "
                    f"Respond with a value in [0, 1]."
                )[: max_prompt_tokens * 4],
                "target": round(target, 6),
                "gate": bool(target >= 0.5),
                "latent_signal": round(float(rng.normal(target, 0.05)), 6),
                "difficulty": difficulty,
                "template_family": family,
                "item_source": "synthetic",
                "is_synthetic": True,
            }
        )
    return items


def _registered_builder(name: str):
    """Look up a project-specific builder, returning None when there is none."""
    import importlib

    try:
        module = importlib.import_module("evalaware.data")
    except ModuleNotFoundError:
        return None
    return dict(getattr(module, "BUILDERS", {})).get(name)


def main(argv: list[str] | None = None) -> int:
    """Build the configured dataset and write items plus a manifest."""
    args = build_parser().parse_args(argv)

    # Imported here so --help works before `pip install -e .` and costs nothing.
    from evalaware.configs.loader import load_config
    from evalaware.utils.run_manifest import utc_now_iso, write_json_atomic

    overrides = list(args.override)
    if args.preset:
        overrides.insert(0, f"experiment={args.preset}")
    cfg = load_config(overrides=overrides)
    data = cfg.data

    out_dir = Path(args.out) if args.out else REPO_ROOT / str(cfg.paths.data) / str(data.name)
    items_path = out_dir / "items.jsonl"
    if items_path.exists() and not args.force:
        print(f"{items_path} already exists; pass --force to rebuild")
        return 2

    splits = assign_splits(
        int(data.n_items),
        (float(data.train_frac), float(data.val_frac), float(data.test_frac)),
        int(data.seed),
        bool(data.shuffle),
    )

    builder = _registered_builder(str(data.name))
    if builder is not None:
        items = list(builder(cfg, splits))
        source = f"evalaware.data.BUILDERS[{data.name!r}]"
    elif str(data.name) == "synthetic":
        items = build_synthetic(
            int(data.n_items), int(data.seed), splits, int(data.max_prompt_tokens)
        )
        source = "scripts/build_dataset.py:build_synthetic"
    else:
        print(
            f"no builder registered for data.name={data.name!r}. Register one in "
            "src/evalaware/data/__init__.py under BUILDERS, or use data=synthetic "
            "for a plumbing check."
        )
        return 2

    if args.limit:
        for item in items[: args.limit]:
            print(json.dumps(item))
        print(f"\npreview only; {len(items)} items would be written to {items_path}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    with items_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")

    counts = {name: splits.count(name) for name in ("train", "val", "test")}
    is_synthetic = all(item.get("is_synthetic") for item in items)
    write_json_atomic(
        out_dir / "manifest.json",
        {
            "task": "D01_build_dataset",
            "name": str(data.name),
            "builder": source,
            "created_at": utc_now_iso(),
            "seed": int(data.seed),
            "n_items": len(items),
            "splits": counts,
            "fields": sorted(items[0]) if items else [],
            "is_synthetic": is_synthetic,
            "max_prompt_tokens": int(data.max_prompt_tokens),
        },
    )
    print(f"built {len(items)} items via {source}")
    print(f"splits: {counts}")
    if is_synthetic:
        print("these items are SYNTHETIC; nothing derived from them is a measured result")
    print(f"wrote {items_path} and {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
