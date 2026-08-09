"""Validate the measurement instrument on synthetic data with planted structure.

This is not a unit test. Unit tests ask whether a function returns what its
docstring says; this asks a different and more important question: *if the
effect we are looking for were present, would this harness find it?* An
instrument that cannot recover a planted effect will not be believed when it
reports a real one, and an instrument that recovers an effect from data
containing none is worse.

The planted world has three components. ``signal`` carries the target.
``context`` sharpens it slightly. ``noise`` contributes nothing at all, and its
recovered marginal value is the null the run is checked against. Removing
``signal`` is planted as a *silent* failure -- the prediction stays far from the
decision boundary while being wrong -- so the loud-versus-silent axis has a known
answer too.

Recovery is reported as a rate across seeds, not as a single pass or fail, and
every number produced here is stamped ``is_synthetic`` in the result payload.
Synthetic output is never a measured finding.

Usage:
    python scripts/validate_harness.py
    python scripts/validate_harness.py --seeds 20 --n-items 512 --min-recovery 0.95
    python scripts/validate_harness.py --out results/harness_validation.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

THRESHOLD = 0.5
TOLERANCE = 0.1
CONFIDENCE_MARGIN = 0.2
NULL_TOLERANCE = 0.02


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recovery-rate check of the failure harness on planted synthetic data.",
    )
    parser.add_argument(
        "--seeds", type=int, default=12,
        help="Number of seeds. Ten is the floor; fewer cannot express a rate.",
    )
    parser.add_argument("--n-items", type=int, default=256, help="Synthetic examples per seed.")
    parser.add_argument(
        "--min-recovery", type=float, default=0.9,
        help="Recovery rate below which this script exits non-zero.",
    )
    parser.add_argument(
        "--extra-components", type=int, default=1,
        help="Inert decoy components added in the generalization arm.",
    )
    parser.add_argument("--out", default=None, help="Where to write the result JSON.")
    parser.add_argument("--quiet", action="store_true", help="Only print the summary.")
    return parser


def _subset_rng(seed: int, present: frozenset[str]) -> np.random.Generator:
    """Deterministic per-subset generator, so a rerun at one seed is identical."""
    key = sum(ord(char) * (index + 1) for index, char in enumerate("".join(sorted(present))))
    return np.random.default_rng([seed, key])


def make_world(seed: int, n_items: int, extra: int = 0):
    """Build the planted world and a mask-aware prediction function for it.

    Returns:
        ``(components, target, gate_true, predict_fn)``.
    """
    rng = np.random.default_rng(seed)
    target = rng.uniform(0.0, 1.0, size=n_items)
    gate_true = target >= THRESHOLD
    components = ["signal", "context", "noise", *[f"decoy{i}" for i in range(extra)]]

    def predict_fn(present: frozenset[str]) -> np.ndarray:
        local = _subset_rng(seed, present)
        jitter = local.normal(0.0, 0.02, size=n_items)
        if "signal" in present:
            base = target.copy()
            if "context" not in present:
                base = base + local.normal(0.0, 0.05, size=n_items)
        else:
            # Planted silent failure: confidently wrong, far from the boundary.
            base = 1.0 - target
        return np.clip(base + jitter, 0.0, 1.0)

    return components, target, gate_true, predict_fn


def check_recovery(report) -> dict[str, bool]:
    """Compare what the harness recovered against what was planted."""
    marginal = report.complementarity["marginal_value"]
    best = max(marginal, key=lambda name: marginal[name])
    return {
        "signal_is_most_valuable": best == "signal",
        "signal_is_most_silent": report.most_silent_component == "signal",
        "noise_marginal_is_null": abs(float(marginal["noise"])) <= NULL_TOLERANCE,
        "context_beats_noise": float(marginal["context"]) > float(marginal["noise"]),
    }


def run_seed(seed: int, n_items: int, extra: int) -> dict:
    from evalaware.evaluation.failure import analyze_component_failure

    components, target, gate_true, predict_fn = make_world(seed, n_items, extra)
    report = analyze_component_failure(
        components,
        target,
        gate_true,
        predict_fn,
        threshold=THRESHOLD,
        tolerance=TOLERANCE,
        confidence_margin=CONFIDENCE_MARGIN,
    )
    checks = check_recovery(report)
    return {
        "seed": seed,
        "n": report.n,
        "components": components,
        "checks": checks,
        "all_recovered": all(checks.values()),
        "marginal_value": report.complementarity["marginal_value"],
        "most_silent_component": report.most_silent_component,
        "silent_rate_on_signal": report.dropout["drop_signal"]["silent_rate"],
    }


def summarize(records: list[dict]) -> dict[str, float]:
    """Per-check recovery rate across seeds."""
    names = sorted(records[0]["checks"])
    rates = {
        name: round(sum(record["checks"][name] for record in records) / len(records), 4)
        for name in names
    }
    rates["all_checks"] = round(
        sum(record["all_recovered"] for record in records) / len(records), 4
    )
    return rates


def main(argv: list[str] | None = None) -> int:
    """Run the recovery check across seeds and report the rate."""
    args = build_parser().parse_args(argv)
    if args.seeds < 10:
        print(f"--seeds must be at least 10 to express a recovery rate, got {args.seeds}")
        return 2

    # Imported here so --help works before `pip install -e .` and costs nothing.
    from evalaware.utils.reproducibility import set_seed
    from evalaware.utils.run_manifest import ResultPayload, save_results

    set_seed(0)

    base_records = [run_seed(seed, args.n_items, 0) for seed in range(args.seeds)]
    wide_records = [
        run_seed(seed, args.n_items, args.extra_components) for seed in range(args.seeds)
    ]

    if not args.quiet:
        print(f"three-component arm, {args.seeds} seeds, n={args.n_items}")
        for record in base_records:
            marks = "".join("." if ok else "X" for _, ok in sorted(record["checks"].items()))
            print(
                f"  seed {record['seed']:>3}  {marks}  "
                f"silent_rate(signal)={record['silent_rate_on_signal']}"
            )

    base_rates = summarize(base_records)
    wide_rates = summarize(wide_records)

    print("\nrecovery rate, three components:")
    for name, rate in base_rates.items():
        print(f"  {name:28s} {rate:.2%}")
    print(f"\nrecovery rate, {3 + args.extra_components} components (generalization arm):")
    for name, rate in wide_rates.items():
        print(f"  {name:28s} {rate:.2%}")

    payload = ResultPayload(
        task="E06_harness_validation",
        seed=0,
        n={"seeds": args.seeds, "items_per_seed": args.n_items},
        metrics={
            "three_component": base_rates,
            "generalization_arm": wide_rates,
            "per_seed": base_records,
        },
        is_synthetic=True,
        notes=[
            "Planted ground truth: 'signal' carries the target, 'noise' is inert, "
            "and removing 'signal' is planted as a silent failure.",
            "This validates the instrument. It is not a measurement of any model.",
        ],
    )
    out = Path(args.out) if args.out else REPO_ROOT / "results" / "harness_validation.json"
    save_results(out, payload)
    print(f"\nwrote {out} (marked synthetic)")

    worst = min(base_rates["all_checks"], wide_rates["all_checks"])
    if worst < args.min_recovery:
        print(
            f"\nFAIL: recovery {worst:.2%} is below the required {args.min_recovery:.2%}. "
            "The instrument cannot reliably find a planted effect, so it cannot be "
            "trusted to report a real one."
        )
        return 1
    print(f"\nPASS: recovery {worst:.2%} meets the {args.min_recovery:.2%} bar")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
