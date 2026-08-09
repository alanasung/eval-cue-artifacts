"""Hydra entry point: compose a config, set up a run, execute its stages.

This is the only place in the repo where an experiment actually starts. Sweeps
and ablations do not reimplement the run loop; they call this script once per
cell so that every recorded number came through the same seeding, the same run
directory layout, and the same manifest.

Stages are resolved through a registry rather than a chain of ``if`` branches::

    # src/evalaware/stages.py
    STAGES = {"build_dataset": build_dataset, "collect": collect, ...}

Each stage is ``run(cfg, run_dir) -> dict``. A stage named in
``experiment.stages`` that is not registered is reported as unavailable in the
run manifest and on stdout, and nothing is computed for it -- a fresh clone can
therefore run this script before the pipeline exists, and it will say so out
loud rather than pretending it did the work.

Usage:
    python scripts/run_experiment.py experiment=pilot
    python scripts/run_experiment.py experiment=sweep_layers model=gpt2 run.seed=7
    python scripts/run_experiment.py --cfg job          # print the resolved config only
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

from evalaware.configs.loader import register_schema

REPO_ROOT = Path(__file__).resolve().parents[1]

# Registering before the decorator runs is what lets `base_schema` appear in the
# defaults list of configs/config.yaml.
register_schema()


def _load_stage_registry(package: str = "evalaware") -> dict:
    """Return the repo's stage registry, or an empty one if it has no stages yet."""
    import importlib

    try:
        module = importlib.import_module(f"{package}.stages")
    except ModuleNotFoundError:
        return {}
    return dict(getattr(module, "STAGES", {}))


def _run_stages(cfg: DictConfig, run_dir: Path) -> dict:
    registry = _load_stage_registry()
    requested = list(cfg.experiment.stages)
    outcomes: dict[str, dict] = {}
    unavailable: list[str] = []

    for name in requested:
        if name not in registry:
            unavailable.append(name)
            outcomes[name] = {"status": "unavailable", "elapsed_seconds": 0.0}
            continue
        print(f"[stage] {name} ...", flush=True)
        started = time.perf_counter()
        payload = registry[name](cfg, run_dir)
        elapsed = time.perf_counter() - started
        outcomes[name] = {
            "status": "ok",
            "elapsed_seconds": round(elapsed, 3),
            "output": payload,
        }
        print(f"[stage] {name} done in {elapsed:.2f}s", flush=True)

    if unavailable:
        print("")
        print("NOTE: nothing was computed for these stages; they are not registered")
        print(f"      in evalaware.stages: {unavailable}")
        print("      See docs/DESIGN.md, section 6, for the stage registry contract.")
        print("")
    return {"stages": outcomes, "unavailable": unavailable}


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run one experiment end to end and write its artifacts."""
    from evalaware.configs.loader import config_to_dict, save_resolved_config
    from evalaware.models.device import get_device
    from evalaware.utils.logging import configure_logging
    from evalaware.utils.reproducibility import set_seed
    from evalaware.utils.run_manifest import create_run_dir, save_run_metadata

    run_dir = create_run_dir(cfg.paths.runs, suffix=cfg.experiment.name)
    configure_logging(cfg.logging, run_dir=run_dir)

    determinism = set_seed(int(cfg.run.seed), deterministic=bool(cfg.run.deterministic))
    device = get_device(cfg)

    print(f"experiment : {cfg.experiment.name} ({cfg.experiment.task_id})")
    print(f"profile    : {cfg.run.profile}")
    print(f"hardware   : {device.describe()}")
    print(f"run dir    : {run_dir}")

    if cfg.run.save_resolved_config:
        save_resolved_config(cfg, run_dir / "config.yaml")

    started = time.perf_counter()
    outcome = _run_stages(cfg, run_dir)
    elapsed = time.perf_counter() - started

    save_run_metadata(
        run_dir,
        {
            "experiment": config_to_dict(cfg.experiment),
            "hardware": device.to_dict(),
            "determinism": determinism.to_dict(),
            "stages": outcome["stages"],
            "stages_unavailable": outcome["unavailable"],
            "elapsed_seconds": round(elapsed, 3),
        },
        seed=int(cfg.run.seed),
        profile=str(cfg.run.profile),
        script="scripts/run_experiment.py",
    )
    print(f"finished in {elapsed:.2f}s; manifest written to {run_dir / 'run_metadata.json'}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
