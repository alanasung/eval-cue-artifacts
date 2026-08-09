# Contributing

## Setup

```bash
make install-dev
cp .env.example .env   # add HF_TOKEN if using gated models
make doctor
make test
```

Python **3.12** is the supported interpreter (`requires-python = ">=3.10,<3.13"`).
The Makefile discovers `python3.12` through a candidate list; it does not
hardcode a single absolute path.

## Command table

| Command | What it does |
|---|---|
| `make install` | Runtime deps from `requirements.txt` |
| `make install-dev` | Runtime + `dev`/`test` extras + pre-commit |
| `make lint` | `ruff check` |
| `make format` | `ruff format` + fix |
| `make test` | pytest |
| `make test-cov` | pytest with `--cov-fail-under=60` |
| `make typecheck` | mypy on `src/evalaware` |
| `make ci` | local mirror of GitHub Actions |
| `make pilot` | pilot experiment profile |
| `make smoke` | config smoke test |
| `make doctor` | device / env diagnostics |
| `make clean` | caches, runs, results |

## Data access

- Do not commit raw data, embeddings, or checkpoints.
- Checked-in files under `data/` are manifests (< 2 KB) and this README.
- `.gitignore` excludes `/data/**` except `data/README.md` and the manifests
  directory placeholder; pre-commit also blocks large adds.

## Code conventions

- Library code: typed, print-free, importable. Scripts own argparse/Hydra.
- Prefer injected callables (`PredictFn`) over importing concrete model classes
  into evaluation.
- Raise `ValueError` with the offending value and the fix at every boundary.
- No `TODO`/`FIXME`/`HACK` in first-party code — track work in issues / `TASK.md`.

## Reproducibility / honesty

When you report a number, state whether it came from a run you actually executed
and on what data. Never present an estimate, a partial run, or a synthetic-demo
output as a measured result. Synthetic harness outputs must carry
`is_synthetic: true` and must be labelled as such in prose.

To regenerate reported numbers a reader needs:

1. The git SHA stamped in the result JSON
2. The resolved Hydra `config.yaml` beside the run
3. The pinned `requirements.lock`
4. The dataset manifest version

## Pull requests

- Branch names: `feat/*` or `feature/*`
- Reference task IDs from `TASK.md` in the PR body (`E03`, `A01`, …)
- CI must be green: lint, test, api-contract, typecheck, coverage
