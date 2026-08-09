# Build instructions

Context for coding agents working in this repository.

## What this is

Research implementation of "Catching Silent Test Recognition Before Scores Look Safe".

## Hard constraints

- **Hardware.** Apple M4, 10 cores, unified memory, PyTorch MPS. No CUDA. No API
  keys are configured. The pilot profile must actually complete locally.
- **Model size.** Stay at or below roughly 1.5B parameters in the pilot path.
- **No fabricated results.** Never write example numbers into docs or tests as if
  they were measured. If a stage has not been run, say so.
- **No silent API dependencies.** Anything needing a key goes behind a config
  flag and raises `MissingCredentialError` naming the variable.

## Where to work

Implement the stages in `src/evalaware/stages.py`. The signatures,
dependency edges, and docstring contracts are fixed; they encode an experiment
design that was checked against the motivating posting before implementation
started. Do not renegotiate them. If a contract is genuinely wrong, change it in
`docs/EXPERIMENT_PLAN.md` first and explain why.

The shared infrastructure in `config.py`, `device.py`, `seeding.py`,
`manifest.py`, `runner.py`, `modelio.py`, `hooks.py`, `metrics.py`, and
`plots.py` is complete and tested. Use it rather than reimplementing it. In
particular use `hooks.capture` / `hooks.steer` / `hooks.ablate` for anything
touching activations, and `metrics.bootstrap_*` for anything reported.

## Style

- Match the surrounding code. Type hints throughout, dataclasses for structured
  state, `logging` rather than `print` inside library code.
- Comments explain constraints and non-obvious choices, not what the next line
  does.
- Every new stage needs a test. Tests must not download model weights; use the
  tiny fixtures in `tests/conftest.py`.

## Checks

```bash
make test     # pytest
make lint     # ruff
make pilot    # the real end-to-end run
```
