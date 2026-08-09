# Repository guide

See [CLAUDE.md](CLAUDE.md) for the full build instructions; this file exists so
that agents which look for `AGENTS.md` find the same context.

Project: Catching Silent Test Recognition Before Scores Look Safe
Package: `src/evalaware`
Entry point: `python -m evalaware --help`

Key rules:
- The pilot profile must run on an Apple M4 with no CUDA and no API keys.
- Do not invent measured numbers.
- Implement `stages.py`; the shared infrastructure is finished.
- Run `make test lint` before considering a change done.
