# Experiment plan — Evaluation Wording That Moves Safety Numbers

Stage-by-stage design. Each stage is registered in `src/evalaware/stages.py`
and appears in `python -m evalaware stages`.

## Stages

| stage | responsibility |
|---|---|
| `corpora` | evaluation-like vs deployment-like contrastive scenarios |
| `probe` | awareness probe fitting and held-out family validation |
| `verbalized` | detection of explicitly stated awareness in CoT |
| `inflation` | safety-score inflation as a function of probe score |
| `rewrite` | probe-supervised rewrite loop with construct-validity guard |

## Execution order

Stages form a linear dependency chain by default; the runner resolves the order
topologically, so a stage may be run alone and its prerequisites are pulled in
automatically:

```bash
python -m evalaware run -c configs/pilot.yaml --stage rewrite
```

## Controls and their purpose

- The probe can latch onto surface register, formatting or politeness, rather than awareness. Held-out scenario families and a style-matched control set are how that is tested.
- The rewrite loop can trivially lower the probe by gutting the eval. The construct-validity guard is what makes the loop meaningful, and it must be adversarially checked rather than assumed.

## Decision rules

Report effect sizes with bootstrap intervals. Treat an interval that spans zero as a null result and report it as such; do not reach for a subgroup that reaches significance.

## Reproducibility

Every run records a manifest with the git sha, a config fingerprint, resolved
device and dtype, package versions, per-stage timings, and metrics. Seeds are
set across python, numpy, and torch. Known determinism limits are recorded in
the manifest rather than assumed away: MPS does not support
`torch.use_deterministic_algorithms`, so small numeric drift between runs is
expected and should not be read as an effect.

## Scale

The pilot profile is what actually runs on the target machine. The full profile
describes the intended scaled-up run. When reporting any result, state which
profile produced it; a pilot-scale null is weaker evidence than a full-scale
null and the writeup must not blur them.
