# Design Document: Catching Silent Test Recognition Before Scores Look Safe

## 1. Purpose

This document is the method contract for **Catching Silent Test Recognition Before Scores Look Safe**. It specifies what is
measured, how measurement is instrumented, which artifacts are produced, and
which claims are out of scope for the pilot profile.

One-liner: Train a probe for 'this is a test', then let it referee an automated rewrite loop that makes evaluations harder to recognize.

Hypothesis: Models can recognize evaluation conditions without saying so; a probe on internals can expose that gap and supervise scenario edits that reduce it while keeping the underlying task intact.

## 2. Scope

| In scope (pilot) | Out of scope (pilot) |
|---|---|
| Shared spine: config, cache, hooks, metrics, ablations, reporting | Production serving |
| Synthetic + small public models (GPT-2 / Qwen 0.5B) | 70B-class models |
| Causal interventions via hooks | Full mechanistic reverse-engineering |
| Harness validation on planted structure | Unlabelled web scrape corpora |

## 3. System overview

| Component | Package | Responsibility |
|---|---|---|
| Config schema | `evalaware.configs.schema` | Typed dataclasses + role-keyed models |
| Loader | `evalaware.configs.loader` | Hydra compose API |
| Device | `evalaware.models.device` | MPS/CUDA/CPU resolution + fallback flag |
| Registry | `evalaware.models.registry` | Pinned revisions + hardware fit |
| Hooks | `evalaware.models.hooks` | Capture, steer, ablate, patch, attn intervene |
| Generation | `evalaware.models.generation` | Chat template path recording |
| Data | `evalaware.data` | Synthetic items, splits, manifests |
| Evaluation | `evalaware.evaluation` | Metrics, failure harness, protocols |
| Ablation | `evalaware.ablation` | Control / dropout / seed-sweep arms |
| Reporting | `evalaware.reporting` | Aggregate → MD/TeX/figures |
| Cache | `evalaware.cache` | Versioned artifact store |

## 4. Configuration model

### 4.1 Groups

| Group | Examples | Role |
|---|---|---|
| `model` | `gpt2`, `qwen0.5b` | Checkpoint + dtype + device |
| `data` | `pilot`, `synthetic` | n_items, splits |
| `eval` | `default`, `fast` | thresholds, layers, bootstrap |
| `paths` | `default`, `scratch` | write roots |
| `experiment` | `pilot`, `baseline`, `ablation_*` | stages + task id |
| `cache` | `default`, `disabled` | artifact cache |
| `sweep` | `seeds`, `layers` | cross-product grids |
| `ablation` | `controls`, `component_dropout` | arm selection |
| `reporting` | `default`, `paper` | figure/table options |
| `logging` | `default`, `quiet` | log level |
| `run` | `pilot`, `full` | profile, seed, memory |

### 4.2 Role-keyed models (X6)

`Config.model` remains the primary model. `Config.roles` maps role names
(`reference`, `explainer`, `simulator`, `monitor`, …) to additional
`ModelConfig` instances so multi-model designs do not overload a single field.

### 4.3 Layer defaults (X11 / X8)

Default `eval.layers = [2, 4, 6]`, valid for GPT-2 (12 layers, indices 0..11).
`validate_layers(layers, n_layers)` raises with the valid range if a config
asks for an out-of-range index (e.g. 12 on GPT-2).

## 5. Model loading and pinning (X9)

| Field | Source | Recorded where |
|---|---|---|
| `ModelSpec.revision` | `models/registry.py` | registry + config YAML |
| Requested revision | `ModelConfig.revision` | load call |
| Resolved commit | `config._commit_hash` when present | `LoadedModel.revision` + notes |

Remote weights must not drift under a fixed seed. Prefer commit SHAs; `main` is
an explicit string field when a stable SHA is not yet recorded.

## 6. Device and dtype policy (X4, X13)

| Policy | Behaviour |
|---|---|
| MPS fallback | `PYTORCH_ENABLE_MPS_FALLBACK=1` via `enable_mps_fallback` |
| Manifest | `mps_fallback_enabled` in `DeviceInfo.to_dict` |
| bfloat16 on MPS | Downgrade to float16 + warning |
| CUDA missing | `HardwareError`, no silent CPU fallback |
| Non-torch work | CPU only (numpy/sklearn/pandas/matplotlib) |

## 7. Generation interface (X5)

Instruction-tuned models must see their chat template. `generate(...,
use_chat_template=True)` calls `tokenizer.apply_chat_template` when available
and returns metadata `chat_template_path ∈ {chat_template, chat_template_unavailable, raw}`.

## 8. Hooks and interventions (X7)

| API | What it does |
|---|---|
| `capture` | Residual-stream activations |
| `steer` | Add coefficient × unit direction (position-optional) |
| `ablate` | Directional projection removal (position-optional) |
| `patch_activations` | Paired activation patching from a source tensor |
| `intervene_attention_heads` | Per-head attention pattern replacement |

Hooks always remove themselves in `finally`. Surviving hooks are treated as
critical bugs.

## 9. Metrics and null claims (X12)

An interval that spans zero is **inconclusive**, not evidence of a null.

| Tool | Use |
|---|---|
| `Estimate` | value + CI + n |
| `minimum_detectable_effect` | reported when CI spans zero |
| `tost_equivalence` | required before claiming equivalence / null |

## 10. Data and power (X11)

| Profile | `n_items` | Rationale |
|---|---:|---|
| pilot | ≥ 512 | After 60/20/20, test ≈ 102; bootstrap meaningful |
| synthetic smoke | small | harness wiring only |
| full | project-specific | documented per experiment |

## 11. Failure harness

The model-agnostic failure analysis reports:

1. Taxonomy under full and missing-component conditions
2. Leave-one-out marginal value / complementarity
3. Loud vs silent failure rates under `confidence_margin`

`PredictFn = Callable[[frozenset[str]], np.ndarray]` keeps evaluation free of
concrete model imports.

## 12. Ablation surface

| Module | Arms |
|---|---|
| `ablation.controls` | identity, label-shuffle, score-shuffle, random |
| `ablation.component_dropout` | full + leave-one-out |
| `ablation.seed_sweep` | cross-seed cells with summary mean/std |

Each returns a structured dict with `task`, `seed`, `git_sha`, `elapsed_seconds`.

## 13. Artifact cache

Version-pinned store with atomic writes, append-only `manifest.jsonl`, `has`
resume, safe-name validation, and hard errors on version mismatch.

## 14. Reporting contract

| Artifact | Path |
|---|---|
| Aggregate JSON | `results/results.json` |
| Markdown table | `results/tables/results.md` |
| LaTeX table | `results/tables/results.tex` |
| Figures | `results/figures/*.{pdf,svg,png}` |
| Captions | `results/figures/*.caption.txt` |

## 15. Experiment presets

| Preset | Task id | Intent |
|---|---|---|
| `smoke` | E00 | Compose + seed + run dir |
| `pilot` | E01 | Local end-to-end |
| `baseline` | E02 | Primary baseline |
| `full` | E03 | Scaled profile |
| `harness_validation` | E04 | Planted recovery |
| `robustness` | E05 | Sensitivity |
| `sweep_layers` | S01 | Layer sweep |
| `sweep_scale` | S02 | Scale sweep |
| `ablation_controls` | A01 | Control arms |
| `ablation_seeds` | A02 | Seed sweep |

## 16. Open technical decisions

- [ ] Domain package stage implementations behind the registry
- [ ] Whether to pin every Hub model to a full commit SHA (partially done: GPT-2)
- [ ] Equivalence band defaults per primary metric
- [ ] Whether multimodal loaders join the shared spine or stay domain-local
- [ ] Logging LLM calls for any API-backed judge
- [ ] Exact falsification thresholds per project hypothesis

## 17. Expected artifacts

| Directory | Contents |
|---|---|
| `runs/<timestamp>/` | `config.yaml`, `run_metadata.json`, stage outputs |
| `results/` | aggregate JSON |
| `results/tables/` | `results.md`, `results.tex` |
| `results/figures/` | PDF + SVG + 300 dpi PNG + captions |
| `.cache/artifacts/` | versioned activation / embedding cache |
| `data/manifests/` | tiny dataset provenance JSON |

## 18. Testing strategy

| Suite | Covers |
|---|---|
| Config composition | every experiment preset |
| Artifact cache | round-trip, resume, version mismatch |
| Metrics | hand-computed values, MDE, TOST |
| Failure harness | planted structure recovery |
| Hooks | capture/steer/ablate/patch/attn |
| Generation | chat template path |
| Device | MPS fallback flag |
| Validation | invalid layer indices |
| Roles | role-keyed `Config.roles` |
| SDK | `tests/test_sdk.py` API contract |

## 19. Threats to validity

| Threat | Mitigation |
|---|---|
| Weight drift | pinned revision + resolved commit |
| Wrong chat interface | template path recording |
| Underpowered null | MDE + TOST |
| Hook leakage | contextmanager `finally` removals |
| Silent CPU fallback | `HardwareError` on missing accelerator |
| Synthetic reported as measured | `is_synthetic` + CONTRIBUTING honesty rule |

## 21. Glossary

| Term | Meaning |
|---|---|
| Pilot | Local profile that must finish on M4-class hardware |
| Full | Scaled profile; may require discrete GPU |
| MDE | Minimum detectable effect |
| TOST | Two one-sided tests for equivalence |
| Role | Named model slot in `Config.roles` |

## 22. Change control

Shared spine changes land in `orchestration/templates/` and are regenerated into
all seven repos by `scaffold2.py`. Domain packages under
`src/evalaware/evalaware/` are preserved across regeneration.

## Appendix: regeneration

Run `python orchestration/scaffold2.py` from the meta-repo root after template edits.

## Appendix: regeneration

Run `python orchestration/scaffold2.py` from the meta-repo root after template edits.

## Appendix: regeneration

Run `python orchestration/scaffold2.py` from the meta-repo root after template edits.

## Appendix: regeneration

Run `python orchestration/scaffold2.py` from the meta-repo root after template edits.

## Appendix: regeneration

Run `python orchestration/scaffold2.py` from the meta-repo root after template edits.
