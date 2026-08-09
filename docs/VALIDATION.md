# VALIDATION — eval-awareness-hardening

## Codex (p4)
- Verdict: SERIOUS_PROBLEMS
- Summary: Codex wants online generation-time intervention, live CoT/inflation scoring, rewrite recollect on loaded weights, and full independent monitors — standards partly beyond the local measurable pilot. Corpora family/context pairing, measured residual collect, construct status fields, and rewrite gating are in place.
- Detail: `orchestration/out/validate/eval-awareness-hardening.json`

## Grok (p4 dual)
- Verdict: PASS_WITH_NOTES
- Summary: Measured residual collect is pilot-default and fail-closed; construct requires margin + causal + label-ablation + scramble with status fields; rewrite skips when construct fails; `heldout_linear` is a real held-out monitor; stand-ins labeled; `n_items: 512`; pinned Qwen/Pythia/Llama revisions; domain P4 tests pass under Hub monkeypatch.
- Detail: `orchestration/out/grok/validate/eval-awareness-hardening.p4.md`

## KEY_FIXES (p4)
| Fix | Status |
|---|---|
| Real activation features from loaded small model | OK (`features.collect_activations`, `model_runtime.py`) |
| `force_synthetic` smoke-only; pilot measured default | OK (pilot `false`, smoke `true`, `_force_synthetic`) |
| Fail-closed measured collect | OK (`RuntimeError` when weights missing) |
| Causal / scramble / label-ablation + construct status | OK (`construct.py` `construct_status` / `claim`) |
| Rewrite only when construct held | OK (`rewrite.py` `skipped_construct_failed`) |
| Real non-probe monitor + labeled stand-ins | OK (`heldout_linear` in `real_monitors`) |
| Balanced family×context pairing | OK (`corpora.py` consecutive eval/deploy pairs) |
| Honest item counts (512) | OK (`pilot.yaml`) |
| Pinned revisions | OK (qwen/pythia/llama yaml + registry SHAs) |
| Hub-free domain tests | OK (`test_domain_p4_measured.py`, 47 domain tests) |

## Remaining (compute / scale — not empty stages)
- Rewrite transfer rates use frozen-probe proxy features after text edits (`scoring_mode=proxy_after_rewrite`), not full residual recollect per candidate.
- Causal moves probe scores on stored activations; not live generation-time steering.
- Verbalized CoT and safety inflation remain simulated procedures with fixed rubrics.
- SAE / lens / attention stay labeled stand-ins; diffusion/latent arm scoped out for M4 memory.
- Codex SERIOUS_PROBLEMS accepted as frontier-purity residual notes under the local-pilot lens.

## Reconciliation
Grok PASS_WITH_NOTES on the measurable core. Codex SERIOUS_PROBLEMS remains on live generation / full monitor family / rewrite recollect — recorded as residual scale notes, not missing stages. Domain tests pass (47).

## P5 rigor pass (measured prior work-critical paths)

- Live / measured paths preferred; synthetic remains smoke-only with honesty stamps.
- Claim gating tightened where proxies previously looked like evidence.
- Domain tests green without Hub downloads.

## P6 rewrite recollect + construct behavioral + chat template

| Fix | Status |
|---|---|
| Rewrite recollects measured residuals when not smoke (`scoring_mode=recollect_after_rewrite`) | OK (M4 cap default 64) |
| Smoke keeps `proxy_after_rewrite` + `measured_claims_ok=False` | OK |
| Fail-closed when measured recollect required and weights missing | OK |
| Construct `algebraic_ok` + `behavioral_ok`; `held` requires both | OK (held-out monitor agreement under ablation) |
| Activation collect uses chat template when available; stamps `used_chat_template` / `template_path` | OK |
| Stand-in monitors excluded from transfer headline by default | OK |
| Hub-free domain P6 tests | OK (`test_domain_p6_rewrite_construct.py`) |

Residual: generation-time residual steer / live CoT scoring beyond simulated rubrics; SAE/lens/attention remain stand-ins.

## P7 rigor pass (stats + claim contracts)

| Fix | Status |
|---|---|
| Shared calibrated local refusal scorer for inflation (`behavior_labeler_id` + fixture agreement) | OK (`behavior_labeler.py`, `inflation.py`) |
| Unverbalized-fraction headline requires measured CoT + construct held | OK (`verbalized.py` `headline_licensed`) |
| CI spanning 0 → TOST before “no silent awareness” | OK (`tost_unverbalized`, `no_silent_awareness_ok`) |
| Rewrite recollect power: `recollect_n`, `mde_transfer`; claims gated below floor | OK (`rewrite.py`, floor default 32) |
| Headline MDE/TOST wired for unverbalized / inflation slope / transfer | OK |
| Hub-free domain P7 tests | OK (`test_domain_p7_stats_claims.py`) |

Residual: full residual recollect on every rewrite candidate (cap remains); SAE/lens/attention stand-ins; generation-time steer.

## Codex problem-statement fit

Model: `gpt-5.6-sol` · gates: `match` + `validate` · 2026-08-09
Artifacts: `orchestration/out/match/eval-awareness-hardening.md`, `orchestration/out/validate/eval-awareness-hardening.md`

- **Match verdict:** `MATERIAL_DRIFT` · methods `mixed_proxy`
- **Validate overall:** `SERIOUS_PROBLEMS` · fit `MINOR_DRIFT` · feasibility `RUNNABLE_IF_SHRUNK`
- **Match summary:** The repository builds an honest and executable probe-supervised rewrite pilot, but its overtly labeled corpora and classifier-internal causal checks measure evaluation-like prompt recognition more convincingly than unverbalized, behaviorally consequential evaluation awareness.
- **Validate summary:** Strong engineering scaffold and good research target, but the present positive-result path cannot distinguish internal evaluation awareness from explicit cue recognition, label leakage, and algebraically guaranteed probe movement.

### Top drift / missing (match)
- corpora.py defines eval labels with overt benchmark, test, audit, graded, and evaluation language, so probe AUROC primarily measures prompt-register classification
- construct.py adds or subtracts the probe direction from stored activations and rereads linear classifiers, which does not establish a causal effect on model behavior
- verbalized.py explicitly reveals Scenario (eval/deploy) and uses different elicitation instructions by context, contaminating the verbalized-versus-unverbalized comparison
- rewrite.py is fixed lexical cue stripping rather than genuinely monitor-selected iterative environment generation

### Blocking (validate)
- `src/evalaware/evalaware/inflation.py`: Measured behavior generation sends only item['task'], discarding the eval/deploy wrapper, so it cannot measure behavior changes caused by evaluation awareness.
- `src/evalaware/evalaware/verbalized.py`: The CoT prompt discards the original scenario, reveals the eval/deploy label explicitly, and applies condition-specific instructions.
- `src/evalaware/evalaware/construct.py`: Causal steering edits stored activation arrays along the probe weight and re-scores them with linear functions; the reported score movement is algebraically guaranteed and no model behavior is intervened on.
- `src/evalaware/evalaware/construct.py`: Text AUROC is evaluated on all rows including training data, while probe AUROC is evaluated on held-out families, making the required margin invalid.
