# VALIDATION — eval-awareness-hardening

## Codex v1 (historical)
- Verdict: SERIOUS_PROBLEMS
- Summary: This is a competent generic runner wrapped around an underspecified experiment whose scientific stages are entirely absent, so it cannot run or support the claimed conclusions.

## Codex v2
- Verdict: PASS_WITH_NOTES
- Summary: Analogous to introspection-verbalization Codex v2: X1–X13 OK; stages implemented with a real `make pilot` path; synthetic/proxy pilot default; several model revisions still on `main`.
- KEY_FIXES_OK: X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13

## Grok (dual-validate)
- Verdict: PASS_WITH_NOTES
- Summary: Corpora→probe→monitors→construct→verbalized→inflation→rewrite is registered and implemented with construct-validity guards. Synthetic activations by default; complexity and X-fixes OK → PASS_WITH_NOTES.

### Remaining
- Activation features are synthetic when Hub weights are unavailable; rewrite/construct loop still executes end-to-end.
- Supervising monitor defaults to the linear probe; other monitor classes are stand-ins in the held-out panel.
- Model revisions mostly `main`.

## Reconciliation
v1 absent scientific stages replaced with full rewrite-loop DAG and construct guards. Grok PASS_WITH_NOTES.
