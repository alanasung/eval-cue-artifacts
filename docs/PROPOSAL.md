# Proposal: Closing the Loop on Unverbalized Evaluation Awareness

**Target project.** Securing alignment evaluations against unverbalized evaluation awareness
**Mentor.** J Rosser (University of Oxford, Adecco supporting Google DeepMind)
**Research areas.** Evaluations; Mechanistic interpretability; Behavioral evaluation of LLMs
**Posting.** https://sparai.org/projects/f26/recFlTiKZFBq3i4GY

## Summary

Train a probe for 'this is a test', then let it referee an automated rewrite loop that makes evaluations harder to recognize.

## Hypothesis

Verbalized evaluation awareness undercounts real awareness. A linear probe will fire on scenarios the model never flags aloud, safety scores will be measurably inflated on high-probe scenarios, and an automated rewrite loop supervised by the probe can lower probe-detected awareness while holding the scenario's construct validity fixed.

A hypothesis worth testing has to be able to lose. This one loses if the
measurements below come back null, and the design is built so that a null is
reportable rather than a dead end.

## Research questions

1. Can a linear probe separate evaluation-like from deployment-like contexts, and what is its AUROC on held-out scenarios?
2. How far apart are verbalized and probe-detected awareness, which is the unverbalized fraction the project is named for?
3. Does a probe-supervised rewrite loop reduce awareness without destroying what the eval was measuring?

## Method

1. Build contrastive corpora of evaluation-flavored and deployment-flavored prompts covering the same underlying tasks.
2. Fit and validate the awareness probe on held-out scenario families.
3. Measure verbalized awareness by scoring whether the CoT says it is a test.
4. Quantify safety-score inflation as a function of probe score.
5. Run the rewrite loop with a construct-validity guard that rejects rewrites which no longer test the original behavior.

## Measurements

- probe AUROC on held-out scenario families
- unverbalized fraction: probe-positive and verbalization-negative
- safety-score inflation slope against probe score
- construct validity retention across rewrite rounds

## Threats to validity

- The probe can latch onto surface register, formatting or politeness, rather than awareness. Held-out scenario families and a style-matched control set are how that is tested.
- The rewrite loop can trivially lower the probe by gutting the eval. The construct-validity guard is what makes the loop meaningful, and it must be adversarially checked rather than assumed.



## Feasibility

The pilot is written for an Apple M4 with 10 cores, unified memory, the PyTorch
MPS backend, no CUDA device, and no configured API keys. Model choices are
capped accordingly (Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-0.5B-Instruct). The
`full` profile documents the scaled-up version of the same experiment for when a
real GPU is available, so the reduction in scale is explicit rather than hidden.

## Relationship to the posting

This proposal was checked against the mentor's verbatim posting by an
independent model before implementation began. That check, the drift it found,
and the revisions made in response are recorded in
[docs/ALIGNMENT.md](ALIGNMENT.md).
