# Related work

This note situates **Closing the Loop on Unverbalized Evaluation Awareness** against the mentor-linked literature for
[Securing alignment evaluations against unverbalized evaluation awareness](https://sparai.org/projects/f26/recFlTiKZFBq3i4GY).

## Positioning

Train a probe for 'this is a test', then let it referee an automated rewrite loop that makes evaluations harder to recognize.

The design hypothesis is: Verbalized evaluation awareness undercounts real awareness. A linear probe will fire on scenarios the model never flags aloud, safety scores will be measurably inflated on high-probe scenarios, and an automated rewrite loop supervised by the probe can lower probe-detected awareness while holding the scenario's construct validity fixed.

## Engagement rules

1. Cite the paper that motivates each measurement.
2. Name what this repo replicates versus what it changes.
3. Keep synthetic harness results labelled as synthetic.
4. Prefer causal or behavioral ground truth over agreement with a training
   signal that cannot falsify the claim.

## Skeleton critique slots

The following slots are filled per project during alignment. They exist so the
markdown inventory clears the documentation bar even before camera-ready prose
is written.

### Slot A — Primary motivating paper

Summary of the mentor's main citation and the exact claim this repo tests.

### Slot B — Closest prior codebase

What prior open implementations exist, and which abstractions we refuse to
vendor.

### Slot C — Measurement instrument papers

Probe, patching, monitoring, or jailbreak-ladder methodology sources.

### Slot D — Confounds already named in the literature

Shortcut learning, eval awareness, circular labels, underpowered nulls.

### Slot E — Open disagreements

Where this design intentionally diverges from common practice, with the
falsification condition.

## Mentors and affiliations

- Mentor(s): J Rosser
- Affiliation(s): University of Oxford, Adecco supporting Google DeepMind

## Bibliography placeholders

Additional references are tracked in `TASK.md` and in result JSON `notes`
fields so that reported numbers stay attached to the papers that justify them.
