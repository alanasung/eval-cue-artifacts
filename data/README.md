# Data

This directory holds **tiny checked-in manifests only**. Raw corpora, embeddings,
and result tables are gitignored.

## Contract

| Path | Committed? | Purpose |
|---|---|---|
| `data/README.md` | yes | This file |
| `data/manifests/*.json` | yes if < 2 KB | Dataset provenance (seed, n, version) |
| `data/raw/` | never | Downstream download target |
| `data/processed/` | never | Split caches |

## How to obtain data

1. Prefer synthetic pilot items via `scripts/build_dataset.py data=synthetic`.
2. For external corpora, document the exact URL, license, and revision in the
   manifest and never commit the rows.
3. Pilot configs require `n_items >= 512` so the test split after a 60/20/20 cut
   still supports bootstrap intervals.

## Honesty

Anything produced from synthetic builders must be labelled `is_synthetic: true`
in result payloads and must never appear in a measured-results table.
