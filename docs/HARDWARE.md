# Hardware

## Summary (X13)

| Workload | Device | Notes |
|---|---|---|
| Torch model forward / generate / hooks | `cpu`, `mps`, or `cuda` via `model.device` | Resolved by `evalaware.models.device` |
| Activation capture / steering | same as model device, tensors moved to CPU for storage | Avoids unified-memory exhaustion |
| Sklearn / numpy / pandas metrics | **CPU only** | Not MPS-accelerated |
| Matplotlib figure export | **CPU only** | Agg backend |
| Bootstrap / TOST / MDE | **CPU only** | numpy |

Saying a pipeline is "MPS-accelerated" when only the transformer forward pass
touches MPS is misleading. Non-torch work runs on CPU even when
`model.device=mps`.

## Apple Silicon (M-series)

- Pilot target: Apple M4 with 16 GiB unified memory.
- On MPS, `PYTORCH_ENABLE_MPS_FALLBACK=1` is set explicitly and recorded as
  `mps_fallback_enabled` in `DeviceInfo.to_dict()`.
- `bfloat16` is downgraded to `float16` on MPS with a logged warning.
- There is no deterministic-algorithms mode on MPS; small numeric drift between
  runs is expected.

## CUDA

- Requesting `model.device=cuda` on a machine without CUDA raises
  `HardwareError` instead of silently falling back to CPU.
- Use `model.device=auto` to pick the best available backend.

## Memory guard

`guard_memory(required_gb=...)` refuses work that will not fit before it starts.
On unified memory, over-large allocations swap rather than OOM cleanly.

## CI

GitHub Actions runs on `ubuntu-latest` CPUs. Tests use a tiny randomly
initialised GPT-2 (2 layers) and never download weights.
