"""Device and dtype resolution, written for Apple Silicon first.

The target machine for pilot runs is an Apple M4, which makes two failure modes
routine enough to deserve code rather than documentation.

The first is bfloat16. It is the default choice on modern NVIDIA hardware and
appears in every config people copy, but several MPS kernels either fall back to
float32 silently or produce wrong values. Rather than let a run finish and
report numbers nobody can trust, :func:`resolve_dtype` downgrades bfloat16 to
float16 on MPS and logs a warning naming the substitution, so the swap appears
in the run log next to the numbers it affected.

The second is a config that asks for CUDA on a machine that has none. The
helpful-looking response is to fall back to CPU, but a pilot config silently
running the full profile on CPU is how a run that should take ten minutes takes
eleven hours. :func:`resolve_device` raises instead, and the message names the
override that fixes it.

Unified memory makes the third case subtle: on Apple Silicon the GPU shares
system RAM, so an allocation that would OOM a discrete GPU instead pushes the
whole machine into swap and slows to a crawl without ever erroring.
:func:`guard_memory` checks the budget up front for that reason.
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from typing import Any, Literal

import torch

__all__ = [
    "DeviceInfo",
    "HardwareError",
    "describe",
    "get_device",
    "guard_memory",
    "resolve_device",
    "resolve_dtype",
    "enable_mps_fallback",
]

logger = logging.getLogger(__name__)

DeviceName = Literal["cpu", "cuda", "mps"]

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_BYTES_PER_GB = 1024**3


class HardwareError(RuntimeError):
    """Raised when the requested hardware is unavailable or over budget.

    A distinct type so entry points can catch a hardware problem and print a
    setup hint, without swallowing genuine numerical or data errors.
    """


def _as_name(value: Any) -> str:
    """Accept a ``DeviceKind``/``DType`` enum, a plain string, or ``None``."""
    if value is None:
        return "auto"
    return str(getattr(value, "value", value)).lower()


def _total_memory_gb(device: str) -> float:
    """Total memory visible to the given device, in GiB.

    On MPS this is system RAM, because Apple Silicon is unified memory and the
    GPU has no separate pool to report.
    """
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / _BYTES_PER_GB
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / _BYTES_PER_GB
    return 0.0


def _available_memory_gb(device: str) -> float:
    """Memory believed free right now, in GiB. Best effort, never raises."""
    if device == "cuda" and torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        return free / _BYTES_PER_GB
    if hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / _BYTES_PER_GB
    return _total_memory_gb(device)


def resolve_device(requested: Any = "auto") -> str:
    """Turn a requested device into a concrete one, or explain why it cannot.

    Args:
        requested: ``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda"``, or the matching
            ``DeviceKind`` enum member.

    Returns:
        One of ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Raises:
        ValueError: If the name is not a known device kind.
        HardwareError: If a specific accelerator was requested and is absent.
            Falling back quietly turns a ten-minute pilot into an overnight CPU
            run, so the caller has to choose the fallback deliberately.
    """
    name = _as_name(requested)
    if name not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError(
            f"unknown device {name!r}; expected one of ['auto', 'cpu', 'mps', 'cuda'] "
            "(set model.device in the config)"
        )
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise HardwareError(
            "model.device=cuda was requested but torch reports no CUDA device on this "
            f"machine ({platform.machine()}). Set model.device=auto to use the best "
            "available backend, or model.device=cpu to accept the slowdown explicitly."
        )
    if name == "mps" and not torch.backends.mps.is_available():
        raise HardwareError(
            "model.device=mps was requested but the MPS backend is unavailable; "
            "this needs macOS on Apple Silicon with a torch build that includes MPS. "
            "Set model.device=auto instead."
        )
    return name


def resolve_dtype(requested: Any = "float32", device: str = "cpu") -> torch.dtype:
    """Pick a dtype the chosen backend can actually run.

    bfloat16 on MPS is downgraded to float16 with a logged warning: several MPS
    kernels do not implement it, and the resulting fallbacks are silent. float16
    on CPU is likewise downgraded to float32, since CPU float16 matmuls are
    emulated and slower than the wider type they were meant to beat.

    Raises:
        ValueError: If the dtype name is not one of the three supported.
    """
    name = _as_name(requested)
    if name == "auto":
        name = "float16" if device in ("cuda", "mps") else "float32"
    if name not in _DTYPES:
        raise ValueError(
            f"unknown dtype {name!r}; expected one of {sorted(_DTYPES)} "
            "(set model.dtype in the config)"
        )
    if name == "bfloat16" and device == "mps":
        logger.warning(
            "bfloat16 is not reliably implemented on MPS; using float16 instead. "
            "Numbers from this run were produced in float16, not bfloat16."
        )
        name = "float16"
    if name == "float16" and device == "cpu":
        logger.warning(
            "float16 matmuls are emulated on CPU and are slower than float32; "
            "using float32 instead."
        )
        name = "float32"
    return _DTYPES[name]


def enable_mps_fallback() -> bool:
    """Set PYTORCH_ENABLE_MPS_FALLBACK and return whether it is active."""
    if not torch.backends.mps.is_available():
        return False
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved hardware state, recorded in every run manifest."""

    device: str
    dtype: str
    total_memory_gb: float
    available_memory_gb: float
    unified_memory: bool
    platform: str
    torch_version: str
    notes: list[str]
    mps_fallback_enabled: bool = False

    @property
    def is_accelerated(self) -> bool:
        return self.device in ("cuda", "mps")

    def describe(self) -> str:
        """One human-readable line for the log header of a run."""
        return (
            f"{self.device} / {self.dtype} / {self.total_memory_gb:.1f} GiB total"
            f"{' unified' if self.unified_memory else ''} / torch {self.torch_version}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "dtype": self.dtype,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "available_memory_gb": round(self.available_memory_gb, 2),
            "unified_memory": self.unified_memory,
            "platform": self.platform,
            "torch_version": self.torch_version,
            "is_accelerated": self.is_accelerated,
            "notes": list(self.notes),
            "mps_fallback_enabled": self.mps_fallback_enabled,
        }


def get_device(cfg: Any = None) -> DeviceInfo:
    """Resolve device and dtype together and snapshot the result.

    Args:
        cfg: A ``ModelConfig``, a full ``Config`` (its ``.model`` is used), or
            ``None`` for defaults.

    Returns:
        A :class:`DeviceInfo` ready to embed in ``run_metadata.json``.
    """
    model_cfg = getattr(cfg, "model", cfg)
    device = resolve_device(getattr(model_cfg, "device", "auto"))
    dtype = resolve_dtype(getattr(model_cfg, "dtype", "float32"), device)

    notes: list[str] = []
    mps_fallback = False
    if device == "mps":
        mps_fallback = enable_mps_fallback()
        notes.append(
            "MPS has no deterministic-algorithms mode; small numeric drift between "
            "runs is expected and is not an effect"
        )
        notes.append("unified memory: GPU allocations compete with system RAM")
    if device == "cpu":
        notes.append("running on CPU; expect generation to be one to two orders slower")

    return DeviceInfo(
        device=device,
        dtype=str(dtype).replace("torch.", ""),
        total_memory_gb=_total_memory_gb(device),
        available_memory_gb=_available_memory_gb(device),
        unified_memory=device == "mps",
        platform=f"{platform.system()} {platform.machine()}",
        torch_version=torch.__version__,
        notes=notes,
        mps_fallback_enabled=mps_fallback,
    )


def guard_memory(required_gb: float, *, device: str | None = None, headroom_gb: float = 2.0) -> None:
    """Refuse to start work that will not fit, before it starts.

    On unified-memory machines an over-large allocation does not fail cleanly;
    it swaps, and a run that should take minutes takes hours while looking
    healthy. Checking up front converts that into an immediate, explicable stop.

    Args:
        required_gb: Estimated peak footprint of the work about to be done.
        device: Device to check; resolved automatically when omitted.
        headroom_gb: Reserve left for the operating system and the rest of the
            process. Zero is never the right value on a laptop.

    Raises:
        ValueError: If ``required_gb`` is not positive.
        HardwareError: If the requirement plus headroom exceeds free memory.
    """
    if required_gb <= 0:
        raise ValueError(f"required_gb must be > 0, got {required_gb}")
    target = device or resolve_device("auto")
    available = _available_memory_gb(target)
    if available <= 0:
        logger.debug("memory reporting unavailable on %s; skipping the budget check", target)
        return
    if required_gb + headroom_gb > available:
        raise HardwareError(
            f"this stage needs about {required_gb:.1f} GiB plus {headroom_gb:.1f} GiB headroom "
            f"but only {available:.1f} GiB is free on {target}. Lower model.batch_size, "
            "choose a smaller model from the registry, or raise run.max_memory_gb after "
            "closing other applications."
        )


def describe(cfg: Any = None) -> str:
    """Shorthand for ``get_device(cfg).describe()``."""
    return get_device(cfg).describe()
