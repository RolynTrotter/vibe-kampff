"""Device selection, memory probing and version reporting.

Split out from ``smoke.py`` because the memory probe is the one piece with
genuinely platform-specific behaviour, and it is easier to get wrong quietly
than to get wrong loudly.
"""

from __future__ import annotations

import platform
import resource
import sys
from dataclasses import asdict, dataclass

import torch


def pick_device(requested: str = "auto") -> str:
    """Resolve a device string, preferring accelerators over CPU."""
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[name]
    except KeyError:
        raise SystemExit(f"unknown dtype {name!r}") from None


def peak_rss_gb() -> float:
    """Peak resident set size of this process, in GB.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS. Getting this
    backwards silently reports figures off by 1024x, which on a memory
    ceiling test is the whole point of the measurement.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024
    return raw * scale / 1e9


def accelerator_allocated_gb(device: str) -> float | None:
    """Currently-allocated accelerator memory in GB, or None on CPU.

    On unified-memory Macs this overlaps with RSS rather than adding to it —
    the GPU allocation is not separate from system memory. Report both and
    don't sum them.
    """
    if device == "mps" and hasattr(torch, "mps"):
        return torch.mps.current_allocated_memory() / 1e9
    if device == "cuda":
        return torch.cuda.memory_allocated() / 1e9
    return None


def reset_peak_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def empty_cache(device: str) -> None:
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


@dataclass
class EnvReport:
    python: str
    platform: str
    machine: str
    torch: str
    transformers: str
    jlens: str
    device: str
    mps_available: bool
    mps_built: bool
    cuda_available: bool

    def as_dict(self) -> dict:
        return asdict(self)


def describe(device: str) -> EnvReport:
    import jlens
    import transformers

    return EnvReport(
        python=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        jlens=getattr(jlens, "__version__", "unknown (installed from git)"),
        device=device,
        mps_available=torch.backends.mps.is_available(),
        mps_built=torch.backends.mps.is_built(),
        cuda_available=torch.cuda.is_available(),
    )
