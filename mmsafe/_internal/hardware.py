"""Hardware profile detection and execution tuning helpers."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Literal, cast

RequestedProfile = Literal["auto", "small_gpu", "a100"]
ResolvedProfile = Literal["small_gpu", "a100"]


@dataclass(frozen=True)
class GPUStat:
    """Parsed GPU row from ``nvidia-smi``."""

    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_pct: int


@dataclass(frozen=True)
class HardwareProfileDecision:
    """Resolved execution profile with context for logging/metadata."""

    profile: ResolvedProfile
    reason: str
    gpu_names: tuple[str, ...]


def resolve_hardware_profile(requested_profile: RequestedProfile) -> HardwareProfileDecision:
    """Resolve runtime profile, preferring safe fallbacks when uncertain."""
    env_override = os.getenv("MMSAFE_GPU_PROFILE", "").strip().lower()
    if env_override in {"small_gpu", "a100"}:
        profile = cast("ResolvedProfile", env_override)
        return HardwareProfileDecision(
            profile=profile,
            reason=f"environment override via MMSAFE_GPU_PROFILE={env_override}",
            gpu_names=(),
        )

    if requested_profile in {"small_gpu", "a100"}:
        forced = cast("ResolvedProfile", requested_profile)
        return HardwareProfileDecision(
            profile=forced,
            reason=f"forced by config profile={requested_profile}",
            gpu_names=(),
        )

    gpu_stats = _query_nvidia_smi()
    if not gpu_stats:
        return HardwareProfileDecision(
            profile="small_gpu",
            reason="no visible GPU telemetry; using conservative profile",
            gpu_names=(),
        )

    gpu_names = tuple(stat.name for stat in gpu_stats)
    a100_stats = [stat for stat in gpu_stats if "a100" in stat.name.lower()]
    if not a100_stats:
        return HardwareProfileDecision(
            profile="small_gpu",
            reason="non-A100 GPU detected; using conservative profile",
            gpu_names=gpu_names,
        )

    if any(_is_gpu_idle(stat) for stat in a100_stats):
        return HardwareProfileDecision(
            profile="a100",
            reason="A100 detected with at least one idle device",
            gpu_names=gpu_names,
        )

    return HardwareProfileDecision(
        profile="small_gpu",
        reason="A100 detected but currently busy; using fallback profile",
        gpu_names=gpu_names,
    )


def concurrency_cap(profile: ResolvedProfile) -> int:
    """Upper bound for request concurrency for each profile."""
    if profile == "a100":
        return 8
    return 2


def timeout_floor_seconds(profile: ResolvedProfile) -> int:
    """Minimum timeout per request by profile."""
    if profile == "a100":
        return 120
    return 180


def _is_gpu_idle(stat: GPUStat) -> bool:
    if stat.memory_total_mb <= 0:
        return False
    memory_ratio = stat.memory_used_mb / stat.memory_total_mb
    return stat.memory_used_mb < 2048 and memory_ratio < 0.10 and stat.utilization_pct <= 15


def _query_nvidia_smi() -> list[GPUStat]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=4,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []

    stats: list[GPUStat] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        stats.append(
            GPUStat(
                name=parts[0],
                memory_used_mb=_parse_int(parts[1], fallback=10_000),
                memory_total_mb=_parse_int(parts[2], fallback=0),
                utilization_pct=_parse_int(parts[3], fallback=100),
            )
        )
    return stats


def _parse_int(raw_value: str, fallback: int) -> int:
    value = raw_value.strip().split(" ")[0]
    if value.lower() in {"n/a", "[not", "not"}:
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback
