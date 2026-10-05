"""Stream caches take memory as they grow: the room the startup plan left them, what they hold, and a reserve."""

from __future__ import annotations

import os
from typing import Callable

from .capacity import cuda_limit_bytes

RESERVE_ENV = "TENSORFOLD_CACHE_RESERVE_MIB"
UNIFIED_RESERVE = 2 * 1024**3        # a unified GPU shares its memory with the host: keep this much back
DISCRETE_RESERVE = 512 * 1024**2     # a discrete card's floor; warm() raises it to the transients it measures
RESERVE_MARGIN = 256 * 1024**2       # allocator rounding and fragmentation above a measured transient peak


def reserve_bytes(workspace: int, unified: bool) -> tuple[int, bool]:
    """The gate's reserve and whether warm() may raise it by measurement: RESERVE_ENV MiB as given, else the prompt
    workspace and at least UNIFIED_RESERVE (unified memory) or DISCRETE_RESERVE (a discrete card)."""

    value = os.environ.get(RESERVE_ENV, "").strip()
    if value:
        if not value.isdecimal():
            raise ValueError(f"{RESERVE_ENV}: whole MiB, not {value!r}")
        return int(value) * 1024**2, False
    return max(UNIFIED_RESERVE if unified else DISCRETE_RESERVE, int(workspace)), not unified


class NoRoom(RuntimeError):
    """A request that can't start until a live stream finishes and frees its caches."""


class MemoryGate:
    """``room`` bytes for every stream's growing caches; ``fits`` also asks the host (MemAvailable on a unified GPU)."""

    def __init__(self, room: int, reserve: int, live: Callable[[], int] | None = None) -> None:
        self.room, self.reserve, self.held = int(room), int(reserve), 0
        self.live = live
        self.waits = self.ends = 0
        self.peak = 0                        # the most ``held`` has been

    def fits(self, extra: int) -> bool:
        """Whether ``extra`` more bytes can be allocated now (a resize holds its old buffers until the copy lands)."""

        if self.held + int(extra) > self.room - self.reserve:
            return False
        return self.live is None or self.live() >= int(extra) + self.reserve

    def take(self, extra: int) -> None:
        self.held += int(extra)
        self.peak = max(self.peak, self.held)

    def give(self, freed: int) -> None:
        self.held = max(0, self.held - int(freed))


def torch_live(torch, available: Callable) -> Callable[[], int]:
    """Reusable allocator bytes plus free memory, capped by the budget left after current allocations."""

    def room() -> int:
        allocated = int(torch.cuda.memory_allocated())
        free = int(available(torch)) + int(torch.cuda.memory_reserved()) - allocated
        limit = cuda_limit_bytes()
        return max(0, min(free, limit - allocated)) if limit is not None else max(0, free)

    return room


__all__ = ["MemoryGate", "NoRoom", "RESERVE_ENV", "reserve_bytes", "torch_live"]
