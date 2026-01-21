from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchLimits:
    depth: int | None = None
    movetime_ms: int | None = None
    wtime_ms: int | None = None
    btime_ms: int | None = None
    winc_ms: int = 0
    binc_ms: int = 0
    infinite: bool = False
