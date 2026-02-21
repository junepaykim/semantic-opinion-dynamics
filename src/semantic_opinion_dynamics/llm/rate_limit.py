from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass
class AsyncRateLimiter:
    max_concurrency: int
    min_interval_sec: float

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.max_concurrency)
        self._lock = asyncio.Lock()
        self._last_ts = 0.0

    async def __aenter__(self) -> "AsyncRateLimiter":
        await self._sem.acquire()
        if self.min_interval_sec > 0:
            async with self._lock:
                now = time.monotonic()
                wait = self.min_interval_sec - (now - self._last_ts)
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_ts = time.monotonic()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._sem.release()
