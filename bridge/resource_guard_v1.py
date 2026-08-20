from __future__ import annotations

import os
import time

from dataclasses import dataclass
from pathlib import Path


GIB = 1024 ** 3
MIB = 1024 ** 2


RESOURCE_GUARD_VERSION = (
    "loopycuts_resource_guard_v1"
)


RESOURCE_STATE_NORMAL = "NORMAL"
RESOURCE_STATE_WARNING = "WARNING"

RESOURCE_STATE_ABORT = (
    "RESOURCE_ABORT"
)

RESOURCE_STATE_ABORT_EMERGENCY = (
    "RESOURCE_ABORT_EMERGENCY"
)


@dataclass(frozen=True)
class ProcessMemorySnapshot:
    pid: int
    rss_bytes: int
    swap_bytes: int

    @property
    def footprint_bytes(
        self,
    ) -> int:
        return (
            int(self.rss_bytes)
            +
            int(self.swap_bytes)
        )


@dataclass(frozen=True)
class ResourceSnapshot:
    mem_available_bytes: int

    swap_total_bytes: int
    swap_free_bytes: int
    swap_used_bytes: int

    python_memory: ProcessMemorySnapshot

    cpp_memory: (
        ProcessMemorySnapshot
        | None
    )


def _parse_kib_file(
    path: Path,
):
    result = {}

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if ":" not in line:
            continue

        key, value = line.split(
            ":",
            1,
        )

        tokens = (
            value
            .strip()
            .split()
        )

        if not tokens:
            continue

        try:
            numeric = int(
                tokens[0]
            )

        except ValueError:
            continue

        result[key] = (
            numeric
            *
            1024
        )

    return result


def read_process_memory(
    pid: int,
) -> ProcessMemorySnapshot:
    pid = int(
        pid
    )

    status_path = (
        Path("/proc")
        /
        str(pid)
        /
        "status"
    )

    if not status_path.is_file():
        return ProcessMemorySnapshot(
            pid=pid,
            rss_bytes=0,
            swap_bytes=0,
        )

    data = _parse_kib_file(
        status_path
    )

    return ProcessMemorySnapshot(
        pid=pid,

        rss_bytes=int(
            data.get(
                "VmRSS",
                0,
            )
        ),

        swap_bytes=int(
            data.get(
                "VmSwap",
                0,
            )
        ),
    )


def read_resource_snapshot(
    *,
    cpp_pid: int | None = None,
) -> ResourceSnapshot:
    meminfo = _parse_kib_file(
        Path(
            "/proc/meminfo"
        )
    )

    swap_total = int(
        meminfo.get(
            "SwapTotal",
            0,
        )
    )

    swap_free = int(
        meminfo.get(
            "SwapFree",
            0,
        )
    )

    swap_used = max(
        0,
        swap_total - swap_free,
    )

    python_memory = (
        read_process_memory(
            os.getpid()
        )
    )

    cpp_memory = None

    if cpp_pid is not None:
        cpp_memory = (
            read_process_memory(
                int(cpp_pid)
            )
        )

    return ResourceSnapshot(
        mem_available_bytes=int(
            meminfo.get(
                "MemAvailable",
                0,
            )
        ),

        swap_total_bytes=swap_total,

        swap_free_bytes=swap_free,

        swap_used_bytes=swap_used,

        python_memory=python_memory,

        cpp_memory=cpp_memory,
    )


class ResourceGuardPolicyV1:
    """
    Pure ResourceGuard state machine.

    It performs NO process termination itself.

    Frozen design intent for the upcoming V3 integration:

        NORMAL:
            SwapUsed < 8 GiB

        WARNING:
            8 GiB <= SwapUsed < 10 GiB

        HARD RESOURCE ABORT:
            SwapUsed >= 10 GiB continuously
            for at least 8 seconds

        EMERGENCY RESOURCE ABORT:
            SwapUsed >= 12 GiB immediately

        RE-ARM:
            SwapUsed <= 6 GiB

    Both abort states mean:

        abort CURRENT LoopyCuts episode only

    They do NOT mean:

        abort the entire formal training run
    """

    def __init__(
        self,
        *,
        warning_swap_used_bytes: int =
            8 * GIB,

        abort_swap_used_bytes: int =
            10 * GIB,

        emergency_swap_used_bytes: int =
            12 * GIB,

        rearm_swap_used_bytes: int =
            6 * GIB,

        abort_hold_seconds: float =
            8.0,
    ):
        self.warning_swap_used_bytes = int(
            warning_swap_used_bytes
        )

        self.abort_swap_used_bytes = int(
            abort_swap_used_bytes
        )

        self.emergency_swap_used_bytes = int(
            emergency_swap_used_bytes
        )

        self.rearm_swap_used_bytes = int(
            rearm_swap_used_bytes
        )

        self.abort_hold_seconds = float(
            abort_hold_seconds
        )

        if not (
            0
            <=
            self.rearm_swap_used_bytes
            <
            self.warning_swap_used_bytes
            <
            self.abort_swap_used_bytes
            <
            self.emergency_swap_used_bytes
        ):
            raise ValueError(
                "ResourceGuard thresholds must satisfy "
                "rearm < warning < abort < emergency"
            )

        if (
            self.abort_hold_seconds
            <=
            0.0
        ):
            raise ValueError(
                "abort_hold_seconds must be positive"
            )

        self._abort_threshold_entered_at = None


    def reset_episode(
        self,
    ):
        self._abort_threshold_entered_at = None


    def can_rearm(
        self,
        snapshot: ResourceSnapshot,
    ) -> bool:
        return (
            int(
                snapshot.swap_used_bytes
            )
            <=
            self.rearm_swap_used_bytes
        )


    def evaluate(
        self,
        snapshot: ResourceSnapshot,
        *,
        now_seconds: float | None = None,
    ) -> str:
        """
        Evaluate one resource sample.

        `now_seconds` exists so tests can use deterministic synthetic
        time. Production callers normally omit it and use
        time.monotonic().
        """

        if now_seconds is None:
            now_seconds = (
                time.monotonic()
            )

        now_seconds = float(
            now_seconds
        )

        swap_used = int(
            snapshot.swap_used_bytes
        )

        # ----------------------------------------------------------
        # Emergency threshold:
        # immediate CURRENT-EPISODE abort.
        # ----------------------------------------------------------

        if (
            swap_used
            >=
            self.emergency_swap_used_bytes
        ):
            return (
                RESOURCE_STATE_ABORT_EMERGENCY
            )

        # ----------------------------------------------------------
        # Hard threshold:
        # must remain >=10 GiB continuously for >=8 seconds.
        # ----------------------------------------------------------

        if (
            swap_used
            >=
            self.abort_swap_used_bytes
        ):
            if (
                self._abort_threshold_entered_at
                is None
            ):
                self._abort_threshold_entered_at = (
                    now_seconds
                )

                return (
                    RESOURCE_STATE_WARNING
                )

            elapsed = (
                now_seconds
                -
                self._abort_threshold_entered_at
            )

            if (
                elapsed
                >=
                self.abort_hold_seconds
            ):
                return (
                    RESOURCE_STATE_ABORT
                )

            return (
                RESOURCE_STATE_WARNING
            )

        # Dropping below 10 GiB resets the continuous-duration clock.
        self._abort_threshold_entered_at = None

        if (
            swap_used
            >=
            self.warning_swap_used_bytes
        ):
            return (
                RESOURCE_STATE_WARNING
            )

        return (
            RESOURCE_STATE_NORMAL
        )
