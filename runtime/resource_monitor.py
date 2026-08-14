from __future__ import annotations

import threading
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path


KB_PER_MIB = 1024.0


@dataclass(frozen=True)
class ResourceStats:
    samples: int

    peak_rss_mb: float
    peak_process_swap_mb: float

    min_mem_available_mb: float
    max_system_swap_used_mb: float

    monitor_elapsed_s: float

    def to_dict(self):
        return asdict(self)


def _parse_key_value_kb(
    path: Path,
    wanted: set[str],
):
    result = {}

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except (
        FileNotFoundError,
        ProcessLookupError,
        PermissionError,
    ):
        return result

    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue

        key, rest = raw_line.split(
            ":",
            1,
        )

        if key not in wanted:
            continue

        parts = rest.strip().split()

        if not parts:
            continue

        try:
            value_kb = float(
                parts[0]
            )
        except ValueError:
            continue

        result[key] = (
            value_kb
            /
            KB_PER_MIB
        )

    return result


class ResourceMonitor:
    """
    Passive resource monitor.

    IMPORTANT:
        This class NEVER terminates or modifies the monitored
        process.

        It has no effect on LoopyCuts action selection,
        episode termination, reward, finalization, or SAC.

    It only samples:
        /proc/<pid>/status
        /proc/meminfo
    """

    def __init__(
        self,
        pid: int,
        sample_interval_s: float = 1.0,
    ):
        if pid <= 0:
            raise ValueError(
                f"Invalid pid: {pid}"
            )

        if sample_interval_s <= 0:
            raise ValueError(
                "sample_interval_s must be positive"
            )

        self.pid = int(
            pid
        )

        self.sample_interval_s = float(
            sample_interval_s
        )

        self._stop_event = (
            threading.Event()
        )

        self._thread = None

        self._start_time = None

        self._samples = 0

        self._peak_rss_mb = 0.0
        self._peak_process_swap_mb = 0.0

        self._min_mem_available_mb = float(
            "inf"
        )

        self._max_system_swap_used_mb = 0.0

    def start(self):
        if self._thread is not None:
            raise RuntimeError(
                "ResourceMonitor already started"
            )

        self._start_time = (
            time.perf_counter()
        )

        self._thread = threading.Thread(
            target=self._run,
            name=(
                f"resource-monitor-{self.pid}"
            ),
            daemon=True,
        )

        self._thread.start()

        return self

    def _sample_once(self):
        process = _parse_key_value_kb(
            Path(
                f"/proc/{self.pid}/status"
            ),
            {
                "VmRSS",
                "VmSwap",
            },
        )

        system = _parse_key_value_kb(
            Path(
                "/proc/meminfo"
            ),
            {
                "MemAvailable",
                "SwapTotal",
                "SwapFree",
            },
        )

        if "VmRSS" in process:
            self._peak_rss_mb = max(
                self._peak_rss_mb,
                process[
                    "VmRSS"
                ],
            )

        if "VmSwap" in process:
            self._peak_process_swap_mb = max(
                self._peak_process_swap_mb,
                process[
                    "VmSwap"
                ],
            )

        if "MemAvailable" in system:
            self._min_mem_available_mb = min(
                self._min_mem_available_mb,
                system[
                    "MemAvailable"
                ],
            )

        if (
            "SwapTotal" in system
            and
            "SwapFree" in system
        ):
            swap_used = (
                system[
                    "SwapTotal"
                ]
                -
                system[
                    "SwapFree"
                ]
            )

            self._max_system_swap_used_mb = max(
                self._max_system_swap_used_mb,
                swap_used,
            )

        self._samples += 1

    def _run(self):
        while not self._stop_event.is_set():
            self._sample_once()

            self._stop_event.wait(
                self.sample_interval_s
            )

    def snapshot(self) -> ResourceStats:
        """
        Return the currently accumulated passive measurements
        WITHOUT stopping the monitor.

        This method only reads statistics already collected by
        the background monitoring thread.

        It does not:
            - stop the monitor;
            - terminate the C++ process;
            - change LoopyCuts state;
            - change actions;
            - change rewards;
            - change episode termination.
        """

        if self._thread is None:
            raise RuntimeError(
                "ResourceMonitor was not started"
            )

        elapsed = (
            time.perf_counter()
            -
            self._start_time
        )

        min_available = (
            self._min_mem_available_mb
        )

        if min_available == float(
            "inf"
        ):
            min_available = 0.0

        return ResourceStats(
            samples=int(
                self._samples
            ),

            peak_rss_mb=float(
                self._peak_rss_mb
            ),

            peak_process_swap_mb=float(
                self._peak_process_swap_mb
            ),

            min_mem_available_mb=float(
                min_available
            ),

            max_system_swap_used_mb=float(
                self._max_system_swap_used_mb
            ),

            monitor_elapsed_s=float(
                elapsed
            ),
        )

    def stop(self) -> ResourceStats:
        if self._thread is None:
            raise RuntimeError(
                "ResourceMonitor was not started"
            )

        self._stop_event.set()

        self._thread.join(
            timeout=(
                self.sample_interval_s
                +
                2.0
            )
        )

        if self._thread.is_alive():
            raise RuntimeError(
                "ResourceMonitor thread did not stop"
            )

        elapsed = (
            time.perf_counter()
            -
            self._start_time
        )

        min_available = (
            self._min_mem_available_mb
        )

        if min_available == float(
            "inf"
        ):
            min_available = 0.0

        return ResourceStats(
            samples=int(
                self._samples
            ),

            peak_rss_mb=float(
                self._peak_rss_mb
            ),

            peak_process_swap_mb=float(
                self._peak_process_swap_mb
            ),

            min_mem_available_mb=float(
                min_available
            ),

            max_system_swap_used_mb=float(
                self._max_system_swap_used_mb
            ),

            monitor_elapsed_s=float(
                elapsed
            ),
        )
