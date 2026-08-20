from __future__ import annotations

import os
import sys

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from bridge.resource_guard_v1 import (
    GIB,

    RESOURCE_STATE_ABORT,
    RESOURCE_STATE_ABORT_EMERGENCY,
    RESOURCE_STATE_NORMAL,
    RESOURCE_STATE_WARNING,

    ProcessMemorySnapshot,
    ResourceGuardPolicyV1,
    ResourceSnapshot,

    read_resource_snapshot,
)


def snapshot(
    swap_used_gib: float,
):
    total = (
        34
        *
        GIB
    )

    used = int(
        swap_used_gib
        *
        GIB
    )

    return ResourceSnapshot(
        mem_available_bytes=
            5 * GIB,

        swap_total_bytes=
            total,

        swap_free_bytes=
            total - used,

        swap_used_bytes=
            used,

        python_memory=
            ProcessMemorySnapshot(
                pid=123,
                rss_bytes=2 * GIB,
                swap_bytes=0,
            ),

        cpp_memory=
            ProcessMemorySnapshot(
                pid=456,
                rss_bytes=4 * GIB,
                swap_bytes=0,
            ),
    )


def main():
    guard = (
        ResourceGuardPolicyV1()
    )

    # ============================================================
    # NORMAL / WARNING
    # ============================================================

    assert (
        guard.evaluate(
            snapshot(
                0.0
            ),
            now_seconds=0.0,
        )
        ==
        RESOURCE_STATE_NORMAL
    )

    assert (
        guard.evaluate(
            snapshot(
                7.9
            ),
            now_seconds=1.0,
        )
        ==
        RESOURCE_STATE_NORMAL
    )

    assert (
        guard.evaluate(
            snapshot(
                8.0
            ),
            now_seconds=2.0,
        )
        ==
        RESOURCE_STATE_WARNING
    )


    # ============================================================
    # >=10 GiB must persist for 8 actual seconds.
    # ============================================================

    guard.reset_episode()

    assert (
        guard.evaluate(
            snapshot(
                10.0
            ),
            now_seconds=100.0,
        )
        ==
        RESOURCE_STATE_WARNING
    )

    assert (
        guard.evaluate(
            snapshot(
                10.3
            ),
            now_seconds=104.0,
        )
        ==
        RESOURCE_STATE_WARNING
    )

    assert (
        guard.evaluate(
            snapshot(
                10.1
            ),
            now_seconds=107.999,
        )
        ==
        RESOURCE_STATE_WARNING
    )

    assert (
        guard.evaluate(
            snapshot(
                10.2
            ),
            now_seconds=108.0,
        )
        ==
        RESOURCE_STATE_ABORT
    )


    # ============================================================
    # Dropping below 10 GiB resets the 8-second clock.
    # ============================================================

    guard.reset_episode()

    assert (
        guard.evaluate(
            snapshot(
                10.5
            ),
            now_seconds=200.0,
        )
        ==
        RESOURCE_STATE_WARNING
    )

    assert (
        guard.evaluate(
            snapshot(
                10.4
            ),
            now_seconds=206.0,
        )
        ==
        RESOURCE_STATE_WARNING
    )

    assert (
        guard.evaluate(
            snapshot(
                9.9
            ),
            now_seconds=207.0,
        )
        ==
        RESOURCE_STATE_WARNING
    )

    # New >=10 GiB period begins here.
    assert (
        guard.evaluate(
            snapshot(
                10.2
            ),
            now_seconds=208.0,
        )
        ==
        RESOURCE_STATE_WARNING
    )

    assert (
        guard.evaluate(
            snapshot(
                10.2
            ),
            now_seconds=215.9,
        )
        ==
        RESOURCE_STATE_WARNING
    )

    assert (
        guard.evaluate(
            snapshot(
                10.2
            ),
            now_seconds=216.0,
        )
        ==
        RESOURCE_STATE_ABORT
    )


    # ============================================================
    # >=12 GiB is immediate CURRENT-EPISODE abort.
    # ============================================================

    guard.reset_episode()

    assert (
        guard.evaluate(
            snapshot(
                12.0
            ),
            now_seconds=300.0,
        )
        ==
        RESOURCE_STATE_ABORT_EMERGENCY
    )


    # ============================================================
    # Re-arm threshold.
    # ============================================================

    assert (
        guard.can_rearm(
            snapshot(
                6.0
            )
        )
        is True
    )

    assert (
        guard.can_rearm(
            snapshot(
                6.1
            )
        )
        is False
    )


    # ============================================================
    # Live /proc smoke.
    # ============================================================

    live = (
        read_resource_snapshot()
    )

    assert (
        live.python_memory.pid
        ==
        os.getpid()
    )

    assert (
        live.python_memory.rss_bytes
        >
        0
    )

    assert (
        live.swap_total_bytes
        >=
        live.swap_used_bytes
    )


    print(
        "resource guard version : "
        "loopycuts_resource_guard_v1"
    )

    print(
        "live Python RSS MiB    :",
        live.python_memory.rss_bytes
        /
        (1024 ** 2),
    )

    print(
        "live Python swap MiB   :",
        live.python_memory.swap_bytes
        /
        (1024 ** 2),
    )

    print(
        "live system swap GiB   :",
        live.swap_used_bytes
        /
        GIB,
    )

    print()

    print(
        "PASS: 8 GiB warning threshold"
    )

    print(
        "PASS: 10 GiB requires continuous 8-second hold"
    )

    print(
        "PASS: dropping below 10 GiB resets hold timer"
    )

    print(
        "PASS: 12 GiB immediately aborts current episode only"
    )

    print(
        "PASS: 6 GiB re-arm threshold"
    )

    print(
        "PASS: live /proc memory telemetry is readable"
    )


if __name__ == "__main__":
    main()
