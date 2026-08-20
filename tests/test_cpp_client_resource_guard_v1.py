from __future__ import annotations

import os
import stat
import sys
import tempfile

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


from bridge.cpp_client import (
    LoopyCutsClient,
    RLServerResourceAbort,
)

from bridge.resource_guard_v1 import (
    GIB,

    RESOURCE_STATE_ABORT,
    RESOURCE_STATE_ABORT_EMERGENCY,

    ProcessMemorySnapshot,
    ResourceGuardPolicyV1,
    ResourceSnapshot,
)


def resource_snapshot(
    *,
    cpp_pid,
    swap_used_gib,
):
    swap_total = (
        34
        *
        GIB
    )

    swap_used = int(
        float(
            swap_used_gib
        )
        *
        GIB
    )

    return ResourceSnapshot(
        mem_available_bytes=
            512 * 1024 * 1024,

        swap_total_bytes=
            swap_total,

        swap_free_bytes=
            swap_total - swap_used,

        swap_used_bytes=
            swap_used,

        python_memory=
            ProcessMemorySnapshot(
                pid=
                    os.getpid(),

                rss_bytes=
                    512 * 1024 * 1024,

                swap_bytes=
                    0,
            ),

        cpp_memory=
            ProcessMemorySnapshot(
                pid=
                    int(
                        cpp_pid
                    ),

                rss_bytes=
                    4 * GIB,

                swap_bytes=
                    6 * GIB,
            ),
    )


def write_fake_server(
    root: Path,
):
    path = (
        root
        /
        "fake_rl_server.py"
    )

    path.write_text(
r'''#!/usr/bin/env python3

import sys
import time


def emit_initial():
    print("[RL] READY")
    print(
        "[RL] STATE "
        "verts=100 "
        "tets=100 "
        "terminal=0 "
        "convergence=0 "
        "regular_phase_closed=0"
    )
    print("[RL] USED")
    print("[RL] REVERTED")
    print("[RL] NICO_BUG")
    print("[RL] TOP_RELEVANT")
    print("[RL] ACTIONS 0")
    sys.stdout.flush()


emit_initial()


for raw in sys.stdin:
    command = raw.strip()

    if command.startswith("STEP "):
        # Deliberately block long enough for the synthetic
        # ResourceGuard watchdog to terminate this process.
        time.sleep(30.0)

    elif command == "QUIT":
        print("[RL] BYE")
        sys.stdout.flush()
        break

    elif command == "STATE":
        print(
            "[RL] STATE "
            "verts=100 "
            "tets=100 "
            "terminal=0 "
            "convergence=0 "
            "regular_phase_closed=0"
        )
        print("[RL] USED")
        print("[RL] REVERTED")
        print("[RL] NICO_BUG")
        print("[RL] TOP_RELEVANT")
        print("[RL] ACTIONS 0")
        sys.stdout.flush()
''',
        encoding="utf-8",
    )

    mode = (
        path.stat().st_mode
    )

    path.chmod(
        mode
        |
        stat.S_IXUSR
    )

    return path


def run_emergency_test(
    *,
    executable,
    mesh,
    loops,
):
    def reader(
        *,
        cpp_pid,
    ):
        return resource_snapshot(
            cpp_pid=
                cpp_pid,

            swap_used_gib=
                12.0,
        )

    client = LoopyCutsClient(
        executable=
            executable,

        mesh_file=
            mesh,

        loop_file=
            loops,

        resource_guard_policy=
            ResourceGuardPolicyV1(),

        resource_guard_sample_interval_seconds=
            0.02,

        resource_snapshot_reader=
            reader,
    )

    try:
        try:
            client.step(
                0
            )

        except RLServerResourceAbort as exc:
            assert (
                exc.phase
                ==
                "STEP"
            )

            assert (
                exc.guard_state
                ==
                RESOURCE_STATE_ABORT_EMERGENCY
            )

            assert (
                exc.snapshot.swap_used_bytes
                >=
                12 * GIB
            )

        else:
            raise AssertionError(
                "Emergency ResourceGuard did not abort STEP"
            )

        # Critical semantic:
        # only child C++/fake server dies.
        # This Python test process is still running.
        assert (
            client.process.poll()
            is not None
        )

    finally:
        client.close()


def run_hard_hold_test(
    *,
    executable,
    mesh,
    loops,
):
    def reader(
        *,
        cpp_pid,
    ):
        return resource_snapshot(
            cpp_pid=
                cpp_pid,

            swap_used_gib=
                10.5,
        )

    # Shorten the hold only for the integration regression.
    #
    # The pure ResourceGuard test already freezes the production
    # semantic at exactly 8.0 seconds.
    policy = ResourceGuardPolicyV1(
        abort_hold_seconds=
            0.20
    )

    client = LoopyCutsClient(
        executable=
            executable,

        mesh_file=
            mesh,

        loop_file=
            loops,

        resource_guard_policy=
            policy,

        resource_guard_sample_interval_seconds=
            0.05,

        resource_snapshot_reader=
            reader,
    )

    try:
        try:
            client.step(
                0
            )

        except RLServerResourceAbort as exc:
            assert (
                exc.phase
                ==
                "STEP"
            )

            assert (
                exc.guard_state
                ==
                RESOURCE_STATE_ABORT
            )

            assert (
                exc.snapshot.swap_used_bytes
                >=
                10 * GIB
            )

        else:
            raise AssertionError(
                "Hard-hold ResourceGuard did not abort STEP"
            )

        assert (
            client.process.poll()
            is not None
        )

    finally:
        client.close()


def main():
    with tempfile.TemporaryDirectory(
        prefix=
            "loopycuts_cpp_guard_"
    ) as tmp:
        root = Path(
            tmp
        )

        executable = (
            write_fake_server(
                root
            )
        )

        mesh = (
            root
            /
            "dummy.obj"
        )

        loops = (
            root
            /
            "dummy_loop.txt"
        )

        mesh.write_text(
            "# dummy\n",
            encoding="utf-8",
        )

        loops.write_text(
            "# dummy\n",
            encoding="utf-8",
        )

        run_emergency_test(
            executable=
                executable,

            mesh=
                mesh,

            loops=
                loops,
        )

        run_hard_hold_test(
            executable=
                executable,

            mesh=
                mesh,

            loops=
                loops,
        )

    print(
        "PASS: emergency >=12 GiB kills current C++ STEP only"
    )

    print(
        "PASS: sustained >=10 GiB hard threshold kills current C++ STEP"
    )

    print(
        "PASS: Python process survives ResourceGuard abort"
    )

    print(
        "PASS: ResourceGuard SIGKILL is distinguished from unexpected C++ crash"
    )


if __name__ == "__main__":
    main()
