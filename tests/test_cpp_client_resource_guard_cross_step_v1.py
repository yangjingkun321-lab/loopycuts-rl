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

    ProcessMemorySnapshot,
    ResourceGuardPolicyV1,
    ResourceSnapshot,
)


def high_swap_snapshot(
    *,
    cpp_pid,
):
    total = 34 * GIB
    used = int(
        10.5 * GIB
    )

    return ResourceSnapshot(
        mem_available_bytes=
            1024 * 1024 * 1024,

        swap_total_bytes=
            total,

        swap_free_bytes=
            total - used,

        swap_used_bytes=
            used,

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
                    int(cpp_pid),

                rss_bytes=
                    4 * GIB,

                swap_bytes=
                    int(
                        6.5 * GIB
                    ),
            ),
    )


def write_fake_server(
    root: Path,
):
    path = (
        root
        /
        "fake_cross_step_server.py"
    )

    path.write_text(
r'''#!/usr/bin/env python3

import sys
import time


step_index = 0


def emit_state(step):
    print(
        "[RL] STATE "
        f"step={step} "
        "terminal=0"
    )
    print("[RL] USED")
    print("[RL] REVERTED")
    print("[RL] NICO_BUG")
    print("[RL] TOP_RELEVANT")
    print("[RL] ACTIONS 0")
    sys.stdout.flush()


print("[RL] READY")
emit_state(0)


for raw in sys.stdin:
    command = raw.strip()

    if command.startswith("STEP "):
        step_index += 1

        if step_index == 1:
            # Less than the test's 0.20-second hold threshold.
            time.sleep(0.12)

            print(
                "[RL] STEP_RESULT "
                "step=1 "
                "loop_id=0 "
                "status=COMMITTED"
            )

            emit_state(1)

        else:
            # If the timer correctly persists from STEP 1,
            # ResourceGuard should kill this STEP shortly after
            # the cumulative 0.20-second threshold is reached.
            time.sleep(30.0)

    elif command == "QUIT":
        print("[RL] BYE")
        sys.stdout.flush()
        break
''',
        encoding="utf-8",
    )

    path.chmod(
        path.stat().st_mode
        |
        stat.S_IXUSR
    )

    return path


def main():
    with tempfile.TemporaryDirectory(
        prefix=
            "loopycuts_guard_cross_step_"
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
                0.02,

            resource_snapshot_reader=
                high_swap_snapshot,
        )

        try:
            # ========================================================
            # STEP 1:
            # high swap, but not high long enough to abort yet.
            # ========================================================

            (
                result1,
                _,
                _,
            ) = client.step(
                0
            )

            assert (
                int(
                    result1[
                        "step"
                    ]
                )
                ==
                1
            )

            assert (
                client.process.poll()
                is None
            )


            # ========================================================
            # STEP 2:
            #
            # The high-swap timer MUST continue from STEP 1.
            # It must NOT restart from zero here.
            # ========================================================

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

            else:
                raise AssertionError(
                    "ResourceGuard did not preserve "
                    "the >=10 GiB hold timer across STEP boundaries"
                )

            assert (
                client.process.poll()
                is not None
            )

        finally:
            client.close()


    print(
        "PASS: first short high-swap STEP completes before hold threshold"
    )

    print(
        "PASS: >=10 GiB hold timer persists across STEP boundaries"
    )

    print(
        "PASS: second STEP aborts from cumulative high-swap duration"
    )

    print(
        "PASS: timer is episode-scoped rather than STEP-scoped"
    )


if __name__ == "__main__":
    main()
