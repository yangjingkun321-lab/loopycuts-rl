from __future__ import annotations

import os
import signal
import stat
import sys
import tempfile

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from bridge.cpp_client import (
    CPP_LEGACY_RSS_ASSERT_GUARD_STATE,
    LoopyCutsClient,
    RLServerProcessError,
    RLServerResourceAbort,
)

from bridge.resource_guard_v1 import (
    GIB,
    ProcessMemorySnapshot,
    ResourceGuardPolicyV1,
    ResourceSnapshot,
)


def resource_snapshot(
    *,
    cpp_pid,
):
    swap_total = (
        34
        *
        GIB
    )

    swap_used = (
        1
        *
        GIB
    )

    return ResourceSnapshot(
        mem_available_bytes=
            8 * GIB,

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

        # Deliberately below the C++ assertion threshold.
        #
        # This proves that Python is NOT proactively applying a new
        # 10-GiB RSS watchdog.  It only normalizes the C++ process
        # after the known assertion actually occurs.
        cpp_memory=
            ProcessMemorySnapshot(
                pid=
                    int(cpp_pid),

                rss_bytes=
                    int(
                        9.5 * GIB
                    ),

                swap_bytes=
                    0,
            ),
    )


def write_fake_server(
    root: Path,
    *,
    name: str,
    assertion_line: str,
):
    path = (
        root
        /
        name
    )

    assertion_literal = repr(
        assertion_line
    )

    script = f'''#!/usr/bin/env python3

import os
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
        # Give the ResourceGuard monitor time to take a NORMAL
        # low-swap sample. It must NOT terminate this process.
        time.sleep(0.15)

        print({assertion_literal})
        sys.stdout.flush()

        # Reproduce real C++ assert()/abort() semantics.
        os.abort()

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
'''

    path.write_text(
        script,
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


def build_client(
    *,
    executable,
    mesh,
    loops,
):
    return LoopyCutsClient(
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
            resource_snapshot,
    )


def run_known_memory_assert_test(
    *,
    root,
    mesh,
    loops,
):
    executable = write_fake_server(
        root,

        name=
            "fake_known_rss_assert.py",

        assertion_line=(
            "volumetric_cutter: cut.cpp:393: "
            "void cut(GlobalState&, uint): "
            "Assertion "
            "`memory_usage_in_giga_bytes()<10' failed."
        ),
    )

    client = build_client(
        executable=
            executable,

        mesh=
            mesh,

        loops=
            loops,
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
                CPP_LEGACY_RSS_ASSERT_GUARD_STATE
            )

            assert (
                exc.return_code
                ==
                -int(
                    signal.SIGABRT
                )
            )

            # External Swap guard was nowhere near its threshold.
            assert (
                exc.snapshot.swap_used_bytes
                ==
                1 * GIB
            )

        else:
            raise AssertionError(
                "Known LoopyCuts RSS assertion was not "
                "normalized to RESOURCE_ABORT"
            )

    finally:
        client.close()


def run_unknown_assert_test(
    *,
    root,
    mesh,
    loops,
):
    executable = write_fake_server(
        root,

        name=
            "fake_unknown_assert.py",

        assertion_line=(
            "volumetric_cutter: some_file.cpp:123: "
            "Assertion `some_geometry_invariant' failed."
        ),
    )

    client = build_client(
        executable=
            executable,

        mesh=
            mesh,

        loops=
            loops,
    )

    try:
        try:
            client.step(
                0
            )

        except RLServerProcessError as exc:
            assert (
                exc.phase
                ==
                "STEP"
            )

            assert (
                exc.return_code
                ==
                -int(
                    signal.SIGABRT
                )
            )

            assert (
                "some_geometry_invariant"
                in
                str(
                    exc
                )
            )

            assert (
                "C++ stdout/stderr tail"
                in
                str(
                    exc
                )
            )

        except RLServerResourceAbort as exc:
            raise AssertionError(
                "Unknown C++ assertion was incorrectly "
                f"classified as RESOURCE_ABORT: {exc}"
            ) from exc

        else:
            raise AssertionError(
                "Unknown C++ assertion did not fail closed"
            )

    finally:
        client.close()


def main():
    with tempfile.TemporaryDirectory(
        prefix=
            "loopycuts_cpp_rss_abort_v4_"
    ) as tmp:
        root = Path(
            tmp
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

        run_known_memory_assert_test(
            root=
                root,

            mesh=
                mesh,

            loops=
                loops,
        )

        run_unknown_assert_test(
            root=
                root,

            mesh=
                mesh,

            loops=
                loops,
        )

    print(
        "PASS: known LoopyCuts 10-GiB RSS assertion "
        "becomes STEP RESOURCE_ABORT"
    )

    print(
        "PASS: low external SwapUsed does not proactively "
        "trigger the new compatibility path"
    )

    print(
        "PASS: unrelated C++ SIGABRT remains "
        "RLServerProcessError"
    )


if __name__ == "__main__":
    main()
