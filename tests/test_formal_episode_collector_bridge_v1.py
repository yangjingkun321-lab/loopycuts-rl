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

    ProcessMemorySnapshot,
    ResourceGuardPolicyV1,
    ResourceSnapshot,
)

from envs.final_reward_wrapper_v3 import (
    FinalRewardWrapperV3,
)

from envs.finalization_eval_wrapper import (
    FinalizationEvalWrapper,
)

from envs.formal_episode_collector_bridge_v1 import (
    FORMAL_EPISODE_COLLECTOR_BRIDGE_VERSION,
    FormalEpisodeCollectorBridgeV1,
)

from envs.loopycuts_env import (
    LoopyCutsEnv,
)


EXECUTABLE = (
    "/home/yjk/codes/LoopyCuts/"
    "volumetric_cutter/"
    "volumetric_cutter"
)

MESH = (
    "/home/yjk/loopycuts_inputs/"
    "cylinder_plate_clean/"
    "cylinder_plate_rem_splitted.obj"
)

LOOP_FILE = (
    "/home/yjk/loopycuts_inputs/"
    "cylinder_plate_clean/"
    "cylinder_plate_rem_loop.txt"
)


def emergency_snapshot_reader(
    *,
    cpp_pid,
):
    total = 34 * GIB
    used = 12 * GIB

    return ResourceSnapshot(
        mem_available_bytes=
            512 * 1024 * 1024,

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
                    int(
                        cpp_pid
                    ),

                rss_bytes=
                    4 * GIB,

                swap_bytes=
                    8 * GIB,
            ),
    )


def build_env(
    *,
    resource_abort,
):
    kwargs = {}

    if resource_abort:
        kwargs = {
            "resource_guard_policy":
                ResourceGuardPolicyV1(),

            "resource_guard_sample_interval_seconds":
                0.02,

            "resource_snapshot_reader":
                emergency_snapshot_reader,
        }

    base = LoopyCutsEnv(
        executable=
            EXECUTABLE,

        mesh_file=
            MESH,

        loop_file=
            LOOP_FILE,

        echo_logs=
            False,

        **kwargs,
    )

    env = (
        FormalEpisodeCollectorBridgeV1(
            FinalRewardWrapperV3(
                FinalizationEvalWrapper(
                    base
                )
            )
        )
    )

    return (
        base,
        env,
    )


def consume_suppressed_reset(
    *,
    base,
    env,
    initial_pid,
):
    observation, info = (
        env.reset()
    )

    assert (
        env.suppressed_reset_count
        ==
        1
    )

    assert (
        env.suppress_next_reset
        is False
    )

    assert (
        env.after_suppressed_reset
        is True
    )

    assert (
        info[
            "formal_episode_collector_autoreset_suppressed"
        ]
        is True
    )

    assert int(
        observation[
            "mask"
        ].sum()
    ) > 0

    # Critical:
    # no replacement C++ server was started.
    assert (
        base.client.process.pid
        ==
        initial_pid
    )

    assert (
        base.client.process.poll()
        is not None
    )

    try:
        env.step(
            0
        )

    except RuntimeError as exc:
        assert (
            "current collector must be closed"
            in
            str(
                exc
            )
        )

    else:
        raise AssertionError(
            "Bridge allowed STEP after suppressed autoreset"
        )


def test_resource_abort_terminal():
    base, env = build_env(
        resource_abort=True
    )

    try:
        observation, info = (
            env.reset(
                seed=123
            )
        )

        initial_pid = (
            base.client.process.pid
        )

        (
            terminal_obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            0
        )

        assert terminated is True
        assert truncated is False
        assert float(reward) == -4.0

        assert (
            info[
                "finalization_outcome"
            ][
                "outcome"
            ]
            ==
            "RESOURCE_ABORT"
        )

        assert (
            env.suppress_next_reset
            is True
        )

        assert (
            base.client.process.poll()
            is not None
        )

        consume_suppressed_reset(
            base=
                base,

            env=
                env,

            initial_pid=
                initial_pid,
        )

    finally:
        env.close()


def test_normal_terminal():
    base, env = build_env(
        resource_abort=False
    )

    try:
        observation, info = (
            env.reset(
                seed=456
            )
        )

        initial_pid = (
            base.client.process.pid
        )

        for action in [
            0,
            1,
            2,
            3,
        ]:
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

        assert terminated is True
        assert truncated is False

        assert (
            info[
                "finalization_outcome"
            ][
                "outcome"
            ]
            in {
                "FULL_HEX",
                "NON_FULL_HEX",
                "FINALIZATION_CRASH",
            }
        )

        assert (
            info[
                "resource_guard"
            ][
                "triggered"
            ]
            is False
        )

        assert (
            env.suppress_next_reset
            is True
        )

        # The original C++ process may already have exited naturally
        # or be closed by finalization semantics.  The critical check
        # is that the suppressed reset does not create a replacement.
        if base.client.process.poll() is None:
            base.client.process.kill()
            base.client.process.wait()

        consume_suppressed_reset(
            base=
                base,

            env=
                env,

            initial_pid=
                initial_pid,
        )

    finally:
        env.close()


def main():
    assert (
        FORMAL_EPISODE_COLLECTOR_BRIDGE_VERSION
        ==
        "loopycuts_formal_episode_collector_bridge_v1"
    )

    test_resource_abort_terminal()

    print(
        "PASS: RESOURCE_ABORT terminal autoreset is suppressed"
    )

    test_normal_terminal()

    print(
        "PASS: ordinary terminal autoreset is also suppressed"
    )

    print(
        "PASS: no terminal outcome launches an unused replacement C++ process"
    )

    print(
        "PASS: suppressed autoreset cannot be used for another action"
    )


if __name__ == "__main__":
    main()
