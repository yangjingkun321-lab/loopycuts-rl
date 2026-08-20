from __future__ import annotations

import os
import sys

from pathlib import Path


PROJECT_ROOT = Path(
    "/home/yjk/codes/loopycuts_rl"
)

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

from envs.finalization_eval_wrapper import (
    FinalizationEvalWrapper,
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
    total = (
        34
        *
        GIB
    )

    used = (
        12
        *
        GIB
    )

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


def main():
    base_env = (
        LoopyCutsEnv(
            executable=
                EXECUTABLE,

            mesh_file=
                MESH,

            loop_file=
                LOOP_FILE,

            echo_logs=
                False,

            resource_guard_policy=
                ResourceGuardPolicyV1(),

            resource_guard_sample_interval_seconds=
                0.02,

            resource_snapshot_reader=
                emergency_snapshot_reader,
        )
    )

    env = (
        FinalizationEvalWrapper(
            base_env
        )
    )

    try:
        observation, info = (
            env.reset(
                seed=123
            )
        )

        assert int(
            observation[
                "mask"
            ].sum()
        ) == 65

        action = 0

        (
            obs_next,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        # ============================================================
        # Genuine Gym terminal transition
        # ============================================================

        assert (
            terminated
            is True
        )

        assert (
            truncated
            is False
        )

        assert (
            float(
                reward
            )
            ==
            0.0
        )

        # Reward is intentionally still a placeholder at this phase.
        assert (
            info[
                "reward_is_placeholder"
            ]
            is True
        )

        assert (
            info[
                "reward_version"
            ]
            ==
            "selection_v1_resource_abort_placeholder"
        )

        # ============================================================
        # Terminal observation
        # ============================================================

        assert (
            base_env
            .observation_space
            .contains(
                obs_next
            )
        )

        assert int(
            obs_next[
                "mask"
            ].sum()
        ) == 0

        # The agent really attempted action 0.
        assert (
            action
            in
            base_env.executed_loop_ids
        )

        assert float(
            obs_next[
                "obs"
            ][
                "loops"
            ][
                action,
                11,
            ]
        ) == 1.0

        # ============================================================
        # Resource outcome
        # ============================================================

        assert (
            info[
                "resource_abort"
            ][
                "outcome"
            ]
            ==
            "RESOURCE_ABORT"
        )

        assert (
            info[
                "resource_abort"
            ][
                "action"
            ]
            ==
            action
        )

        assert (
            info[
                "resource_abort"
            ][
                "swap_used_bytes"
            ]
            ==
            12 * GIB
        )

        # ============================================================
        # CRITICAL:
        # FINALIZE_EVAL must NOT run after ResourceGuard killed C++.
        # ============================================================

        assert (
            info[
                "finalization_attempted"
            ]
            is False
        )

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
            info[
                "finalization_outcome"
            ][
                "completed"
            ]
            is False
        )

        assert (
            info[
                "finalization_outcome"
            ][
                "crashed"
            ]
            is False
        )

        # Child is dead, Python remains alive.
        assert (
            base_env.client.process.poll()
            is not None
        )

        # ============================================================
        # Reset after resource abort must create a fresh C++ server.
        # ============================================================

        observation2, info2 = (
            env.reset(
                seed=124
            )
        )

        assert int(
            observation2[
                "mask"
            ].sum()
        ) == 65

        assert (
            base_env.client.process.poll()
            is None
        )

        assert (
            base_env.executed_loop_ids
            ==
            set()
        )

    finally:
        env.close()

    print(
        "PASS: ResourceGuard STEP becomes one Gym terminal transition"
    )

    print(
        "PASS: RESOURCE_ABORT terminal observation has all-False mask"
    )

    print(
        "PASS: aborted action is preserved as the attempted action"
    )

    print(
        "PASS: RESOURCE_ABORT skips FINALIZE_EVAL"
    )

    print(
        "PASS: current C++ dies but Python/environment can reset"
    )


if __name__ == "__main__":
    main()
