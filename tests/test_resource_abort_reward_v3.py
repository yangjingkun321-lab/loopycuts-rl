from __future__ import annotations

import math
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

from envs.final_reward_wrapper import (
    FinalRewardWrapper,
)

from envs.final_reward_wrapper_v3 import (
    FinalRewardWrapperV3,
)

from envs.finalization_eval_wrapper import (
    FinalizationEvalWrapper,
)

from envs.loopycuts_env import (
    LoopyCutsEnv,
)

from rewards.reward_v3 import (
    REWARD_V3_VERSION,
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


def build_normal_v2():
    return FinalRewardWrapper(
        FinalizationEvalWrapper(
            LoopyCutsEnv(
                executable=
                    EXECUTABLE,

                mesh_file=
                    MESH,

                loop_file=
                    LOOP_FILE,

                echo_logs=
                    False,
            )
        )
    )


def build_normal_v3():
    return FinalRewardWrapperV3(
        FinalizationEvalWrapper(
            LoopyCutsEnv(
                executable=
                    EXECUTABLE,

                mesh_file=
                    MESH,

                loop_file=
                    LOOP_FILE,

                echo_logs=
                    False,
            )
        )
    )


def build_abort_v3():
    return FinalRewardWrapperV3(
        FinalizationEvalWrapper(
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
    )


def test_normal_v2_v3_compatibility():
    env_v2 = build_normal_v2()
    env_v3 = build_normal_v3()

    try:
        obs2, info2 = env_v2.reset(
            seed=123
        )

        obs3, info3 = env_v3.reset(
            seed=123
        )

        assert (
            info2[
                "reward_version"
            ]
            ==
            "final_v2"
        )

        assert (
            info3[
                "reward_version"
            ]
            ==
            REWARD_V3_VERSION
        )

        actions = [
            0,
            1,
            2,
            3,
        ]

        for action in actions:
            (
                next2,
                reward2,
                term2,
                trunc2,
                info2,
            ) = env_v2.step(
                action
            )

            (
                next3,
                reward3,
                term3,
                trunc3,
                info3,
            ) = env_v3.step(
                action
            )

            assert math.isclose(
                float(
                    reward2
                ),
                float(
                    reward3
                ),
                rel_tol=0.0,
                abs_tol=1e-15,
            )

            assert (
                term2
                ==
                term3
            )

            assert (
                trunc2
                is False
            )

            assert (
                trunc3
                is False
            )

            assert (
                info3[
                    "reward_version"
                ]
                ==
                REWARD_V3_VERSION
            )

            assert (
                info3[
                    "selection_reward_available"
                ]
                is True
            )

            assert (
                info3[
                    "resource_guard"
                ][
                    "triggered"
                ]
                is False
            )

            obs2 = next2
            obs3 = next3

    finally:
        env_v2.close()
        env_v3.close()


def test_resource_abort_reward():
    env = build_abort_v3()

    try:
        observation, info = env.reset(
            seed=456
        )

        assert (
            info[
                "reward_version"
            ]
            ==
            REWARD_V3_VERSION
        )

        (
            obs_next,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            0
        )

        assert (
            terminated
            is True
        )

        assert (
            truncated
            is False
        )

        assert math.isclose(
            float(
                reward
            ),
            -4.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )

        assert (
            info[
                "reward_version"
            ]
            ==
            REWARD_V3_VERSION
        )

        assert (
            info[
                "selection_reward_available"
            ]
            is False
        )

        assert math.isclose(
            float(
                info[
                    "reward_v3_breakdown"
                ][
                    "step"
                ]
            ),
            0.0,
            abs_tol=0.0,
        )

        assert math.isclose(
            float(
                info[
                    "reward_v3_breakdown"
                ][
                    "tet_growth"
                ]
            ),
            0.0,
            abs_tol=0.0,
        )

        assert math.isclose(
            float(
                info[
                    "reward_v3_breakdown"
                ][
                    "revert"
                ]
            ),
            0.0,
            abs_tol=0.0,
        )

        assert math.isclose(
            float(
                info[
                    "reward_v3_breakdown"
                ][
                    "convergence"
                ]
            ),
            0.0,
            abs_tol=0.0,
        )

        assert math.isclose(
            float(
                info[
                    "reward_v3_breakdown"
                ][
                    "terminal"
                ]
            ),
            -4.0,
            abs_tol=0.0,
        )

        assert math.isclose(
            float(
                info[
                    "reward_v3_breakdown"
                ][
                    "total"
                ]
            ),
            -4.0,
            abs_tol=0.0,
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
                "outcome_code"
            ]
            ==
            4
        )

        assert (
            info[
                "finalization_outcome"
            ][
                "attempted"
            ]
            is False
        )

        assert (
            info[
                "resource_guard"
            ][
                "triggered"
            ]
            is True
        )

        assert (
            info[
                "resource_guard"
            ][
                "phase"
            ]
            ==
            "STEP"
        )

        assert (
            info[
                "resource_guard"
            ][
                "action"
            ]
            ==
            0
        )

        assert (
            info[
                "resource_guard"
            ][
                "swap_used_bytes"
            ]
            ==
            12 * GIB
        )

        assert int(
            obs_next[
                "mask"
            ].sum()
        ) == 0

        # Rich sparse record must not reach Tianshou.
        assert (
            "resource_abort"
            not in info
        )

    finally:
        env.close()


def main():
    test_normal_v2_v3_compatibility()

    print(
        "PASS: Reward V3 is bitwise/numerically compatible "
        "with Reward V2 on ordinary Cylinder transitions"
    )

    test_resource_abort_reward()

    print(
        "PASS: RESOURCE_ABORT receives exactly -4"
    )

    print(
        "PASS: RESOURCE_ABORT has zero fabricated dense geometry reward"
    )

    print(
        "PASS: RESOURCE_ABORT outcome_code=4 and skips finalization"
    )

    print(
        "PASS: resource telemetry uses fixed collector schema"
    )


if __name__ == "__main__":
    main()
