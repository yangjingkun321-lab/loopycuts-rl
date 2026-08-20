from __future__ import annotations

import math
import sys

from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from bridge.cpp_client import (
    FINALIZE_EVAL_SWAP_CAP_GUARD_STATE,
)

from bridge.resource_guard_v1 import (
    GIB,
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


EXECUTABLE = Path(
    "/home/yjk/codes/LoopyCuts/"
    "volumetric_cutter/"
    "volumetric_cutter"
)

MESH = Path(
    "/home/yjk/codes/LoopyCuts/"
    "test_data/Plate3/"
    "5_rem_rem_splitted.obj"
)

LOOPS = Path(
    "/home/yjk/codes/LoopyCuts/"
    "test_data/Plate3/"
    "5_rem_rem_loop.txt"
)

FINALIZE_CAP_BYTES = (
    25
    *
    GIB
)


def fake_snapshot(
    swap_used_gib: float,
):
    swap_used_bytes = int(
        float(
            swap_used_gib
        )
        *
        GIB
    )

    swap_total_bytes = (
        34
        *
        GIB
    )

    process_memory = SimpleNamespace(
        rss_bytes=
            128
            *
            1024
            *
            1024,

        swap_bytes=
            0,
    )

    return SimpleNamespace(
        mem_available_bytes=
            4
            *
            GIB,

        swap_total_bytes=
            swap_total_bytes,

        swap_free_bytes=
            (
                swap_total_bytes
                -
                swap_used_bytes
            ),

        swap_used_bytes=
            swap_used_bytes,

        python_memory=
            process_memory,

        cpp_memory=
            process_memory,
    )


def build_env(
    *,
    swap_used_gib: float,
):
    snapshot = fake_snapshot(
        swap_used_gib
    )

    def reader(
        *,
        cpp_pid=None,
    ):
        del cpp_pid
        return snapshot

    base = LoopyCutsEnv(
        executable=
            EXECUTABLE,

        mesh_file=
            MESH,

        loop_file=
            LOOPS,

        echo_logs=
            False,

        # STEP is deliberately unguarded in this test.
        # We are isolating the FINALIZE_EVAL fuse.
        resource_guard_policy=
            None,

        resource_guard_sample_interval_seconds=
            0.02,

        resource_snapshot_reader=
            reader,

        finalize_eval_swap_abort_bytes=
            FINALIZE_CAP_BYTES,
    )

    env = (
        FinalRewardWrapperV3(
            FinalizationEvalWrapper(
                base
            )
        )
    )

    return (
        env,
        base,
    )


# =====================================================================
# CASE 1
#
# SwapUsed = 24 GiB:
# below the FINALIZE_EVAL 25 GiB system-survival cap.
#
# FINALIZE_EVAL must be allowed to finish normally even though this is
# far above STEP's 10 / 12 GiB thresholds.
# =====================================================================

env, base = build_env(
    swap_used_gib=
        24.0
)

try:
    observation, info = (
        env.reset()
    )

    observation, reward_1, terminated, truncated, info = (
        env.step(
            1
        )
    )

    assert terminated is False
    assert truncated is False

    observation, reward_2, terminated, truncated, info = (
        env.step(
            0
        )
    )

    assert terminated is True
    assert truncated is False

    assert (
        info[
            "finalization_attempted"
        ]
        is True
    )

    assert (
        info[
            "finalization_outcome"
        ][
            "outcome"
        ]
        ==
        "FULL_HEX"
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
        base.executed_loop_ids
        ==
        {
            1,
            0,
        }
    )

finally:
    env.close()


print(
    "PASS: FINALIZE_EVAL continues normally at 24 GiB SwapUsed"
)


# =====================================================================
# CASE 2
#
# SwapUsed = 25 GiB:
# FINALIZE_EVAL hard system-survival fuse fires immediately.
#
# The second real Stage-2 action is already complete, therefore:
#
#     actions = [1, 0]
#     transitions = 2
#
# There must NOT be a synthetic third action/transition.
# =====================================================================

env, base = build_env(
    swap_used_gib=
        25.0
)

try:
    observation, info = (
        env.reset()
    )

    observation, reward_1, terminated, truncated, info = (
        env.step(
            1
        )
    )

    assert terminated is False
    assert truncated is False

    observation, reward_2, terminated, truncated, info = (
        env.step(
            0
        )
    )

    assert terminated is True
    assert truncated is False

    assert math.isclose(
        float(
            reward_2
        ),
        -4.0,
        rel_tol=0.0,
        abs_tol=0.0,
    )

    # The last STEP was real and completed.
    assert (
        "transition_metrics"
        in info
    )

    assert (
        info[
            "selection_reward_available"
        ]
        is True
    )

    assert (
        info[
            "finalization_attempted"
        ]
        is True
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

    guard = (
        info[
            "resource_guard"
        ]
    )

    assert (
        guard[
            "triggered"
        ]
        is True
    )

    assert (
        guard[
            "phase"
        ]
        ==
        "FINALIZE_EVAL"
    )

    assert (
        guard[
            "guard_state"
        ]
        ==
        FINALIZE_EVAL_SWAP_CAP_GUARD_STATE
    )

    assert (
        int(
            guard[
                "swap_used_bytes"
            ]
        )
        ==
        25
        *
        GIB
    )

    breakdown = (
        info[
            "reward_v3_breakdown"
        ]
    )

    assert math.isclose(
        float(
            breakdown[
                "step"
            ]
        ),
        0.0,
        rel_tol=0.0,
        abs_tol=0.0,
    )

    assert math.isclose(
        float(
            breakdown[
                "tet_growth"
            ]
        ),
        0.0,
        rel_tol=0.0,
        abs_tol=0.0,
    )

    assert math.isclose(
        float(
            breakdown[
                "revert"
            ]
        ),
        0.0,
        rel_tol=0.0,
        abs_tol=0.0,
    )

    assert math.isclose(
        float(
            breakdown[
                "convergence"
            ]
        ),
        0.0,
        rel_tol=0.0,
        abs_tol=0.0,
    )

    assert math.isclose(
        float(
            breakdown[
                "terminal"
            ]
        ),
        -4.0,
        rel_tol=0.0,
        abs_tol=0.0,
    )

    assert math.isclose(
        float(
            breakdown[
                "total"
            ]
        ),
        -4.0,
        rel_tol=0.0,
        abs_tol=0.0,
    )

    # Exactly the two real Stage-2 actions.
    assert (
        base.executed_loop_ids
        ==
        {
            1,
            0,
        }
    )

    # Resource fuse killed the current C++ process only.
    assert (
        base.client.process.poll()
        is not None
    )

finally:
    env.close()


print(
    "PASS: FINALIZE_EVAL aborts immediately at 25 GiB SwapUsed"
)

print(
    "PASS: FINALIZE_EVAL RESOURCE_ABORT reuses the genuine final STEP transition"
)

print(
    "PASS: FINALIZE_EVAL RESOURCE_ABORT reward is exactly -4 with zero dense shaping"
)

print(
    "PASS: no synthetic post-finalization action is fabricated"
)
