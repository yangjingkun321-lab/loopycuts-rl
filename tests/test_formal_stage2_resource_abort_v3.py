from __future__ import annotations

import math
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

from training.formal_training_v1 import (
    FORMAL_STAGE2_ONLINE_VERSION,

    collect_formal_stage2_model_episode,
    enter_formal_stage2,
    prepare_formal_stage2_state,
    prepare_formal_training_core,
)

from training.protocol_v1 import (
    PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS,
    PROJECT_STAGE1_GRADIENT_STEPS,
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
                    768 * 1024 * 1024,

                swap_bytes=
                    256 * 1024 * 1024,
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
    core = (
        prepare_formal_training_core(
            seed=42
        )
    )

    # Infrastructure regression:
    # do not repeat all 782 Stage-I updates.
    core.stage1_updates_completed = (
        PROJECT_STAGE1_GRADIENT_STEPS
    )

    core.stage1_sampled_demo_transitions = (
        PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS
    )

    enter_formal_stage2(
        core
    )

    state = (
        prepare_formal_stage2_state(
            core
        )
    )

    assert (
        FORMAL_STAGE2_ONLINE_VERSION
        ==
        "loopycuts_formal_stage2_online_v5_quality_aware"
    )

    plate3 = next(
        model
        for model in state.models
        if model.model == "Plate3"
    )

    assert (
        len(
            state.expo_buffer
        )
        ==
        0
    )

    assert (
        state.total_environment_steps
        ==
        0
    )

    assert (
        state.total_gradient_updates
        ==
        0
    )


    record = (
        collect_formal_stage2_model_episode(
            core,
            state,

            model=
                plate3,

            resource_guard_policy=
                ResourceGuardPolicyV1(),

            resource_guard_sample_interval_seconds=
                0.02,

            resource_snapshot_reader=
                emergency_snapshot_reader,
        )
    )


    print()
    print("=" * 96)
    print("FORMAL STAGE-II RESOURCE_ABORT")
    print("=" * 96)

    print(
        "model                  :",
        record[
            "model"
        ],
    )

    print(
        "steps                  :",
        record[
            "steps"
        ],
    )

    print(
        "actions                :",
        record[
            "actions"
        ],
    )

    print(
        "outcome                :",
        record[
            "finalization_outcome"
        ],
    )

    print(
        "episode return         :",
        record[
            "episode_return"
        ],
    )

    print(
        "D_expo size            :",
        len(
            state.expo_buffer
        ),
    )

    print(
        "environment steps      :",
        state.total_environment_steps,
    )

    print(
        "gradient updates       :",
        state.total_gradient_updates,
    )

    print(
        "guard state            :",
        record[
            "resource_guard_state"
        ],
    )

    print(
        "SwapUsed GiB           :",
        record[
            "resource_guard_swap_used_bytes"
        ]
        /
        GIB,
    )


    # ============================================================
    # One REAL attempted action = one REAL terminal transition.
    # ============================================================

    assert (
        record[
            "steps"
        ]
        ==
        1
    )

    assert (
        len(
            record[
                "actions"
            ]
        )
        ==
        1
    )

    assert (
        record[
            "terminated"
        ]
        is True
    )

    assert (
        record[
            "truncated"
        ]
        is False
    )

    assert (
        record[
            "completed"
        ]
        is True
    )

    assert (
        record[
            "resource_abort"
        ]
        is True
    )

    assert (
        record[
            "finalization_outcome"
        ]
        ==
        "RESOURCE_ABORT"
    )

    assert math.isclose(
        float(
            record[
                "episode_return"
            ]
        ),
        -4.0,
        rel_tol=0.0,
        abs_tol=0.0,
    )


    # ============================================================
    # Reward V5 episode-level telemetry.
    #
    # STEP RESOURCE_ABORT has no completed post-STEP geometry and
    # therefore no terminal quality. Reward remains the exact -4
    # override with every dense component zero.
    # ============================================================

    terminal_quality = (
        record[
            "terminal_quality"
        ]
    )

    terminal_reward_v5 = (
        record[
            "terminal_reward_v5"
        ]
    )

    assert (
        terminal_quality[
            "available"
        ]
        is False
    )

    assert (
        terminal_reward_v5[
            "available"
        ]
        is True
    )

    assert (
        terminal_reward_v5[
            "quality_available"
        ]
        is False
    )

    assert (
        terminal_reward_v5[
            "step"
        ]
        ==
        0.0
    )

    assert (
        terminal_reward_v5[
            "tet_growth"
        ]
        ==
        0.0
    )

    assert (
        terminal_reward_v5[
            "revert"
        ]
        ==
        0.0
    )

    assert (
        terminal_reward_v5[
            "convergence"
        ]
        ==
        0.0
    )

    assert (
        terminal_reward_v5[
            "utility"
        ]
        ==
        0.0
    )

    assert (
        terminal_reward_v5[
            "terminal"
        ]
        ==
        -4.0
    )

    assert (
        terminal_reward_v5[
            "total"
        ]
        ==
        -4.0
    )


    # ============================================================
    # Critical 1 transition : 1 env step : 1 update invariant.
    # ============================================================

    assert (
        len(
            state.expo_buffer
        )
        ==
        1
    )

    assert (
        state.total_environment_steps
        ==
        1
    )

    assert (
        state.total_gradient_updates
        ==
        1
    )

    assert (
        record[
            "gradient_updates"
        ]
        ==
        1
    )

    assert (
        record[
            "expo_buffer_size"
        ]
        ==
        1
    )


    # ============================================================
    # Inspect the actual replay transition.
    # ============================================================

    transition = (
        state.expo_buffer[
            [0]
        ]
    )

    assert math.isclose(
        float(
            transition.rew[
                0
            ]
        ),
        -4.0,
        rel_tol=0.0,
        abs_tol=0.0,
    )

    assert bool(
        transition.terminated[
            0
        ]
    ) is True

    assert bool(
        transition.truncated[
            0
        ]
    ) is False

    assert (
        str(
            transition
            .info
            .finalization_outcome
            .outcome[
                0
            ]
        )
        ==
        "RESOURCE_ABORT"
    )

    assert bool(
        transition
        .info
        .finalization_outcome
        .attempted[
            0
        ]
    ) is False

    assert bool(
        transition
        .info
        .terminal_quality
        .available[
            0
        ]
    ) is False

    assert bool(
        transition
        .info
        .reward_v5_breakdown
        .quality_available[
            0
        ]
    ) is False

    assert (
        float(
            transition
            .info
            .reward_v5_breakdown
            .utility[
                0
            ]
        )
        ==
        0.0
    )

    assert (
        float(
            transition
            .info
            .reward_v5_breakdown
            .terminal[
                0
            ]
        )
        ==
        -4.0
    )

    assert (
        float(
            transition
            .info
            .reward_v5_breakdown
            .total[
                0
            ]
        )
        ==
        -4.0
    )

    assert not hasattr(
        transition.info,
        "transition_metrics",
    )

    assert bool(
        transition
        .info
        .resource_guard
        .triggered[
            0
        ]
    ) is True

    assert (
        str(
            transition
            .info
            .resource_guard
            .phase[
                0
            ]
        )
        ==
        "STEP"
    )

    assert (
        int(
            transition
            .info
            .resource_guard
            .swap_used_bytes[
                0
            ]
        )
        ==
        12 * GIB
    )


    # ============================================================
    # Telemetry survives into episode history.
    # ============================================================

    assert (
        record[
            "resource_guard_phase"
        ]
        ==
        "STEP"
    )

    assert (
        record[
            "resource_guard_state"
        ]
        ==
        "RESOURCE_ABORT_EMERGENCY"
    )

    assert (
        record[
            "resource_guard_swap_used_bytes"
        ]
        ==
        12 * GIB
    )

    assert (
        record[
            "resource_guard_python_rss_bytes"
        ]
        ==
        768 * 1024 * 1024
    )

    assert (
        record[
            "resource_guard_python_swap_bytes"
        ]
        ==
        256 * 1024 * 1024
    )

    assert (
        record[
            "resource_guard_cpp_rss_bytes"
        ]
        ==
        4 * GIB
    )

    assert (
        record[
            "resource_guard_cpp_swap_bytes"
        ]
        ==
        8 * GIB
    )


    # ============================================================
    # Episode accounting.
    # ============================================================

    assert (
        state.episode_attempts
        ==
        1
    )

    assert (
        state.completed_episodes
        ==
        1
    )

    assert (
        len(
            state.history
        )
        ==
        1
    )

    assert (
        state.history[
            0
        ]
        ==
        record
    )


    final_stats = (
        record[
            "final_training_stats"
        ]
    )

    assert (
        final_stats
        is not None
    )

    assert (
        float(
            final_stats[
                "bc_loss"
            ]
        )
        ==
        0.0
    )


    print()
    print(
        "PASS: RESOURCE_ABORT attempted action enters D_expo exactly once"
    )

    print(
        "PASS: RESOURCE_ABORT increments environment_steps exactly once"
    )

    print(
        "PASS: RESOURCE_ABORT triggers exactly one Stage-II SAC update"
    )

    print(
        "PASS: RESOURCE_ABORT reward is exactly -4 in formal replay"
    )

    print(
        "PASS: ResourceGuard telemetry survives formal episode accounting"
    )

    print(
        "PASS: Python formal trainer survives and completes episode accounting"
    )


if __name__ == "__main__":
    main()
