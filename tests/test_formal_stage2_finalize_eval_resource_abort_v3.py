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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from bridge.cpp_client import (
    FINALIZE_EVAL_SWAP_CAP_GUARD_STATE,
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
    PROJECT_STAGE2_RESOURCE_GUARD_FINALIZE_EVAL_SWAP_ABORT_GIB,
)


FINALIZE_SWAP_USED = (
    25
    *
    GIB
)


def finalize_cap_snapshot_reader(
    *,
    cpp_pid,
):
    total = (
        34
        *
        GIB
    )

    return ResourceSnapshot(
        mem_available_bytes=
            2 * GIB,

        swap_total_bytes=
            total,

        swap_free_bytes=
            total - FINALIZE_SWAP_USED,

        swap_used_bytes=
            FINALIZE_SWAP_USED,

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
                    int(cpp_pid),

                rss_bytes=
                    5 * GIB,

                swap_bytes=
                    12 * GIB,
            ),
    )


def main():
    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_FINALIZE_EVAL_SWAP_ABORT_GIB
        ==
        25
    )

    core = (
        prepare_formal_training_core(
            seed=42
        )
    )

    # Infrastructure regression only:
    # do not repeat all 782 Stage-I gradient updates.
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
        "loopycuts_formal_stage2_online_v3_resource_guard"
    )

    plate3 = next(
        model
        for model in state.models
        if model.model == "Plate3"
    )


    # ------------------------------------------------------------
    # Test-only STEP policy.
    #
    # Fake system SwapUsed is always 25 GiB.
    #
    # Therefore:
    #
    # STEP:
    #     25 < 30 / 31 / 32 GiB
    #     -> no STEP ResourceGuard abort
    #
    # FINALIZE_EVAL:
    #     25 >= dedicated 25-GiB cap
    #     -> RESOURCE_ABORT
    # ------------------------------------------------------------

    step_policy = (
        ResourceGuardPolicyV1(
            warning_swap_used_bytes=
                30 * GIB,

            abort_swap_used_bytes=
                31 * GIB,

            emergency_swap_used_bytes=
                32 * GIB,

            rearm_swap_used_bytes=
                6 * GIB,

            abort_hold_seconds=
                8.0,
        )
    )


    assert len(
        state.expo_buffer
    ) == 0

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
                step_policy,

            resource_guard_sample_interval_seconds=
                0.02,

            resource_snapshot_reader=
                finalize_cap_snapshot_reader,

            finalize_eval_swap_abort_bytes=
                25 * GIB,
        )
    )


    print()
    print("=" * 100)
    print(
        "FORMAL STAGE-II FINALIZE_EVAL RESOURCE_ABORT"
    )
    print("=" * 100)

    print(
        "model                  :",
        record["model"],
    )

    print(
        "steps                  :",
        record["steps"],
    )

    print(
        "actions                :",
        record["actions"],
    )

    print(
        "outcome                :",
        record["finalization_outcome"],
    )

    print(
        "resource phase         :",
        record["resource_guard_phase"],
    )

    print(
        "guard state            :",
        record["resource_guard_state"],
    )

    print(
        "D_expo size            :",
        len(state.expo_buffer),
    )

    print(
        "environment steps      :",
        state.total_environment_steps,
    )

    print(
        "gradient updates       :",
        state.total_gradient_updates,
    )


    # ============================================================
    # Plate3 still executes the two genuine Stage-II actions.
    # ============================================================

    assert (
        record["actions"]
        ==
        [1, 0]
    )

    assert (
        record["steps"]
        ==
        2
    )

    assert (
        record["terminated"]
        is True
    )

    assert (
        record["truncated"]
        is False
    )

    assert (
        record["completed"]
        is True
    )

    assert (
        record["resource_abort"]
        is True
    )

    assert (
        record["finalization_outcome"]
        ==
        "RESOURCE_ABORT"
    )

    assert (
        record["resource_guard_phase"]
        ==
        "FINALIZE_EVAL"
    )

    assert (
        record["resource_guard_state"]
        ==
        FINALIZE_EVAL_SWAP_CAP_GUARD_STATE
    )

    assert (
        record[
            "resource_guard_swap_used_bytes"
        ]
        ==
        25 * GIB
    )


    # ============================================================
    # CRITICAL:
    #
    # N real actions == N replay transitions
    #                == N environment steps
    #                == N SAC updates
    #
    # FINALIZE_EVAL does NOT create transition N+1.
    # ============================================================

    assert (
        len(state.expo_buffer)
        ==
        2
    )

    assert (
        state.total_environment_steps
        ==
        2
    )

    assert (
        state.total_gradient_updates
        ==
        2
    )

    assert (
        record["gradient_updates"]
        ==
        2
    )

    assert (
        record["expo_buffer_size"]
        ==
        2
    )


    # ============================================================
    # First transition is ordinary.
    # ============================================================

    first = (
        state.expo_buffer[
            [0]
        ]
    )

    assert bool(
        first.terminated[0]
    ) is False

    assert bool(
        first
        .info
        .resource_guard
        .triggered[0]
    ) is False


    # ============================================================
    # Second transition is the genuine terminal STEP whose
    # FINALIZE_EVAL outcome was replaced by RESOURCE_ABORT.
    # ============================================================

    last = (
        state.expo_buffer[
            [1]
        ]
    )

    assert (
        int(last.act[0])
        ==
        0
    )

    assert bool(
        last.terminated[0]
    ) is True

    assert bool(
        last.truncated[0]
    ) is False

    assert math.isclose(
        float(last.rew[0]),
        -4.0,
        rel_tol=0.0,
        abs_tol=0.0,
    )

    assert (
        str(
            last
            .info
            .finalization_outcome
            .outcome[0]
        )
        ==
        "RESOURCE_ABORT"
    )

    assert bool(
        last
        .info
        .resource_guard
        .triggered[0]
    ) is True

    assert (
        str(
            last
            .info
            .resource_guard
            .phase[0]
        )
        ==
        "FINALIZE_EVAL"
    )

    assert (
        str(
            last
            .info
            .resource_guard
            .guard_state[0]
        )
        ==
        FINALIZE_EVAL_SWAP_CAP_GUARD_STATE
    )

    assert (
        int(
            last
            .info
            .resource_guard
            .swap_used_bytes[0]
        )
        ==
        25 * GIB
    )


    # The real final STEP completed before finalization failed,
    # therefore its selection transition metrics must still exist.
    assert hasattr(
        last.info,
        "transition_metrics",
    )


    # ============================================================
    # Episode accounting/history.
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
        len(state.history)
        ==
        1
    )

    assert (
        state.history[0]
        ==
        record
    )

    final_stats = (
        record[
            "final_training_stats"
        ]
    )

    assert final_stats is not None

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
        "PASS: Plate3 executes exactly two genuine Stage-II actions"
    )

    print(
        "PASS: FINALIZE_EVAL RESOURCE_ABORT enters D_expo on the existing final STEP"
    )

    print(
        "PASS: FINALIZE_EVAL RESOURCE_ABORT adds no third replay transition"
    )

    print(
        "PASS: D_expo/env_steps/gradient_updates remain exactly 2/2/2"
    )

    print(
        "PASS: final replay reward is exactly -4"
    )

    print(
        "PASS: ResourceGuard phase is preserved as FINALIZE_EVAL"
    )


if __name__ == "__main__":
    main()
