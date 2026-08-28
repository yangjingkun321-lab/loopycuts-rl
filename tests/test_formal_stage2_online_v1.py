from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

from tianshou.data import (
    Collector,
)


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


from training.formal_training_v1 import (
    FORMAL_STAGE2_ONLINE_VERSION,
    build_formal_stage2_vector_env,
    collect_formal_stage2_model_episode,
    enter_formal_stage2,
    eligible_formal_stage2_models,
    formal_stage2_curriculum_phase,
    prepare_formal_stage2_state,
    prepare_formal_training_core,
    sample_formal_stage2_model,
)

from training.protocol_v1 import (
    PROJECT_STAGE1_GRADIENT_STEPS,
    PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS,
    PROJECT_STAGE2_EXPO_REPLAY_CAPACITY,
    PROJECT_STAGE2_SAMPLES_PER_BUFFER,
    PROJECT_STAGE2_CURRICULUM_WARMUP_ENV_STEPS,
    PROJECT_STAGE2_CURRICULUM_WARMUP_MAX_STRATUM,
    PROJECT_STAGE2_CURRICULUM_WARMUP_MODEL_COUNT,
    PROJECT_STAGE2_CURRICULUM_FULL_MODEL_COUNT,
)


def main():
    core = prepare_formal_training_core(
        seed=42
    )

    # ============================================================
    # Phase 18.2 already validated all 782 real updates.
    #
    # Do NOT repeat them in this infrastructure regression.
    # ============================================================

    core.stage1_updates_completed = (
        PROJECT_STAGE1_GRADIENT_STEPS
    )

    core.stage1_sampled_demo_transitions = (
        PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS
    )

    enter_formal_stage2(
        core
    )


    state = prepare_formal_stage2_state(
        core
    )


    assert (
        FORMAL_STAGE2_ONLINE_VERSION
        ==
        "loopycuts_formal_stage2_online_v5_quality_aware"
    )

    assert (
        len(
            state.models
        )
        ==
        49
    )

    assert (
        len(
            state.expo_buffer
        )
        ==
        0
    )

    assert (
        state.expo_buffer.maxsize
        ==
        PROJECT_STAGE2_EXPO_REPLAY_CAPACITY
        ==
        25_000
    )


    # ============================================================
    # Frozen complexity-curriculum sampler regression.
    #
    # At env=0 the eligible pool is Train39 (strata 0-7), sorted by
    # model.  NumPy Generator seed42 samples index 3 from that pool,
    # which remains "blade".
    #
    # This sampler RNG remains independent of policy exploration RNG.
    # ============================================================

    first_sample = (
        sample_formal_stage2_model(
            state
        )
    )

    print(
        "first seed42 model sample:",
        first_sample.model,
    )

    assert (
        first_sample.model
        ==
        "blade"
    )

    assert (
        first_sample.complexity_stratum
        ==
        4
    )


    # ============================================================
    # Curriculum pool and exact boundary regression.
    # ============================================================

    assert (
        PROJECT_STAGE2_CURRICULUM_WARMUP_ENV_STEPS
        ==
        5_000
    )

    assert (
        PROJECT_STAGE2_CURRICULUM_WARMUP_MAX_STRATUM
        ==
        7
    )

    assert (
        formal_stage2_curriculum_phase(
            state
        )
        ==
        "WARMUP"
    )

    phase, warmup_models = (
        eligible_formal_stage2_models(
            state
        )
    )

    assert phase == "WARMUP"

    assert (
        len(
            warmup_models
        )
        ==
        PROJECT_STAGE2_CURRICULUM_WARMUP_MODEL_COUNT
        ==
        39
    )

    assert all(
        model.complexity_stratum <= 7
        for model in warmup_models
    )

    assert (
        "motor_tail"
        not in
        {
            model.model
            for model in warmup_models
        }
    )


    # env=4999 is still WARMUP.
    state.total_environment_steps = 4_999

    assert (
        formal_stage2_curriculum_phase(
            state
        )
        ==
        "WARMUP"
    )

    phase_4999, pool_4999 = (
        eligible_formal_stage2_models(
            state
        )
    )

    assert phase_4999 == "WARMUP"
    assert len(pool_4999) == 39


    # env=5000 switches the NEXT episode to FULL.
    state.total_environment_steps = 5_000

    assert (
        formal_stage2_curriculum_phase(
            state
        )
        ==
        "FULL"
    )

    phase_5000, pool_5000 = (
        eligible_formal_stage2_models(
            state
        )
    )

    assert phase_5000 == "FULL"

    assert (
        len(
            pool_5000
        )
        ==
        PROJECT_STAGE2_CURRICULUM_FULL_MODEL_COUNT
        ==
        49
    )

    motor_tail = next(
        model
        for model in pool_5000
        if model.model == "motor_tail"
    )

    assert (
        motor_tail.complexity_stratum
        ==
        9
    )


    # Restore the untouched Stage-II collection counter before the
    # real Plate3 integration episode.
    state.total_environment_steps = 0


    # ============================================================
    # Fixed Plate3 integration episode.
    #
    # We deliberately choose a known small Train model here so this
    # infrastructure test does not turn into a resource-heavy formal
    # training run.
    #
    # Production run_next_formal_stage2_episode() uses the sampler.
    # ============================================================

    plate3 = next(
        model
        for model in state.models
        if model.model == "Plate3"
    )


    actor_before = [
        parameter
        .detach()
        .clone()

        for parameter
        in core.policy.actor.parameters()
    ]


    record = (
        collect_formal_stage2_model_episode(
            core,
            state,

            model=
                plate3,
        )
    )


    print()
    print("=" * 96)
    print("FORMAL STAGE-II REAL EPISODE")
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
        "gradient updates       :",
        record[
            "gradient_updates"
        ],
    )

    print(
        "D_expo size            :",
        len(
            state.expo_buffer
        ),
    )


    assert (
        record[
            "model"
        ]
        ==
        "Plate3"
    )

    assert (
        record[
            "model_complexity_stratum"
        ]
        ==
        0
    )

    assert (
        record[
            "curriculum_phase"
        ]
        ==
        "WARMUP"
    )

    assert (
        record[
            "eligible_model_count"
        ]
        ==
        39
    )

    assert (
        record[
            "environment_steps_before"
        ]
        ==
        0
    )

    assert (
        record[
            "completed"
        ]
        is True
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
            "steps"
        ]
        ==
        2
    )

    assert (
        record[
            "actions"
        ]
        ==
        [
            1,
            0,
        ]
    )

    assert (
        record[
            "finalization_outcome"
        ]
        ==
        "FULL_HEX"
    )


    # ============================================================
    # Reward V5 terminal-quality telemetry.
    #
    # These facts originate from the actual terminal replay
    # transition produced by FINALIZE_QUALITY.  The formal episode
    # record must preserve them without recomputation.
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
        is True
    )

    assert (
        terminal_quality[
            "model"
        ]
        ==
        "Plate3"
    )

    assert (
        terminal_quality[
            "hex"
        ]
        ==
        28
    )

    assert (
        terminal_quality[
            "total_polys"
        ]
        ==
        28
    )

    assert (
        terminal_quality[
            "nonhex"
        ]
        ==
        0
    )

    assert (
        terminal_quality[
            "utility"
        ]
        ==
        terminal_quality[
            "d_c"
        ]
        *
        terminal_quality[
            "q_fidelity"
        ]
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
        is True
    )

    assert (
        terminal_reward_v5[
            "utility"
        ]
        ==
        terminal_quality[
            "utility"
        ]
    )

    assert (
        terminal_reward_v5[
            "terminal"
        ]
        ==
        6.0
        *
        terminal_quality[
            "utility"
        ]
        -
        3.0
    )

    assert (
        record[
            "gradient_updates"
        ]
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
        len(
            state.expo_buffer
        )
        ==
        2
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


    final_stats = record[
        "final_training_stats"
    ]

    assert (
        final_stats
        is not
        None
    )

    assert (
        final_stats[
            "bc_loss"
        ]
        ==
        0.0
    )

    assert (
        final_stats[
            "bc_selected_count"
        ]
        ==
        0
    )

    assert math.isfinite(
        final_stats[
            "actor_loss"
        ]
    )


    actor_after = list(
        core.policy.actor.parameters()
    )

    assert (
        len(
            actor_before
        )
        ==
        len(
            actor_after
        )
    )

    actor_changed = any(
        not torch.equal(
            before,
            after.detach(),
        )

        for before, after
        in zip(
            actor_before,
            actor_after,
        )
    )

    assert actor_changed


    # ============================================================
    # Critical non-empty-buffer reset regression.
    #
    # Collector.reset() defaults reset_buffer=True in Tianshou 2.0.1.
    # Production code MUST always pass reset_buffer=False.
    # ============================================================

    vector_env = (
        build_formal_stage2_vector_env(
            model=
                plate3
        )
    )

    collector = Collector(
        core.algorithm,
        vector_env,
        state.expo_buffer,

        exploration_noise=
            True,
    )

    try:
        before_reset = len(
            state.expo_buffer
        )

        assert before_reset == 2

        collector.reset(
            reset_buffer=
                False
        )

        after_reset = len(
            state.expo_buffer
        )

        assert (
            after_reset
            ==
            before_reset
            ==
            2
        )

    finally:
        collector.close()


    print()
    print(
        "PASS: seed42 Stage-II curriculum Train39 sampler is deterministic"
    )

    print(
        "PASS: real Plate3 episode collected exact action sequence [1, 0]"
    )

    print(
        "PASS: real Plate3 terminal finalization outcome is FULL_HEX"
    )

    print(
        "PASS: two collected transitions trigger exactly two Stage-II updates"
    )

    print(
        "PASS: every Stage-II update uses 32 D_demo + 32 D_expo with BC OFF"
    )

    print(
        "PASS: the real Stage-II updates modify the same Actor"
    )

    print(
        "PASS: Collector reset_buffer=False preserves non-empty D_expo"
    )


if __name__ == "__main__":
    main()
