from __future__ import annotations

import math
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


from training.formal_training_v1 import (
    FORMAL_TRAINER_CORE_VERSION,
    enter_formal_stage2,
    prepare_formal_training_core,
    run_formal_stage1,
)

from training.protocol_v1 import (
    PROJECT_BC_WEIGHT,
    PROJECT_MAIN_DEMO_EPISODES,
    PROJECT_MAIN_DEMO_TRANSITIONS,
    PROJECT_STAGE1_GRADIENT_STEPS,
    PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS,
    PROJECT_STAGE2_EXPLORATION_EPSILON,
    PROJECT_STAGE2_SAMPLES_PER_BUFFER,
)


def main():
    core = prepare_formal_training_core(
        seed=
            42
    )


    # ============================================================
    # Prepared formal Stage-I state.
    # ============================================================

    assert (
        FORMAL_TRAINER_CORE_VERSION
        ==
        "loopycuts_formal_training_core_v1"
    )

    assert (
        core.seed
        ==
        42
    )

    assert (
        core.stage
        ==
        "STAGE_I"
    )

    assert (
        len(
            core.demo_records
        )
        ==
        PROJECT_MAIN_DEMO_EPISODES
        ==
        30
    )

    assert (
        len(
            core.demo_buffer
        )
        ==
        PROJECT_MAIN_DEMO_TRANSITIONS
        ==
        605
    )

    assert (
        core.input_provenance[
            "train49_models"
        ]
        ==
        49
    )

    assert (
        core.input_provenance[
            "selected_bc_weight"
        ]
        ==
        PROJECT_BC_WEIGHT
        ==
        3.0
    )

    assert (
        core.algorithm.bc_enabled
        is True
    )

    assert math.isclose(
        core.algorithm.bc_weight,
        3.0,
    )

    assert (
        core.policy.deterministic_eval
        is True
    )

    assert math.isclose(
        core.policy.exploration_epsilon,
        0.0,
    )

    assert math.isclose(
        core.auto_alpha.value,
        1.0,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )


    identities_before_stage1 = {
        "algorithm":
            id(
                core.algorithm
            ),

        "policy":
            id(
                core.policy
            ),

        "actor":
            id(
                core.policy.actor
            ),

        "critic1":
            id(
                core.algorithm.critic
            ),

        "critic2":
            id(
                core.algorithm.critic2
            ),

        "auto_alpha":
            id(
                core.auto_alpha
            ),
    }


    print("=" * 96)
    print("FORMAL TRAINER CORE -- EXACT STAGE-I INTEGRATION SMOKE")
    print("=" * 96)

    print(
        "seed                  :",
        core.seed,
    )

    print(
        "D_demo episodes       :",
        len(
            core.demo_records
        ),
    )

    print(
        "D_demo transitions    :",
        len(
            core.demo_buffer
        ),
    )

    print(
        "lambda_BC             :",
        core.algorithm.bc_weight,
    )

    print(
        "alpha initial         :",
        core.auto_alpha.value,
    )

    print()


    # ============================================================
    # Exact formal Stage-I budget.
    #
    # This is an integration smoke of the trainer core.
    # It is NOT a formal experimental run and writes no checkpoint.
    # ============================================================

    stage1 = run_formal_stage1(
        core
    )


    assert (
        stage1[
            "gradient_updates"
        ]
        ==
        PROJECT_STAGE1_GRADIENT_STEPS
        ==
        782
    )

    assert (
        stage1[
            "sampled_demo_transitions"
        ]
        ==
        PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS
        ==
        50_048
    )

    assert (
        core.stage
        ==
        "STAGE_I"
    )

    assert (
        core.algorithm.bc_enabled
        is True
    )

    assert (
        math.isfinite(
            stage1[
                "alpha_after_stage1"
            ]
        )
    )

    assert (
        stage1[
            "alpha_after_stage1"
        ]
        >
        0.0
    )


    identities_after_stage1 = {
        "algorithm":
            id(
                core.algorithm
            ),

        "policy":
            id(
                core.policy
            ),

        "actor":
            id(
                core.policy.actor
            ),

        "critic1":
            id(
                core.algorithm.critic
            ),

        "critic2":
            id(
                core.algorithm.critic2
            ),

        "auto_alpha":
            id(
                core.auto_alpha
            ),
    }


    assert (
        identities_after_stage1
        ==
        identities_before_stage1
    )


    # ============================================================
    # Enter Stage-II without reinitialization.
    # ============================================================

    transition = enter_formal_stage2(
        core
    )


    assert (
        core.stage
        ==
        "STAGE_II"
    )

    assert (
        core.algorithm.bc_enabled
        is False
    )

    assert (
        transition[
            "bc_enabled"
        ]
        is False
    )

    assert math.isclose(
        core.policy.exploration_epsilon,
        PROJECT_STAGE2_EXPLORATION_EPSILON,
    )

    assert math.isclose(
        core.policy.exploration_epsilon,
        0.05,
    )

    assert (
        transition[
            "samples_per_replay_source"
        ]
        ==
        PROJECT_STAGE2_SAMPLES_PER_BUFFER
        ==
        32
    )

    assert (
        transition[
            "object_identities"
        ]
        ==
        identities_before_stage1
    )


    # D_demo remains the same formal replay.
    assert (
        len(
            core.demo_buffer
        )
        ==
        605
    )


    print()
    print("=" * 96)
    print("STAGE-I SUMMARY")
    print("=" * 96)

    print(
        "gradient updates        :",
        stage1[
            "gradient_updates"
        ],
    )

    print(
        "sampled transitions     :",
        stage1[
            "sampled_demo_transitions"
        ],
    )

    print(
        "alpha after Stage-I     :",
        stage1[
            "alpha_after_stage1"
        ],
    )

    print(
        "elapsed seconds         :",
        stage1[
            "elapsed_seconds"
        ],
    )

    print()
    print("=" * 96)
    print("STAGE-II ENTRY")
    print("=" * 96)

    print(
        "stage                   :",
        core.stage,
    )

    print(
        "BC enabled              :",
        core.algorithm.bc_enabled,
    )

    print(
        "exploration epsilon     :",
        core.policy.exploration_epsilon,
    )

    print(
        "replay samples/source   :",
        transition[
            "samples_per_replay_source"
        ],
    )

    print()

    print(
        "PASS: formal core loads exact 30/605 D_demo and lambda_BC=3.0"
    )

    print(
        "PASS: exact 782-update / 50048-transition Stage-I budget executes"
    )

    print(
        "PASS: Stage-I -> Stage-II preserves algorithm/network/alpha identities"
    )

    print(
        "PASS: Stage-II begins with BC OFF and epsilon=0.05"
    )


if __name__ == "__main__":
    main()
