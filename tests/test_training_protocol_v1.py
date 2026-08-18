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


from training.protocol_v1 import (
    FORMAL_TRAINING_BLOCKERS,
    PAPER_BATCH_SIZE,
    PAPER_DEMO_REPLAY_CAPACITY,
    PAPER_DISCOUNT_FACTOR,
    PAPER_ENTROPY_TARGET_COEFFICIENT,
    PAPER_EXPO_REPLAY_CAPACITY,
    PAPER_TRAIN_EPSILON_GREEDY,
    PAPER_LEARNING_RATE,
    PAPER_SOFT_UPDATE_TAU,
    PAPER_STAGE2_REPLAY_RATIO,
    PAPER_TEMPERATURE_MODE,
    PROJECT_BLIND_TEST_ALLOWED_DURING_TRAINING,
    PROJECT_MAIN_DEMO_EPISODES,
    PROJECT_MAIN_DEMO_MODELS,
    PROJECT_MAIN_DEMO_TRANSITIONS,
    PROJECT_MASKED_ENTROPY_ACTION_COUNT_SEMANTICS,
    PROJECT_MASKED_ENTROPY_TARGET_FORMULA,
    PROJECT_EPSILON_RANDOM_SUPPORT,
    PROJECT_MASKED_EPSILON_GREEDY_VERSION,
    PROJECT_MAX_ACTIONS,
    PROJECT_NETWORK_REINITIALIZATION,
    PROJECT_INITIAL_ALPHA,
    PROJECT_INITIAL_ALPHA_BASIS,
    PROJECT_STAGE2_BC_ENABLED,
    PROJECT_STAGE2_EQUAL_REPLAY,
    PROJECT_STAGE2_COLLECTION_UPDATE_RATIO,
    PROJECT_STAGE2_COLLECTION_UPDATE_RATIO_SEMANTICS,
    PROJECT_STAGE2_COLLECTION_UPDATE_RATIO_BASIS,
    UNRESOLVED_BC_WEIGHT,
    UNRESOLVED_STAGE1_GRADIENT_STEPS,
    UNRESOLVED_STAGE1_STOPPING_RULE,
    UNRESOLVED_STAGE2_TOTAL_ENVIRONMENT_STEPS,
    assert_formal_training_ready,
    formal_training_ready,
    paper_entropy_target,
    project_masked_entropy_target,
)


def main():
    # ------------------------------------------------------------
    # Paper-specified facts.
    # ------------------------------------------------------------

    assert PAPER_BATCH_SIZE == 64

    assert (
        PAPER_DEMO_REPLAY_CAPACITY
        ==
        50_000
    )

    assert (
        PAPER_EXPO_REPLAY_CAPACITY
        ==
        25_000
    )

    assert math.isclose(
        PAPER_DISCOUNT_FACTOR,
        0.95,
    )

    assert math.isclose(
        PAPER_LEARNING_RATE,
        0.001,
    )

    assert math.isclose(
        PAPER_SOFT_UPDATE_TAU,
        0.005,
    )

    assert math.isclose(
        PAPER_ENTROPY_TARGET_COEFFICIENT,
        0.6,
    )

    assert (
        PAPER_TEMPERATURE_MODE
        ==
        "AUTO_ALPHA"
    )

    assert math.isclose(
        PAPER_TRAIN_EPSILON_GREEDY,
        0.05,
    )

    assert (
        PAPER_STAGE2_REPLAY_RATIO
        ==
        (1, 1)
    )


    # ------------------------------------------------------------
    # Entropy formula is implemented, but |A| semantics are NOT
    # chosen here.
    # ------------------------------------------------------------

    target_1 = paper_entropy_target(
        1
    )

    assert math.isclose(
        target_1,
        0.0,
        abs_tol=1.0e-12,
    )

    target_331 = (
        paper_entropy_target(
            331
        )
    )

    expected_331 = (
        0.6
        *
        math.log(
            331.0
        )
    )

    assert math.isclose(
        target_331,
        expected_331,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    )

    # LoopyCuts masked adaptation:
    # target is always 60% of the maximum entropy available under
    # the CURRENT legal-action support.
    for legal_count in [
        1,
        2,
        14,
        32,
        65,
        331,
    ]:
        target = (
            project_masked_entropy_target(
                legal_count
            )
        )

        max_entropy = math.log(
            float(
                legal_count
            )
        )

        assert math.isclose(
            target,
            0.6 * max_entropy,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )

        assert (
            target
            <=
            max_entropy
            +
            1.0e-12
        )


    # ------------------------------------------------------------
    # Frozen project facts.
    # ------------------------------------------------------------

    assert PROJECT_MAX_ACTIONS == 331

    assert (
        PROJECT_MASKED_ENTROPY_ACTION_COUNT_SEMANTICS
        ==
        "CURRENT_LEGAL_ACTION_COUNT"
    )

    assert (
        PROJECT_MASKED_ENTROPY_TARGET_FORMULA
        ==
        "0.6 * log(n_legal(s))"
    )

    assert (
        PROJECT_EPSILON_RANDOM_SUPPORT
        ==
        "CURRENT_LEGAL_ACTIONS"
    )

    assert (
        PROJECT_MASKED_EPSILON_GREEDY_VERSION
        ==
        "loopycuts_masked_epsilon_greedy_v1"
    )

    assert (
        "masked_epsilon_greedy_implementation"
        not in
        FORMAL_TRAINING_BLOCKERS
    )

    assert (
        PROJECT_MAIN_DEMO_EPISODES
        ==
        30
    )

    assert (
        PROJECT_MAIN_DEMO_TRANSITIONS
        ==
        605
    )

    assert set(
        PROJECT_MAIN_DEMO_MODELS
    ) == {
        "mech10",
        "Plate3",
        "dog",
        "prism",
        "wave",
        "cat",
        "kong",
        "bolt",
        "dancer",
        "hand",
        "kitten",
        "trebol",
        "wedge",
        "Plate2",
        "cube_minus_sphere",
        "bunny",
        "Plate4",
        "tris_closed",
        "cup",
        "tris_open",
        "ujoint",
        "metatron",
        "joint",
        "lever_arm",
        "mechanical02",
        "gear",
        "bearing_plate",
        "hinge",
        "impeller",
        "motor_tail",
    }

    assert (
        PROJECT_STAGE2_EQUAL_REPLAY
        is True
    )

    assert (
        PROJECT_STAGE2_BC_ENABLED
        is False
    )

    assert (
        PROJECT_NETWORK_REINITIALIZATION
        is False
    )

    assert (
        PROJECT_BLIND_TEST_ALLOWED_DURING_TRAINING
        is False
    )


    # ------------------------------------------------------------
    # Project-specified / framework-default-aligned facts.
    # ------------------------------------------------------------

    assert math.isclose(
        PROJECT_INITIAL_ALPHA,
        1.0,
    )

    assert (
        PROJECT_INITIAL_ALPHA_BASIS
        ==
        "TIANSHOU_2_0_1_AUTO_ALPHA_DEFAULT_LOG_ALPHA_0"
    )

    assert math.isclose(
        PROJECT_STAGE2_COLLECTION_UPDATE_RATIO,
        1.0,
    )

    assert (
        PROJECT_STAGE2_COLLECTION_UPDATE_RATIO_SEMANTICS
        ==
        "GRADIENT_UPDATES_PER_COLLECTED_ENVIRONMENT_TRANSITION"
    )

    assert (
        PROJECT_STAGE2_COLLECTION_UPDATE_RATIO_BASIS
        ==
        "TIANSHOU_2_0_1_OFFPOLICY_DEFAULT"
    )

    assert (
        "initial_alpha"
        not in
        FORMAL_TRAINING_BLOCKERS
    )

    assert (
        "stage2_collection_update_ratio"
        not in
        FORMAL_TRAINING_BLOCKERS
    )


    # ------------------------------------------------------------
    # Critical unresolved values must remain explicit.
    # ------------------------------------------------------------

    assert UNRESOLVED_BC_WEIGHT is None

    assert (
        UNRESOLVED_STAGE1_GRADIENT_STEPS
        is None
    )

    assert (
        UNRESOLVED_STAGE1_STOPPING_RULE
        is None
    )

    assert (
        UNRESOLVED_STAGE2_TOTAL_ENVIRONMENT_STEPS
        is None
    )

    assert set(
        FORMAL_TRAINING_BLOCKERS
    ) == {
        "bc_weight",
        "stage1_gradient_steps_or_stopping_rule",
        "stage2_total_environment_steps",
    }

    assert (
        formal_training_ready()
        is False
    )


    try:
        assert_formal_training_ready()
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "Formal training gate unexpectedly opened"
        )


    print(
        "PASS: Training Protocol V1 preserves "
        "paper-specified facts and explicitly "
        "blocks unresolved formal-training choices"
    )


if __name__ == "__main__":
    main()
