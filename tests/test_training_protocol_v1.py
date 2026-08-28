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
    PAPER_OPTIMIZER_FAMILY,
    PAPER_ADAMW_BETAS,
    PAPER_ADAMW_EPS,
    PAPER_ADAMW_WEIGHT_DECAY,
    PAPER_GRADIENT_STEPS_PER_EPISODE,
    PAPER_UPDATE_CADENCE,
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
    PROJECT_ACTOR_CRITIC_OPTIMIZER_FAMILY,
    PROJECT_ACTOR_CRITIC_BETAS,
    PROJECT_ACTOR_CRITIC_EPS,
    PROJECT_ACTOR_CRITIC_WEIGHT_DECAY,
    PROJECT_ACTOR_CRITIC_OPTIMIZER_BASIS,
    PROJECT_ALPHA_OPTIMIZER_FAMILY,
    PROJECT_ALPHA_ADAM_BETAS,
    PROJECT_ALPHA_ADAM_EPS,
    PROJECT_ALPHA_ADAM_WEIGHT_DECAY,
    PROJECT_ALPHA_OPTIMIZER_BASIS,
    PROJECT_ACTOR_LEARNING_RATE,
    PROJECT_CRITIC1_LEARNING_RATE,
    PROJECT_CRITIC2_LEARNING_RATE,
    PROJECT_ACTOR_CRITIC_LEARNING_RATE_BASIS,
    PROJECT_ALPHA_LEARNING_RATE,
    PROJECT_ALPHA_LEARNING_RATE_BASIS,
    PROJECT_N_STEP_RETURN_HORIZON,
    PROJECT_N_STEP_RETURN_HORIZON_BASIS,
    PROJECT_LEARNING_RATE_SCHEDULER,
    PROJECT_LEARNING_RATE_SCHEDULER_BASIS,
    PROJECT_STAGE1_TARGET_SAMPLED_DEMO_TRANSITIONS,
    PROJECT_STAGE1_GRADIENT_STEPS,
    PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS,
    PROJECT_STAGE1_SAMPLING_SEMANTICS,
    PROJECT_STAGE1_GRADIENT_STEPS_BASIS,
    PROJECT_STAGE1_STOPPING_RULE,
    PROJECT_STAGE1_STOPPING_RULE_SEMANTICS,
    PROJECT_BC_WEIGHT_CALIBRATION_VERSION,
    PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES,
    PROJECT_BC_WEIGHT_CALIBRATION_SEEDS,
    PROJECT_BC_WEIGHT_CALIBRATION_SPLIT,
    PROJECT_BC_WEIGHT_CALIBRATION_MODELS,
    PROJECT_BC_WEIGHT_CALIBRATION_STAGE1_GRADIENT_STEPS,
    PROJECT_BC_WEIGHT_CALIBRATION_STAGE2_ENABLED,
    PROJECT_BC_WEIGHT_CALIBRATION_EVAL_DETERMINISTIC,
    PROJECT_BC_WEIGHT_CALIBRATION_EVAL_EPSILON,
    PROJECT_BC_WEIGHT_CALIBRATION_DEVICE,
    PROJECT_BC_WEIGHT_CALIBRATION_TORCH_NUM_THREADS,
    PROJECT_BC_WEIGHT_CALIBRATION_TORCH_NUM_INTEROP_THREADS,
    PROJECT_BC_WEIGHT_CALIBRATION_TORCH_DETERMINISTIC_ALGORITHMS,
    PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_VERSION,
    PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_SEED,
    PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_BC_WEIGHT,
    PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_UPDATES,
    PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_BASE_COMMIT,
    PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_SHA256,
    PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_RESULT,
    PROJECT_BC_WEIGHT_CALIBRATION_DEV_ALLOWED,
    PROJECT_BC_WEIGHT_CALIBRATION_BLIND_ALLOWED,
    PROJECT_BC_WEIGHT_CALIBRATION_NONHEX_METRIC,
    PROJECT_BC_WEIGHT_CALIBRATION_SELECTION_RULE,
    PROJECT_STAGE2_BC_ENABLED,
    PROJECT_STAGE2_EQUAL_REPLAY,
    PROJECT_STAGE2_COLLECTION_UPDATE_RATIO,
    PROJECT_STAGE2_COLLECTION_UPDATE_RATIO_SEMANTICS,
    PROJECT_STAGE2_COLLECTION_UPDATE_RATIO_BASIS,
    PROJECT_STAGE2_COLLECTION_UPDATE_RATIO_REFERENCE,
    PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS,
    PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS_SEMANTICS,
    PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS_BASIS,
    PROJECT_BC_WEIGHT,
    PROJECT_FORMAL_TRAINING_SEEDS,
    PROJECT_FORMAL_TRAINING_DEVICE,
    PROJECT_FORMAL_TRAINING_TORCH_NUM_THREADS,
    PROJECT_FORMAL_TRAINING_TORCH_NUM_INTEROP_THREADS,
    PROJECT_FORMAL_TRAINING_TORCH_DETERMINISTIC_ALGORITHMS,
    PROJECT_STAGE2_EXPLORATION_EPSILON,
    PROJECT_STAGE2_COLLECTOR_EXPLORATION_NOISE,
    PROJECT_STAGE2_SAMPLES_PER_BUFFER,
    PROJECT_STAGE2_EXPO_REPLAY_CAPACITY,
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

    assert (
        PAPER_OPTIMIZER_FAMILY
        ==
        "ADAMW"
    )

    assert (
        PAPER_ADAMW_BETAS
        ==
        (
            0.99,
            0.999,
        )
    )

    assert math.isclose(
        PAPER_ADAMW_EPS,
        1.0e-8,
    )

    assert math.isclose(
        PAPER_ADAMW_WEIGHT_DECAY,
        0.01,
    )

    assert (
        PAPER_GRADIENT_STEPS_PER_EPISODE
        ==
        32
    )

    assert (
        PAPER_UPDATE_CADENCE
        ==
        "ONE_COMPLETED_EPISODE_THEN_32_GRADIENT_STEPS"
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


    # ------------------------------------------------------------
    # Explicit SAC runtime configuration.
    # ------------------------------------------------------------

    assert (
        PROJECT_ACTOR_CRITIC_OPTIMIZER_FAMILY
        ==
        "ADAMW"
    )

    assert (
        PROJECT_ACTOR_CRITIC_OPTIMIZER_FAMILY
        ==
        PAPER_OPTIMIZER_FAMILY
    )

    assert (
        PROJECT_ACTOR_CRITIC_BETAS
        ==
        PAPER_ADAMW_BETAS
        ==
        (
            0.99,
            0.999,
        )
    )

    assert math.isclose(
        PROJECT_ACTOR_CRITIC_EPS,
        PAPER_ADAMW_EPS,
    )

    assert math.isclose(
        PROJECT_ACTOR_CRITIC_WEIGHT_DECAY,
        PAPER_ADAMW_WEIGHT_DECAY,
    )

    assert (
        PROJECT_ACTOR_CRITIC_OPTIMIZER_BASIS
        ==
        "ZHANG_2025_IMPLEMENTATION_DETAILS"
    )

    assert (
        PROJECT_ALPHA_OPTIMIZER_FAMILY
        ==
        "ADAM"
    )

    assert (
        PROJECT_ALPHA_ADAM_BETAS
        ==
        (
            0.9,
            0.999,
        )
    )

    assert math.isclose(
        PROJECT_ALPHA_ADAM_EPS,
        1.0e-8,
    )

    assert math.isclose(
        PROJECT_ALPHA_ADAM_WEIGHT_DECAY,
        0.0,
    )

    assert (
        PROJECT_ALPHA_OPTIMIZER_BASIS
        ==
        "TIANSHOU_2_0_1_DEFAULT_ADAM_PROJECT_ADAPTATION"
    )

    assert math.isclose(
        PROJECT_ACTOR_LEARNING_RATE,
        0.001,
    )

    assert math.isclose(
        PROJECT_CRITIC1_LEARNING_RATE,
        0.001,
    )

    assert math.isclose(
        PROJECT_CRITIC2_LEARNING_RATE,
        0.001,
    )

    assert math.isclose(
        PROJECT_ACTOR_LEARNING_RATE,
        PAPER_LEARNING_RATE,
    )

    assert math.isclose(
        PROJECT_CRITIC1_LEARNING_RATE,
        PAPER_LEARNING_RATE,
    )

    assert math.isclose(
        PROJECT_CRITIC2_LEARNING_RATE,
        PAPER_LEARNING_RATE,
    )

    assert (
        PROJECT_ACTOR_CRITIC_LEARNING_RATE_BASIS
        ==
        "PAPER_LEARNING_RATE"
    )

    assert math.isclose(
        PROJECT_ALPHA_LEARNING_RATE,
        3.0e-4,
    )

    assert (
        PROJECT_ALPHA_LEARNING_RATE_BASIS
        ==
        "TIANSHOU_2_0_1_AUTO_ALPHA_DEFAULT"
    )

    assert (
        PROJECT_N_STEP_RETURN_HORIZON
        ==
        1
    )

    assert (
        PROJECT_N_STEP_RETURN_HORIZON_BASIS
        ==
        "TIANSHOU_2_0_1_DISCRETE_SAC_DEFAULT"
    )

    assert (
        PROJECT_LEARNING_RATE_SCHEDULER
        ==
        "NONE"
    )

    assert (
        PROJECT_LEARNING_RATE_SCHEDULER_BASIS
        ==
        "TIANSHOU_2_0_1_DEFAULT_NO_SCHEDULER"
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
        "PROJECT_SPECIFIED_TRANSITION_NORMALIZED_OFFPOLICY_CADENCE"
    )

    assert (
        PROJECT_STAGE2_COLLECTION_UPDATE_RATIO_REFERENCE
        ==
        "ZHANG_2025_USES_32_GRADIENT_STEPS_PER_COMPLETED_EPISODE"
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

    assert (
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
        ==
        25_000
    )

    assert (
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
        ==
        PAPER_EXPO_REPLAY_CAPACITY
    )

    assert (
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS_SEMANTICS
        ==
        "EXACT_NEWLY_COLLECTED_ENVIRONMENT_TRANSITIONS"
    )

    assert (
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS_BASIS
        ==
        "PROJECT_SPECIFIED_ONE_PAPER_D_EXPO_REPLAY_CAPACITY"
    )

    assert (
        "stage2_total_environment_steps"
        not in
        FORMAL_TRAINING_BLOCKERS
    )


    # ------------------------------------------------------------
    # Project-specified Stage-I fixed optimization budget.
    # ------------------------------------------------------------

    assert (
        PROJECT_STAGE1_TARGET_SAMPLED_DEMO_TRANSITIONS
        ==
        50_000
    )

    assert (
        PROJECT_STAGE1_TARGET_SAMPLED_DEMO_TRANSITIONS
        ==
        PAPER_DEMO_REPLAY_CAPACITY
    )

    assert (
        PROJECT_STAGE1_GRADIENT_STEPS
        ==
        782
    )

    assert (
        PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS
        ==
        50_048
    )

    assert (
        PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS
        ==
        PROJECT_STAGE1_GRADIENT_STEPS
        *
        PAPER_BATCH_SIZE
    )

    assert (
        PROJECT_STAGE1_SAMPLING_SEMANTICS
        ==
        "TIANSHOU_2_0_1_RANDOM_WITH_REPLACEMENT"
    )

    assert (
        PROJECT_STAGE1_GRADIENT_STEPS_BASIS
        ==
        "PROJECT_SPECIFIED_ONE_PAPER_D_DEMO_REPLAY_CAPACITY"
    )

    assert (
        PROJECT_STAGE1_STOPPING_RULE
        ==
        "FIXED_GRADIENT_BUDGET_NO_DATA_DEPENDENT_EARLY_STOP"
    )

    assert (
        PROJECT_STAGE1_STOPPING_RULE_SEMANTICS
        ==
        "STOP_AFTER_EXACT_PROJECT_STAGE1_GRADIENT_STEPS"
    )

    assert (
        "stage1_gradient_steps_or_stopping_rule"
        not in
        FORMAL_TRAINING_BLOCKERS
    )


    # ------------------------------------------------------------
    # Frozen BC-weight calibration protocol.
    # ------------------------------------------------------------

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_VERSION
        ==
        "bc_weight_calibration_v2"
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES
        ==
        (
            0.1,
            0.3,
            0.5,
            1.0,
            3.0,
        )
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_SEEDS
        ==
        (
            42,
            43,
            44,
        )
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_SPLIT
        ==
        "engineering_calibration"
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_MODELS
        ==
        (
            "bimba",
            "deckel",
            "BracketInches",
            "eraser_ball",
            "cylinder_plate",
        )
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_STAGE1_GRADIENT_STEPS
        ==
        PROJECT_STAGE1_GRADIENT_STEPS
        ==
        782
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_STAGE2_ENABLED
        is False
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_EVAL_DETERMINISTIC
        is True
    )

    assert math.isclose(
        PROJECT_BC_WEIGHT_CALIBRATION_EVAL_EPSILON,
        0.0,
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_DEVICE
        ==
        "cpu"
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_TORCH_NUM_THREADS
        ==
        8
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_TORCH_NUM_INTEROP_THREADS
        ==
        8
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_TORCH_DETERMINISTIC_ALGORITHMS
        is False
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_VERSION
        ==
        "stage1_cpu_repeatability_probe_v1"
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_SEED
        ==
        42
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_BC_WEIGHT
        ==
        2.0
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_UPDATES
        ==
        20
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_BASE_COMMIT
        ==
        "77f393af7ae2a0a07b86b86c05c2e03f51b3210b"
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_SHA256
        ==
        (
            "a3c367167cb8f1f710c4eaea5edf23f"
            "dee50d401e5ca083df5fc82f4cf7f83fe"
        )
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_RESULT
        ==
        (
            "BITWISE_IDENTICAL_ACROSS_TWO_"
            "INDEPENDENT_PROCESSES"
        )
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_DEV_ALLOWED
        is False
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_BLIND_ALLOWED
        is False
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_NONHEX_METRIC
        ==
        "SUM_TOTAL_MINUS_HEX_DIV_SUM_TOTAL_FOR_COMPLETED_FINALIZATIONS"
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_SELECTION_RULE
        ==
        (
            "LEXICOGRAPHIC:"
            "MAX_FULL_HEX,"
            "MIN_FINALIZATION_CRASH,"
            "MIN_AGGREGATE_NONHEX_FRACTION,"
            "MAX_MEAN_EPISODE_RETURN,"
            "MIN_BC_WEIGHT"
        )
    )


    # ------------------------------------------------------------
    # Final formal-training freeze.
    # ------------------------------------------------------------

    assert math.isclose(
        PROJECT_BC_WEIGHT,
        3.0,
    )

    assert (
        PROJECT_FORMAL_TRAINING_SEEDS
        ==
        (
            42,
            43,
            44,
        )
    )

    assert (
        PROJECT_FORMAL_TRAINING_DEVICE
        ==
        "cpu"
    )

    assert (
        PROJECT_FORMAL_TRAINING_TORCH_NUM_THREADS
        ==
        8
    )

    assert (
        PROJECT_FORMAL_TRAINING_TORCH_NUM_INTEROP_THREADS
        ==
        8
    )

    assert (
        PROJECT_FORMAL_TRAINING_TORCH_DETERMINISTIC_ALGORITHMS
        is False
    )

    assert math.isclose(
        PROJECT_STAGE2_EXPLORATION_EPSILON,
        0.05,
    )

    assert (
        PROJECT_STAGE2_COLLECTOR_EXPLORATION_NOISE
        is True
    )

    assert (
        PROJECT_STAGE2_SAMPLES_PER_BUFFER
        ==
        32
    )

    assert (
        PROJECT_STAGE2_EXPO_REPLAY_CAPACITY
        ==
        25_000
    )

    assert (
        FORMAL_TRAINING_BLOCKERS
        ==
        ()
    )

    assert (
        formal_training_ready()
        is True
    )

    assert_formal_training_ready()


    print(
        "PASS: current training protocol preserves "
        "paper-specified facts and is fully frozen "
        "for formal training"
    )


if __name__ == "__main__":
    main()
