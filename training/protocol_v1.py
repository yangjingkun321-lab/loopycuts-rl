from __future__ import annotations

import math


PROTOCOL_VERSION = (
    "loopycuts_training_protocol_v1"
)


# ================================================================
# PAPER-SPECIFIED PARAMETERS
#
# Zhang et al.,
# Reinforcement learning based automatic block decomposition of
# solid models for hexahedral meshing,
# Computer-Aided Design 182 (2025) 103850.
#
# Table 1 / Algorithm 1.
# ================================================================

PAPER_BATCH_SIZE = 64

PAPER_DEMO_REPLAY_CAPACITY = 50_000

PAPER_EXPO_REPLAY_CAPACITY = 25_000

PAPER_DISCOUNT_FACTOR = 0.95

PAPER_LEARNING_RATE = 0.001

# Zhang et al. implementation details.
PAPER_OPTIMIZER_FAMILY = (
    "ADAMW"
)

PAPER_ADAMW_BETAS = (
    0.99,
    0.999,
)

PAPER_ADAMW_EPS = 1.0e-8

PAPER_ADAMW_WEIGHT_DECAY = 0.01

# The reference method performs 32 gradient steps after each
# completed environment episode.  This is recorded as a paper fact,
# but is NOT transferred directly to LoopyCuts because LoopyCuts
# episode lengths vary substantially.
PAPER_GRADIENT_STEPS_PER_EPISODE = 32

PAPER_UPDATE_CADENCE = (
    "ONE_COMPLETED_EPISODE_THEN_32_GRADIENT_STEPS"
)

PAPER_SOFT_UPDATE_TAU = 0.005

PAPER_ENTROPY_TARGET_COEFFICIENT = 0.6

PAPER_ENTROPY_TARGET_FORMULA = (
    "-0.6 * log(1 / |A|)"
)

PAPER_ACTION_SPACE_DEFINITION = (
    "|A| = 4 * |F|"
)

PAPER_TRAIN_EPSILON_GREEDY = 0.05

PAPER_TEMPERATURE_MODE = (
    "AUTO_ALPHA"
)

PAPER_STAGE2_REPLAY_RATIO = (
    1,
    1,
)


def paper_entropy_target(
    action_count: int,
) -> float:
    """
    Evaluate the paper's entropy-target formula for a GIVEN
    interpretation of |A|:

        H_target = -0.6 * log(1 / |A|)
                 =  0.6 * log(|A|)

    For the original paper, |A| denotes the current discrete
    action-space size.

    LoopyCuts uses explicit legality masking. Therefore the effective
    categorical support contains only currently legal actions.

    The project-specific helper below applies this same 0.6 * log(|A|)
    rule to the current legal-action support.
    """

    action_count = int(
        action_count
    )

    if action_count <= 0:
        raise ValueError(
            "action_count must be positive"
        )

    return float(
        -PAPER_ENTROPY_TARGET_COEFFICIENT
        *
        math.log(
            1.0
            /
            float(
                action_count
            )
        )
    )


def project_masked_entropy_target(
    legal_action_count: int,
) -> float:
    """
    LoopyCuts masked-action adaptation of the paper entropy target.

    Because illegal actions have exactly zero probability, the
    categorical support at state s has:

        n_legal(s)

    actions and maximum entropy:

        log(n_legal(s)).

    Therefore:

        H_target(s)
            =
        0.6 * log(n_legal(s)).
    """

    return paper_entropy_target(
        legal_action_count
    )


# ================================================================
# PROJECT-FROZEN ADAPTATIONS
# ================================================================

PROJECT_OBSERVATION_VERSION = (
    "observation_v1"
)

PROJECT_REWARD_VERSION = (
    "reward_v2"
)

PROJECT_RUNTIME_REWARD_VERSION = (
    "final_v2"
)

PROJECT_DEMO_QUALITY_VERSION = (
    "demo_quality_v1"
)

PROJECT_ALGORITHM_VERSION = (
    "loopycuts_demo_guided_discrete_sac_v1"
)

PROJECT_NETWORK_VERSION = (
    "loopycuts_actor_critic_v1"
)

PROJECT_MAX_ACTIONS = 331

PROJECT_MASKED_ENTROPY_ACTION_COUNT_SEMANTICS = (
    "CURRENT_LEGAL_ACTION_COUNT"
)

PROJECT_MASKED_ENTROPY_TARGET_FORMULA = (
    "0.6 * log(n_legal(s))"
)

PROJECT_EPSILON_RANDOM_SUPPORT = (
    "CURRENT_LEGAL_ACTIONS"
)

PROJECT_MASKED_EPSILON_GREEDY_VERSION = (
    "loopycuts_masked_epsilon_greedy_v1"
)

# Final formal D_demo snapshot frozen from
# Demonstration Quality V1.
PROJECT_MAIN_DEMO_EPISODES = 30

PROJECT_MAIN_DEMO_TRANSITIONS = 605

PROJECT_MAIN_DEMO_MODELS = (
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
)

PROJECT_STAGE1_REPLAY = (
    "D_demo_only"
)

PROJECT_STAGE1_BC = (
    "Q_FILTERED_MASKED_BC"
)

PROJECT_STAGE2_BC_ENABLED = False

PROJECT_STAGE2_EQUAL_REPLAY = True

PROJECT_NETWORK_REINITIALIZATION = False

PROJECT_BLIND_TEST_ALLOWED_DURING_TRAINING = False


# ================================================================
# PROJECT-SPECIFIED / FRAMEWORK-DEFAULT-ALIGNED PARAMETERS
#
# These are NOT paper-specified values.
#
# They deliberately follow the defaults of the frozen experimental
# framework environment:
#
#     Tianshou 2.0.1
#
# Verified locally before formal training.
# ================================================================

PROJECT_INITIAL_ALPHA = 1.0

PROJECT_INITIAL_ALPHA_BASIS = (
    "TIANSHOU_2_0_1_AUTO_ALPHA_DEFAULT_LOG_ALPHA_0"
)


# ================================================================
# EXPLICIT SAC RUNTIME CONFIGURATION
#
# IMPORTANT:
#
# Do not construct formal/calibration SAC by inheriting the complete
# Tianshou DiscreteSACParams defaults.
#
# Example:
#
#     Tianshou 2.0.1 default gamma = 0.99
#     frozen project/paper gamma   = 0.95
#
# Therefore every runtime-sensitive setting used by this project is
# explicit below.
# ================================================================

# Actor and twin critics follow the optimizer configuration
# explicitly reported by Zhang et al.
#
# Tianshou 2.0.1 has no dedicated AdamWOptimizerFactory, so the
# runtime uses:
#
#     TorchOptimizerFactory(torch.optim.AdamW, ...)
#
# This has been capability-audited in the frozen environment.
PROJECT_ACTOR_CRITIC_OPTIMIZER_FAMILY = (
    PAPER_OPTIMIZER_FAMILY
)

PROJECT_ACTOR_CRITIC_BETAS = (
    PAPER_ADAMW_BETAS
)

PROJECT_ACTOR_CRITIC_EPS = (
    PAPER_ADAMW_EPS
)

PROJECT_ACTOR_CRITIC_WEIGHT_DECAY = (
    PAPER_ADAMW_WEIGHT_DECAY
)

PROJECT_ACTOR_CRITIC_OPTIMIZER_BASIS = (
    "ZHANG_2025_IMPLEMENTATION_DETAILS"
)


# Auto-alpha remains an explicit project/framework adaptation.
#
# The paper states that AdamW is used during training, but does not
# separately establish that the temperature parameter uses exactly
# the same optimizer instance/configuration.  We therefore do not
# silently promote that inference into a paper fact.
PROJECT_ALPHA_OPTIMIZER_FAMILY = (
    "ADAM"
)

PROJECT_ALPHA_ADAM_BETAS = (
    0.9,
    0.999,
)

PROJECT_ALPHA_ADAM_EPS = 1.0e-8

PROJECT_ALPHA_ADAM_WEIGHT_DECAY = 0.0

PROJECT_ALPHA_OPTIMIZER_BASIS = (
    "TIANSHOU_2_0_1_DEFAULT_ADAM_PROJECT_ADAPTATION"
)


# Actor / critics use the learning rate specified by the paper.
PROJECT_ACTOR_LEARNING_RATE = (
    PAPER_LEARNING_RATE
)

PROJECT_CRITIC1_LEARNING_RATE = (
    PAPER_LEARNING_RATE
)

PROJECT_CRITIC2_LEARNING_RATE = (
    PAPER_LEARNING_RATE
)

PROJECT_ACTOR_CRITIC_LEARNING_RATE_BASIS = (
    "PAPER_LEARNING_RATE"
)


# Auto-alpha optimizer LR is not paper-specified.
# It is frozen to the audited Tianshou 2.0.1 auto-alpha default.
PROJECT_ALPHA_LEARNING_RATE = 3.0e-4

PROJECT_ALPHA_LEARNING_RATE_BASIS = (
    "TIANSHOU_2_0_1_AUTO_ALPHA_DEFAULT"
)


# Tianshou 2.0.1 Discrete SAC default, explicitly frozen rather
# than silently inherited by the runner.
PROJECT_N_STEP_RETURN_HORIZON = 1

PROJECT_N_STEP_RETURN_HORIZON_BASIS = (
    "TIANSHOU_2_0_1_DISCRETE_SAC_DEFAULT"
)


# No LR scheduler is used for actor, critics, or alpha.
PROJECT_LEARNING_RATE_SCHEDULER = (
    "NONE"
)

PROJECT_LEARNING_RATE_SCHEDULER_BASIS = (
    "TIANSHOU_2_0_1_DEFAULT_NO_SCHEDULER"
)


PROJECT_STAGE2_COLLECTION_UPDATE_RATIO = 1.0

PROJECT_STAGE2_COLLECTION_UPDATE_RATIO_SEMANTICS = (
    "GRADIENT_UPDATES_PER_COLLECTED_ENVIRONMENT_TRANSITION"
)

PROJECT_STAGE2_COLLECTION_UPDATE_RATIO_BASIS = (
    "PROJECT_SPECIFIED_TRANSITION_NORMALIZED_OFFPOLICY_CADENCE"
)

PROJECT_STAGE2_COLLECTION_UPDATE_RATIO_REFERENCE = (
    "ZHANG_2025_USES_32_GRADIENT_STEPS_PER_COMPLETED_EPISODE"
)


# Stage-II formal online-collection budget.
#
# IMPORTANT:
#   25,000 is NOT claimed to be a paper-specified training budget.
#
# The paper specifies D_expo replay capacity = 25,000.
# This project deliberately uses one full D_expo-capacity worth of
# newly collected environment transitions as the fixed Stage-II
# budget. Therefore formal Stage-II collection does not require
# replay overwrite before the budget is exhausted.
PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS = 25_000

PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS_SEMANTICS = (
    "EXACT_NEWLY_COLLECTED_ENVIRONMENT_TRANSITIONS"
)

PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS_BASIS = (
    "PROJECT_SPECIFIED_ONE_PAPER_D_EXPO_REPLAY_CAPACITY"
)


# Stage-I formal optimization budget.
#
# IMPORTANT:
#   The paper does NOT specify 782 Stage-I gradient updates.
#
# The paper specifies D_demo replay capacity = 50,000.
# This project deliberately uses one paper D_demo-capacity worth of
# cumulative sampled demonstration transitions as the fixed Stage-I
# optimization exposure.
#
# ReplayBuffer sampling under the frozen Tianshou 2.0.1 environment
# is random with replacement, so "epoch" is intentionally not used
# as the formal Stage-I budget unit.
PROJECT_STAGE1_TARGET_SAMPLED_DEMO_TRANSITIONS = (
    PAPER_DEMO_REPLAY_CAPACITY
)

PROJECT_STAGE1_GRADIENT_STEPS = math.ceil(
    PROJECT_STAGE1_TARGET_SAMPLED_DEMO_TRANSITIONS
    /
    PAPER_BATCH_SIZE
)

PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS = (
    PROJECT_STAGE1_GRADIENT_STEPS
    *
    PAPER_BATCH_SIZE
)

PROJECT_STAGE1_SAMPLING_SEMANTICS = (
    "TIANSHOU_2_0_1_RANDOM_WITH_REPLACEMENT"
)

PROJECT_STAGE1_GRADIENT_STEPS_BASIS = (
    "PROJECT_SPECIFIED_ONE_PAPER_D_DEMO_REPLAY_CAPACITY"
)

PROJECT_STAGE1_STOPPING_RULE = (
    "FIXED_GRADIENT_BUDGET_NO_DATA_DEPENDENT_EARLY_STOP"
)

PROJECT_STAGE1_STOPPING_RULE_SEMANTICS = (
    "STOP_AFTER_EXACT_PROJECT_STAGE1_GRADIENT_STEPS"
)


# ================================================================
# BC-WEIGHT CALIBRATION PROTOCOL
#
# IMPORTANT:
#   The paper does not specify lambda_BC.
#
# The project therefore freezes the complete calibration procedure
# BEFORE any lambda candidate is evaluated.
#
# Dev and blind splits are forbidden for this calibration.
# ================================================================

PROJECT_BC_WEIGHT_CALIBRATION_VERSION = (
    "bc_weight_calibration_v2"
)

PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES = (
    0.1,
    0.3,
    0.5,
    1.0,
    3.0,
)

PROJECT_BC_WEIGHT_CALIBRATION_SEEDS = (
    42,
    43,
    44,
)

PROJECT_BC_WEIGHT_CALIBRATION_SPLIT = (
    "engineering_calibration"
)

PROJECT_BC_WEIGHT_CALIBRATION_MODELS = (
    "bimba",
    "deckel",
    "BracketInches",
    "eraser_ball",
    "cylinder_plate",
)

PROJECT_BC_WEIGHT_CALIBRATION_STAGE1_GRADIENT_STEPS = (
    PROJECT_STAGE1_GRADIENT_STEPS
)

PROJECT_BC_WEIGHT_CALIBRATION_STAGE2_ENABLED = False

PROJECT_BC_WEIGHT_CALIBRATION_EVAL_DETERMINISTIC = True

PROJECT_BC_WEIGHT_CALIBRATION_EVAL_EPSILON = 0.0

# All paired lambda-calibration runs use the same numerical device.
# This is a calibration-only setting and does not yet freeze the
# final formal-training device policy.
PROJECT_BC_WEIGHT_CALIBRATION_DEVICE = (
    "cpu"
)


# Frozen CPU numerical runtime for BC-weight calibration.
#
# This policy was audited on 2026-08-18 using two completely
# independent Python processes with:
#
#     lambda_BC = 2.0  (non-candidate)
#     seed      = 42
#     D_demo    = 30 episodes / 605 transitions
#     updates   = 20
#
# Both processes produced byte-identical audit JSON, identical
# ReplayBuffer minibatches, update statistics, RNG states, and
# final model state.
#
# IMPORTANT:
# torch deterministic-algorithm enforcement remains False because
# that is the exact runtime configuration that was empirically
# audited.  Enabling it here would introduce a new unaudited runtime.
PROJECT_BC_WEIGHT_CALIBRATION_TORCH_NUM_THREADS = 8

PROJECT_BC_WEIGHT_CALIBRATION_TORCH_NUM_INTEROP_THREADS = 8

PROJECT_BC_WEIGHT_CALIBRATION_TORCH_DETERMINISTIC_ALGORITHMS = False

PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_VERSION = (
    "stage1_cpu_repeatability_probe_v1"
)

PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_SEED = 42

PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_BC_WEIGHT = 2.0

PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_UPDATES = 20

PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_BASE_COMMIT = (
    "77f393af7ae2a0a07b86b86c05c2e03f51b3210b"
)

PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_SHA256 = (
    "a3c367167cb8f1f710c4eaea5edf23f"
    "dee50d401e5ca083df5fc82f4cf7f83fe"
)

PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_RESULT = (
    "BITWISE_IDENTICAL_ACROSS_TWO_INDEPENDENT_PROCESSES"
)

PROJECT_BC_WEIGHT_CALIBRATION_DEV_ALLOWED = False

PROJECT_BC_WEIGHT_CALIBRATION_BLIND_ALLOWED = False

PROJECT_BC_WEIGHT_CALIBRATION_NONHEX_METRIC = (
    "SUM_TOTAL_MINUS_HEX_DIV_SUM_TOTAL_FOR_COMPLETED_FINALIZATIONS"
)

PROJECT_BC_WEIGHT_CALIBRATION_SELECTION_RULE = (
    "LEXICOGRAPHIC:"
    "MAX_FULL_HEX,"
    "MIN_FINALIZATION_CRASH,"
    "MIN_AGGREGATE_NONHEX_FRACTION,"
    "MAX_MEAN_EPISODE_RETURN,"
    "MIN_BC_WEIGHT"
)


# ================================================================
# FINAL FORMAL-TRAINING FREEZE
#
# BC weight was selected by the complete frozen
# BC-weight Calibration V2 grid:
#
#     5 lambda candidates
#   x 3 seeds
#   x 5 Engineering5 models
#   = 75 evaluation rows
#
# The committed selector chose lambda_BC = 3.0 by the frozen
# lexicographic rule.
# ================================================================

PROJECT_BC_WEIGHT = 3.0

PROJECT_BC_WEIGHT_BASIS = (
    "FORMAL_BC_WEIGHT_CALIBRATION_V2_SELECTION"
)

PROJECT_BC_WEIGHT_SELECTION_AUDIT_PATH = (
    "data/audits/bc_weight_selection_v1.json"
)

PROJECT_BC_WEIGHT_SELECTION_SHA256 = (
    "50e7814b75fdf1add71dcd101f9d256b1"
    "eadd0406d305bed82da0824c6d79611"
)

PROJECT_BC_WEIGHT_SELECTION_SOURCE_GIT_COMMIT = (
    "8c2ed2887f272d27f19d848dba4cf21e92a9b6d3"
)

PROJECT_BC_WEIGHT_SELECTION_PAIR_ARTIFACTS = 15

PROJECT_BC_WEIGHT_SELECTION_EPISODE_ROWS = 75

PROJECT_BC_WEIGHT_SELECTION_GRID_SHA256 = (
    "fa0a94f24ea528fe72c3543847ddf339"
    "766becb06ff06d12ab3f0a320ad2bf22"
)


# Formal replication seeds.
#
# Reuse the same three replication seeds as the frozen calibration
# rather than introducing a new arbitrary seed set after calibration.
PROJECT_FORMAL_TRAINING_SEEDS = (
    42,
    43,
    44,
)

PROJECT_FORMAL_TRAINING_SEED_BASIS = (
    "REUSE_FORMAL_BC_CALIBRATION_REPLICATION_SEEDS"
)


# Formal numerical runtime.
#
# Reuse the exact CPU runtime that passed the independent-process
# bitwise Stage-I repeatability audit.
PROJECT_FORMAL_TRAINING_DEVICE = "cpu"

PROJECT_FORMAL_TRAINING_TORCH_NUM_THREADS = (
    PROJECT_BC_WEIGHT_CALIBRATION_TORCH_NUM_THREADS
)

PROJECT_FORMAL_TRAINING_TORCH_NUM_INTEROP_THREADS = (
    PROJECT_BC_WEIGHT_CALIBRATION_TORCH_NUM_INTEROP_THREADS
)

PROJECT_FORMAL_TRAINING_TORCH_DETERMINISTIC_ALGORITHMS = (
    PROJECT_BC_WEIGHT_CALIBRATION_TORCH_DETERMINISTIC_ALGORITHMS
)

PROJECT_FORMAL_TRAINING_RUNTIME_BASIS = (
    "BC_WEIGHT_CALIBRATION_CPU_BITWISE_REPEATABILITY_AUDIT"
)


# Stage-II behavior-policy exploration.
PROJECT_STAGE2_EXPLORATION_EPSILON = (
    PAPER_TRAIN_EPSILON_GREEDY
)

PROJECT_STAGE2_COLLECTOR_EXPLORATION_NOISE = True


# Equal replay means the total SAC minibatch remains 64:
#
#     32 D_demo + 32 D_expo
#
PROJECT_STAGE2_SAMPLES_PER_BUFFER = (
    PAPER_BATCH_SIZE // 2
)

PROJECT_STAGE2_EXPO_REPLAY_CAPACITY = (
    PAPER_EXPO_REPLAY_CAPACITY
)


# ================================================================
# FORMAL-TRAINING GATE
# ================================================================

FORMAL_TRAINING_BLOCKERS = ()


def formal_training_ready() -> bool:
    """
    Return True only when Training Protocol V1 contains no unresolved
    formal-training choices.
    """

    return (
        len(
            FORMAL_TRAINING_BLOCKERS
        )
        ==
        0
    )


def assert_formal_training_ready() -> None:
    if formal_training_ready():
        return

    raise RuntimeError(
        "Formal training is blocked until Training Protocol V1 "
        "resolves: "
        +
        ", ".join(
            FORMAL_TRAINING_BLOCKERS
        )
    )
