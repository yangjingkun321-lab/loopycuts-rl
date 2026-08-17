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

PROJECT_MAIN_DEMO_EPISODES = 3

PROJECT_MAIN_DEMO_TRANSITIONS = 29

PROJECT_MAIN_DEMO_MODELS = (
    "mech10",
    "Plate3",
    "bearing_plate",
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
# UNRESOLVED FORMAL-TRAINING ITEMS
#
# None of these may be assigned an arbitrary smoke-test value.
# ================================================================

UNRESOLVED_BC_WEIGHT = None

UNRESOLVED_STAGE1_GRADIENT_STEPS = None

UNRESOLVED_STAGE1_STOPPING_RULE = None

UNRESOLVED_STAGE2_TOTAL_ENVIRONMENT_STEPS = None

UNRESOLVED_STAGE2_COLLECTION_UPDATE_RATIO = None

UNRESOLVED_INITIAL_ALPHA = None


FORMAL_TRAINING_BLOCKERS = (
    "bc_weight",
    "stage1_gradient_steps_or_stopping_rule",
    "stage2_total_environment_steps",
    "stage2_collection_update_ratio",
    "initial_alpha",
)


def formal_training_ready() -> bool:
    """
    Training Protocol V1 is deliberately NOT ready yet.

    This prevents smoke-test hyperparameters from silently becoming
    formal experimental settings.
    """

    return False


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
