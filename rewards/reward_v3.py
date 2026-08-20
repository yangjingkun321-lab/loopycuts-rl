from __future__ import annotations

from dataclasses import dataclass

from rewards.reward_v2 import (
    RewardV2Weights,
    compute_reward_v2,
)

from rewards.transition_metrics import (
    TransitionMetrics,
)


REWARD_V3_VERSION = (
    "final_v3_resource_guard"
)


OUTCOME_FULL_HEX = "FULL_HEX"
OUTCOME_NON_FULL_HEX = "NON_FULL_HEX"
OUTCOME_CRASH = "FINALIZATION_CRASH"
OUTCOME_RESOURCE_ABORT = "RESOURCE_ABORT"


class RewardV3Error(
    ValueError
):
    pass


@dataclass(frozen=True)
class RewardV3Weights:
    """
    Reward V3 is a strict extension of Reward V2.

    For every ordinary LoopyCuts transition:

        Reward V3 == Reward V2

    The only new terminal outcome is:

        RESOURCE_ABORT = -4

    No dense memory shaping is introduced.
    """

    step: float = 1.0
    tet_growth: float = 1.0
    revert: float = 0.10

    convergence_loss: float = 1.0
    convergence_recovery: float = 1.0

    final_full_hex: float = 3.0
    final_non_full_hex: float = 3.0
    final_crash: float = 4.0

    final_resource_abort: float = 4.0


DEFAULT_REWARD_V3_WEIGHTS = (
    RewardV3Weights()
)


@dataclass(frozen=True)
class RewardV3Breakdown:
    step: float
    tet_growth: float
    revert: float
    convergence: float

    # "terminal" is deliberately broader than V2's
    # "finalization" because RESOURCE_ABORT never enters
    # FINALIZE_EVAL.
    terminal: float

    total: float


def _as_v2_weights(
    weights: RewardV3Weights,
) -> RewardV2Weights:
    return RewardV2Weights(
        step=float(
            weights.step
        ),

        tet_growth=float(
            weights.tet_growth
        ),

        revert=float(
            weights.revert
        ),

        convergence_loss=float(
            weights.convergence_loss
        ),

        convergence_recovery=float(
            weights.convergence_recovery
        ),

        final_full_hex=float(
            weights.final_full_hex
        ),

        final_non_full_hex=float(
            weights.final_non_full_hex
        ),

        final_crash=float(
            weights.final_crash
        ),
    )


def compute_reward_v3(
    *,
    metrics: TransitionMetrics | None,
    initial_actionable_count: int,
    terminal_outcome: str | None,
    resource_abort: bool,
    weights: RewardV3Weights = (
        DEFAULT_REWARD_V3_WEIGHTS
    ),
) -> RewardV3Breakdown:
    """
    Compute Reward V3.

    Ordinary transition:
        delegates exactly to Reward V2.

    RESOURCE_ABORT:
        C++ did not return a complete post-STEP geometry state,
        therefore geometric dense terms cannot be measured and are
        deliberately set to zero.

        The attempted action receives only:

            -final_resource_abort
    """

    initial_actionable_count = int(
        initial_actionable_count
    )

    if initial_actionable_count <= 0:
        raise RewardV3Error(
            "initial_actionable_count must be positive"
        )

    resource_abort = bool(
        resource_abort
    )

    if resource_abort:
        if (
            terminal_outcome
            !=
            OUTCOME_RESOURCE_ABORT
        ):
            raise RewardV3Error(
                "resource_abort=True requires "
                "terminal_outcome='RESOURCE_ABORT'"
            )

        if metrics is not None:
            raise RewardV3Error(
                "RESOURCE_ABORT must not fabricate "
                "TransitionMetrics"
            )

        terminal_component = (
            -float(
                weights.final_resource_abort
            )
        )

        return RewardV3Breakdown(
            step=0.0,
            tet_growth=0.0,
            revert=0.0,
            convergence=0.0,
            terminal=terminal_component,
            total=terminal_component,
        )

    if metrics is None:
        raise RewardV3Error(
            "Ordinary Reward V3 transition requires "
            "real TransitionMetrics"
        )

    if (
        terminal_outcome
        ==
        OUTCOME_RESOURCE_ABORT
    ):
        raise RewardV3Error(
            "RESOURCE_ABORT requires resource_abort=True"
        )

    base = compute_reward_v2(
        metrics=
            metrics,

        initial_actionable_count=
            initial_actionable_count,

        finalization_outcome=
            terminal_outcome,

        weights=
            _as_v2_weights(
                weights
            ),
    )

    return RewardV3Breakdown(
        step=float(
            base.step
        ),

        tet_growth=float(
            base.tet_growth
        ),

        revert=float(
            base.revert
        ),

        convergence=float(
            base.convergence
        ),

        terminal=float(
            base.finalization
        ),

        total=float(
            base.total
        ),
    )
