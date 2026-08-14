from __future__ import annotations

from dataclasses import dataclass

from rewards.transition_metrics import (
    TransitionMetrics,
)


OUTCOME_FULL_HEX = "FULL_HEX"
OUTCOME_NON_FULL_HEX = "NON_FULL_HEX"
OUTCOME_CRASH = "FINALIZATION_CRASH"

VALID_FINALIZATION_OUTCOMES = {
    OUTCOME_FULL_HEX,
    OUTCOME_NON_FULL_HEX,
    OUTCOME_CRASH,
}


class RewardV2Error(
    ValueError
):
    pass


@dataclass(frozen=True)
class RewardV2Weights:
    """
    Final-aware Reward V2 weights.

    All values are positive magnitudes.

    Signs are assigned by compute_reward_v2():

        step              negative
        tet growth        negative
        revert            negative
        convergence loss  negative
        recovery          positive

        FULL_HEX          positive
        NON_FULL_HEX      negative
        CRASH             negative
    """

    step: float = 1.0
    tet_growth: float = 1.0
    revert: float = 0.10

    convergence_loss: float = 1.0
    convergence_recovery: float = 1.0

    final_full_hex: float = 3.0
    final_non_full_hex: float = 3.0
    final_crash: float = 4.0


#
# Frozen development/calibration default.
#
# Selected after the Phase-2D-D4-A offline audit:
#
#     FULL_HEX            +3
#     NON_FULL_HEX        -3
#     FINALIZATION_CRASH  -4
#
# Conservative (+2/-2/-3) and strong (+4/-4/-5)
# remain reward-scale ablations rather than defaults.
#
DEFAULT_REWARD_V2_WEIGHTS = (
    RewardV2Weights()
)


@dataclass(frozen=True)
class RewardV2Breakdown:
    step: float
    tet_growth: float
    revert: float
    convergence: float
    finalization: float
    total: float


def compute_reward_v2(
    *,
    metrics: TransitionMetrics,
    initial_actionable_count: int,
    finalization_outcome: str | None,
    weights: RewardV2Weights = (
        DEFAULT_REWARD_V2_WEIGHTS
    ),
) -> RewardV2Breakdown:
    """
    Final-aware Reward V2.

    V2 deliberately removes the Selection Reward V1 terminal proxy:

        selection_success
        terminal_failure

    from the reward.

    Instead, the only terminal success/failure signal comes from the
    real FINALIZE_EVAL outcome:

        FULL_HEX
        NON_FULL_HEX
        FINALIZATION_CRASH

    Non-terminal steps must have:

        finalization_outcome = None

    Terminal steps must have exactly one real finalization outcome.
    """

    initial_actionable_count = int(
        initial_actionable_count
    )

    if initial_actionable_count <= 0:
        raise RewardV2Error(
            "initial_actionable_count "
            "must be positive"
        )

    terminal = int(
        metrics.terminal
    )

    if terminal not in (
        0,
        1,
    ):
        raise RewardV2Error(
            f"metrics.terminal must be 0 or 1, "
            f"got {terminal}"
        )

    # ============================================================
    # Finalization-outcome consistency
    # ============================================================

    if terminal:
        if (
            finalization_outcome
            not in
            VALID_FINALIZATION_OUTCOMES
        ):
            raise RewardV2Error(
                "Terminal transition requires one of: "
                + ", ".join(
                    sorted(
                        VALID_FINALIZATION_OUTCOMES
                    )
                )
                +
                f"; got "
                f"{finalization_outcome!r}"
            )

    else:
        if finalization_outcome is not None:
            raise RewardV2Error(
                "Non-terminal transition must not "
                "contain a finalization outcome"
            )

    # ============================================================
    # Dense selection-level components
    #
    # These intentionally preserve Reward V1's already audited
    # local costs.
    # ============================================================

    step_component = (
        - float(
            weights.step
        )
        /
        initial_actionable_count
    )

    tet_component = (
        - float(
            weights.tet_growth
        )
        *
        float(
            metrics.log_tet_growth
        )
    )

    revert_component = (
        - float(
            weights.revert
        )
        *
        int(
            metrics.reverted
        )
    )

    convergence_component = 0.0

    if (
        int(
            metrics.convergence_delta
        )
        ==
        -1
    ):
        convergence_component = (
            - float(
                weights.convergence_loss
            )
        )

    elif (
        int(
            metrics.convergence_delta
        )
        ==
        1
        and
        not int(
            metrics.first_convergence
        )
    ):
        convergence_component = (
            float(
                weights.convergence_recovery
            )
        )

    # ============================================================
    # Real terminal finalization outcome
    # ============================================================

    finalization_component = 0.0

    if terminal:
        if (
            finalization_outcome
            ==
            OUTCOME_FULL_HEX
        ):
            finalization_component = (
                float(
                    weights.final_full_hex
                )
            )

        elif (
            finalization_outcome
            ==
            OUTCOME_NON_FULL_HEX
        ):
            finalization_component = (
                - float(
                    weights.final_non_full_hex
                )
            )

        elif (
            finalization_outcome
            ==
            OUTCOME_CRASH
        ):
            finalization_component = (
                - float(
                    weights.final_crash
                )
            )

        else:
            raise RewardV2Error(
                "Unexpected finalization outcome: "
                f"{finalization_outcome!r}"
            )

    total = (
        step_component
        +
        tet_component
        +
        revert_component
        +
        convergence_component
        +
        finalization_component
    )

    return RewardV2Breakdown(
        step=float(
            step_component
        ),

        tet_growth=float(
            tet_component
        ),

        revert=float(
            revert_component
        ),

        convergence=float(
            convergence_component
        ),

        finalization=float(
            finalization_component
        ),

        total=float(
            total
        ),
    )
