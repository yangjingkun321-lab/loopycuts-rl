from __future__ import annotations

from dataclasses import dataclass

import math

from finalization.terminal_quality_v1 import (
    TerminalQualityFacts,
)

from rewards.reward_v2 import (
    RewardV2Weights,
    compute_reward_v2,
)

from rewards.transition_metrics import (
    TransitionMetrics,
)


REWARD_V5_VERSION = (
    "final_v5_quality_aware_v1"
)


OUTCOME_FULL_HEX = (
    "FULL_HEX"
)

OUTCOME_NON_FULL_HEX = (
    "NON_FULL_HEX"
)

OUTCOME_CRASH = (
    "FINALIZATION_CRASH"
)

OUTCOME_RESOURCE_ABORT = (
    "RESOURCE_ABORT"
)


class RewardV5Error(
    ValueError
):
    pass


@dataclass(
    frozen=True
)
class RewardV5Weights:
    """
    Reward V5 preserves the existing V3/V2 dense shaping.

    Dense components:
        step                 = -1 / N0
        tet growth           = -log(T_after / T_before)
        reverted cut         = -0.10
        convergence loss     = -1
        later recovery       = +1

    Successful terminal:
        U = D_C * Q_fidelity
        R_quality = 6 * U - 3

    The old FULL_HEX=+3 / NON_FULL_HEX=-3 distinction is removed.

    FINALIZATION_CRASH keeps the existing -4 terminal penalty.
    RESOURCE_ABORT keeps the existing exact -4 override.
    """

    step: float = 1.0
    tet_growth: float = 1.0
    revert: float = 0.10

    convergence_loss: float = 1.0
    convergence_recovery: float = 1.0

    terminal_scale: float = 6.0
    terminal_bias: float = -3.0

    final_crash: float = 4.0
    final_resource_abort: float = 4.0


DEFAULT_REWARD_V5_WEIGHTS = (
    RewardV5Weights()
)


@dataclass(
    frozen=True
)
class RewardV5Breakdown:
    step: float
    tet_growth: float
    revert: float
    convergence: float

    quality_available: bool
    utility: float

    terminal: float
    total: float


def _as_v2_weights(
    weights: RewardV5Weights,
    *,
    crash_penalty: float,
) -> RewardV2Weights:
    """
    Reuse the frozen Reward V2 dense implementation.

    FULL/NON_FULL terminal weights are deliberately zero here because
    V5 supplies the quality-aware successful-terminal component itself.
    """

    return RewardV2Weights(
        step=
            float(
                weights.step
            ),

        tet_growth=
            float(
                weights.tet_growth
            ),

        revert=
            float(
                weights.revert
            ),

        convergence_loss=
            float(
                weights.convergence_loss
            ),

        convergence_recovery=
            float(
                weights.convergence_recovery
            ),

        final_full_hex=
            0.0,

        final_non_full_hex=
            0.0,

        final_crash=
            float(
                crash_penalty
            ),
    )


def compute_reward_v5(
    *,
    metrics: TransitionMetrics | None,
    initial_actionable_count: int,
    terminal_outcome: str | None,
    terminal_quality: TerminalQualityFacts | None,
    resource_abort: bool,
    weights: RewardV5Weights = (
        DEFAULT_REWARD_V5_WEIGHTS
    ),
) -> RewardV5Breakdown:
    """
    Compute Reward V5.

    Ordinary transition:
        preserve Reward V2/V3 dense components exactly.

    Successful FULL_HEX/NON_FULL_HEX:
        preserve dense components and replace the old +/-3 terminal
        class reward with 6 * D_C * Q_fidelity - 3.

    FINALIZATION_CRASH:
        preserve the existing real STEP dense components and the
        existing -4 crash terminal penalty.

    RESOURCE_ABORT:
        preserve V3 exact override semantics: total reward = -4,
        with no fabricated geometric components.
    """

    initial_actionable_count = int(
        initial_actionable_count
    )

    if initial_actionable_count <= 0:
        raise RewardV5Error(
            "initial_actionable_count must be positive"
        )

    resource_abort = bool(
        resource_abort
    )

    # ============================================================
    # RESOURCE_ABORT: exact V3 override.
    # ============================================================

    if resource_abort:
        if (
            terminal_outcome
            !=
            OUTCOME_RESOURCE_ABORT
        ):
            raise RewardV5Error(
                "resource_abort=True requires "
                "terminal_outcome='RESOURCE_ABORT'"
            )

        if metrics is not None:
            raise RewardV5Error(
                "RESOURCE_ABORT must not fabricate "
                "TransitionMetrics"
            )

        if terminal_quality is not None:
            raise RewardV5Error(
                "RESOURCE_ABORT must not carry terminal quality"
            )

        terminal_component = (
            -float(
                weights.final_resource_abort
            )
        )

        return RewardV5Breakdown(
            step=
                0.0,

            tet_growth=
                0.0,

            revert=
                0.0,

            convergence=
                0.0,

            quality_available=
                False,

            utility=
                0.0,

            terminal=
                terminal_component,

            total=
                terminal_component,
        )

    if (
        terminal_outcome
        ==
        OUTCOME_RESOURCE_ABORT
    ):
        raise RewardV5Error(
            "RESOURCE_ABORT requires resource_abort=True"
        )

    if metrics is None:
        raise RewardV5Error(
            "Non-resource Reward V5 transition "
            "requires real TransitionMetrics"
        )

    # ============================================================
    # FINALIZATION_CRASH:
    #
    # Keep the old V3/V2 dense STEP terms and the existing
    # -4 crash terminal component.
    # ============================================================

    if (
        terminal_outcome
        ==
        OUTCOME_CRASH
    ):
        if terminal_quality is not None:
            raise RewardV5Error(
                "FINALIZATION_CRASH must not carry "
                "terminal quality"
            )

        base = compute_reward_v2(
            metrics=
                metrics,

            initial_actionable_count=
                initial_actionable_count,

            finalization_outcome=
                OUTCOME_CRASH,

            weights=
                _as_v2_weights(
                    weights,
                    crash_penalty=
                        float(
                            weights.final_crash
                        ),
                ),
        )

        return RewardV5Breakdown(
            step=
                float(
                    base.step
                ),

            tet_growth=
                float(
                    base.tet_growth
                ),

            revert=
                float(
                    base.revert
                ),

            convergence=
                float(
                    base.convergence
                ),

            quality_available=
                False,

            utility=
                0.0,

            terminal=
                float(
                    base.finalization
                ),

            total=
                float(
                    base.total
                ),
        )

    if (
        terminal_outcome
        not in {
            None,
            OUTCOME_FULL_HEX,
            OUTCOME_NON_FULL_HEX,
        }
    ):
        raise RewardV5Error(
            "Unknown terminal outcome: "
            f"{terminal_outcome!r}"
        )

    # ============================================================
    # Frozen V2/V3 dense reward.
    #
    # Successful FULL/NON_FULL weights are zeroed so only the
    # old terminal class term is removed; all dense components
    # remain exactly the current implementation.
    # ============================================================

    base = compute_reward_v2(
        metrics=
            metrics,

        initial_actionable_count=
            initial_actionable_count,

        finalization_outcome=
            terminal_outcome,

        weights=
            _as_v2_weights(
                weights,
                crash_penalty=
                    0.0,
            ),
    )

    if terminal_outcome is None:
        if terminal_quality is not None:
            raise RewardV5Error(
                "Non-terminal transition must not carry "
                "terminal quality"
            )

        return RewardV5Breakdown(
            step=
                float(
                    base.step
                ),

            tet_growth=
                float(
                    base.tet_growth
                ),

            revert=
                float(
                    base.revert
                ),

            convergence=
                float(
                    base.convergence
                ),

            quality_available=
                False,

            utility=
                0.0,

            terminal=
                0.0,

            total=
                float(
                    base.total
                ),
        )

    if not isinstance(
        terminal_quality,
        TerminalQualityFacts,
    ):
        raise RewardV5Error(
            "Successful finalization requires "
            "TerminalQualityFacts"
        )

    if (
        terminal_outcome
        ==
        OUTCOME_FULL_HEX
    ):
        if (
            terminal_quality.hex
            !=
            terminal_quality.total_polys
        ):
            raise RewardV5Error(
                "FULL_HEX quality facts have "
                "hex != total_polys"
            )

    else:
        if (
            terminal_quality.hex
            >=
            terminal_quality.total_polys
        ):
            raise RewardV5Error(
                "NON_FULL_HEX quality facts have "
                "hex >= total_polys"
            )

    utility = (
        float(
            terminal_quality.d_c
        )
        *
        float(
            terminal_quality.q_fidelity
        )
    )

    if (
        not math.isfinite(
            utility
        )
        or
        not (
            0.0
            <=
            utility
            <=
            1.0
        )
    ):
        raise RewardV5Error(
            "terminal utility must be finite and in [0, 1]"
        )

    terminal_component = (
        float(
            weights.terminal_scale
        )
        *
        utility
        +
        float(
            weights.terminal_bias
        )
    )

    total = (
        float(
            base.total
        )
        +
        terminal_component
    )

    return RewardV5Breakdown(
        step=
            float(
                base.step
            ),

        tet_growth=
            float(
                base.tet_growth
            ),

        revert=
            float(
                base.revert
            ),

        convergence=
            float(
                base.convergence
            ),

        quality_available=
            True,

        utility=
            float(
                utility
            ),

        terminal=
            float(
                terminal_component
            ),

        total=
            float(
                total
            ),
    )
