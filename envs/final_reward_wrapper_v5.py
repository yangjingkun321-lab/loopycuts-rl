from __future__ import annotations

import gymnasium as gym


from envs.final_reward_wrapper_v3 import (
    FinalRewardWrapperV3,
)

from finalization.outcome import (
    OUTCOME_RESOURCE_ABORT,
)

from finalization.terminal_quality_v1 import (
    parse_terminal_quality_facts,
)

from rewards.reward_v5 import (
    DEFAULT_REWARD_V5_WEIGHTS,
    REWARD_V5_VERSION,
    RewardV5Weights,
    compute_reward_v5,
)


class FinalRewardWrapperV5(
    FinalRewardWrapperV3
):
    """
    Quality-aware Reward V5.

    Dense reward remains bit-exact Reward V3/V2:

        step
        tet_growth
        revert
        convergence

    Only successful finalization terminal semantics change:

        old:
            FULL_HEX      +3
            NON_FULL_HEX  -3

        V5:
            U = D_C * Q_fidelity
            terminal = 6 U - 3

    RESOURCE_ABORT remains exact -4 override.
    """

    def __init__(
        self,
        env,
        weights: RewardV5Weights = (
            DEFAULT_REWARD_V5_WEIGHTS
        ),
    ):
        #
        # Initialize gym.Wrapper directly rather than feeding V5 weights
        # through FinalRewardWrapperV3's RewardV3Weights annotation.
        #
        gym.Wrapper.__init__(
            self,
            env,
        )

        self.weights = (
            weights
        )

        self._initial_actionable_count = None


    # ------------------------------------------------------------------

    @staticmethod
    def _collector_terminal_quality_record(
        quality_record,
    ):
        """
        Fixed scalar Tianshou schema on every transition.
        """

        if quality_record is None:
            return {
                "available":
                    False,

                "model":
                    "",

                "hex":
                    -1,

                "total_polys":
                    -1,

                "nonhex":
                    -1,

                "d_c":
                    0.0,

                "q_missing":
                    0.0,

                "q_spurious":
                    0.0,

                "q_shape":
                    0.0,

                "sharp_active":
                    0,

                "sharp_metrics_valid":
                    0,

                "q_sharp_available":
                    False,

                "q_sharp":
                    0.0,

                "q_fidelity":
                    0.0,

                "utility":
                    0.0,
            }

        facts = (
            parse_terminal_quality_facts(
                quality_record
            )
        )

        return {
            "available":
                True,

            "model":
                str(
                    facts.model
                ),

            "hex":
                int(
                    facts.hex
                ),

            "total_polys":
                int(
                    facts.total_polys
                ),

            "nonhex":
                int(
                    facts.nonhex
                ),

            "d_c":
                float(
                    facts.d_c
                ),

            "q_missing":
                float(
                    facts.q_missing
                ),

            "q_spurious":
                float(
                    facts.q_spurious
                ),

            "q_shape":
                float(
                    facts.q_shape
                ),

            "sharp_active":
                int(
                    facts.sharp_active
                ),

            "sharp_metrics_valid":
                int(
                    facts.sharp_metrics_valid
                ),

            "q_sharp_available":
                (
                    facts.q_sharp
                    is not None
                ),

            "q_sharp":
                (
                    0.0
                    if facts.q_sharp is None
                    else
                    float(
                        facts.q_sharp
                    )
                ),

            "q_fidelity":
                float(
                    facts.q_fidelity
                ),

            "utility":
                float(
                    facts.utility
                ),
        }


    # ------------------------------------------------------------------

    @staticmethod
    def _collector_finalization_record_v5(
        outcome_record,
        *,
        attempted,
    ):
        record = (
            FinalRewardWrapperV3
            ._collector_finalization_record(
                outcome_record
            )
        )

        if (
            outcome_record is None
            and
            attempted
        ):
            raise RuntimeError(
                "finalization_attempted=True "
                "without terminal outcome"
            )

        record[
            "attempted"
        ] = bool(
            attempted
        )

        return record


    # ------------------------------------------------------------------

    def reset(
        self,
        **kwargs,
    ):
        observation, info = (
            self.env.reset(
                **kwargs
            )
        )

        initial_actions = (
            self.unwrapped.legal_actions
        )

        self._initial_actionable_count = (
            len(
                initial_actions
            )
        )

        if (
            self._initial_actionable_count
            <=
            0
        ):
            raise RuntimeError(
                "Reset produced no initial legal actions"
            )

        info = dict(
            info
        )

        info[
            "reward_version"
        ] = REWARD_V5_VERSION

        info[
            "selection_reward_version"
        ] = "selection_v1"

        info[
            "selection_reward_available"
        ] = False

        info[
            "reward_v5_breakdown"
        ] = {
            "step":
                0.0,

            "tet_growth":
                0.0,

            "revert":
                0.0,

            "convergence":
                0.0,

            "quality_available":
                False,

            "utility":
                0.0,

            "terminal":
                0.0,

            "total":
                0.0,
        }

        info[
            "finalization_outcome"
        ] = (
            self._collector_finalization_record_v5(
                None,
                attempted=
                    False,
            )
        )

        info[
            "terminal_quality"
        ] = (
            self._collector_terminal_quality_record(
                None
            )
        )

        info[
            "resource_guard"
        ] = (
            self._collector_resource_guard_record(
                None
            )
        )

        info.pop(
            "resource_abort",
            None,
        )

        return (
            observation,
            info,
        )


    # ------------------------------------------------------------------

    def step(
        self,
        action,
    ):
        if (
            self._initial_actionable_count
            is None
        ):
            raise RuntimeError(
                "FinalRewardWrapperV5.step() "
                "called before reset()"
            )

        (
            observation,
            selection_reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(
            action
        )

        if truncated:
            raise RuntimeError(
                "Reward V5 does not define "
                "artificial truncation semantics"
            )

        info = dict(
            info
        )

        rich_resource_abort = (
            info.get(
                "resource_abort"
            )
        )

        resource_abort = (
            rich_resource_abort
            is not None
        )

        outcome_record = (
            info.get(
                "finalization_outcome"
            )
        )

        quality_record = (
            info.get(
                "terminal_quality"
            )
        )

        finalization_attempted = bool(
            info.get(
                "finalization_attempted",
                False,
            )
        )

        terminal_outcome = None

        if terminated:
            if not isinstance(
                outcome_record,
                dict,
            ):
                raise RuntimeError(
                    "Terminal Reward V5 transition "
                    "requires a terminal outcome"
                )

            if (
                "outcome"
                not in outcome_record
            ):
                raise RuntimeError(
                    "Terminal outcome is missing "
                    "'outcome'"
                )

            terminal_outcome = str(
                outcome_record[
                    "outcome"
                ]
            )

        else:
            if outcome_record is not None:
                raise RuntimeError(
                    "Non-terminal transition unexpectedly "
                    "contains terminal outcome"
                )

            if quality_record is not None:
                raise RuntimeError(
                    "Non-terminal transition unexpectedly "
                    "contains terminal quality"
                )

            if finalization_attempted:
                raise RuntimeError(
                    "Non-terminal transition cannot have "
                    "finalization_attempted=True"
                )


        # ============================================================
        # RESOURCE_ABORT
        # ============================================================

        if resource_abort:
            if not terminated:
                raise RuntimeError(
                    "RESOURCE_ABORT must terminate episode"
                )

            if (
                terminal_outcome
                !=
                OUTCOME_RESOURCE_ABORT
            ):
                raise RuntimeError(
                    "resource_abort/finalization_outcome mismatch"
                )

            if quality_record is not None:
                raise RuntimeError(
                    "RESOURCE_ABORT must not carry terminal quality"
                )

            if not isinstance(
                rich_resource_abort,
                dict,
            ):
                raise RuntimeError(
                    "resource_abort must be a dict"
                )

            resource_abort_phase = str(
                rich_resource_abort.get(
                    "phase",
                    "",
                )
            )

            # --------------------------------------------------------
            # STEP abort:
            # no complete post-STEP geometry exists.
            # --------------------------------------------------------

            if (
                resource_abort_phase
                ==
                "STEP"
            ):
                if (
                    "transition_metrics"
                    in info
                ):
                    raise RuntimeError(
                        "STEP RESOURCE_ABORT must not fabricate "
                        "transition_metrics"
                    )

                if (
                    float(
                        selection_reward
                    )
                    !=
                    0.0
                ):
                    raise RuntimeError(
                        "STEP RESOURCE_ABORT inner reward "
                        "must be zero placeholder"
                    )

                if finalization_attempted:
                    raise RuntimeError(
                        "STEP RESOURCE_ABORT must not attempt "
                        "FINALIZE_QUALITY"
                    )

                selection_reward_available = False
                selection_reward_v1 = 0.0

            # --------------------------------------------------------
            # FINALIZE_QUALITY abort:
            #
            # terminal Stage-2 STEP completed normally, but the
            # independent finalization swap guard fired.
            # --------------------------------------------------------

            elif (
                resource_abort_phase
                ==
                "FINALIZE_QUALITY"
            ):
                if (
                    "transition_metrics"
                    not in info
                ):
                    raise RuntimeError(
                        "FINALIZE_QUALITY RESOURCE_ABORT "
                        "must preserve final STEP metrics"
                    )

                if not finalization_attempted:
                    raise RuntimeError(
                        "FINALIZE_QUALITY RESOURCE_ABORT "
                        "must mark finalization attempted"
                    )

                selection_reward_available = True

                selection_reward_v1 = float(
                    selection_reward
                )

            else:
                raise RuntimeError(
                    "Unknown RESOURCE_ABORT phase: "
                    f"{resource_abort_phase!r}"
                )

            #
            # Preserve frozen exact V3 RESOURCE_ABORT semantics.
            #
            breakdown = (
                compute_reward_v5(
                    metrics=
                        None,

                    initial_actionable_count=
                        self._initial_actionable_count,

                    terminal_outcome=
                        terminal_outcome,

                    terminal_quality=
                        None,

                    resource_abort=
                        True,

                    weights=
                        self.weights,
                )
            )

            terminal_quality = None


        # ============================================================
        # Ordinary LoopyCuts transition
        # ============================================================

        else:
            if (
                "transition_metrics"
                not in info
            ):
                raise RuntimeError(
                    "Ordinary transition lacks "
                    "transition_metrics"
                )

            metrics = (
                self._metrics_from_dict(
                    info[
                        "transition_metrics"
                    ]
                )
            )

            terminal_quality = None

            if quality_record is not None:
                terminal_quality = (
                    parse_terminal_quality_facts(
                        quality_record
                    )
                )

            breakdown = (
                compute_reward_v5(
                    metrics=
                        metrics,

                    initial_actionable_count=
                        self._initial_actionable_count,

                    terminal_outcome=
                        terminal_outcome,

                    terminal_quality=
                        terminal_quality,

                    resource_abort=
                        False,

                    weights=
                        self.weights,
                )
            )

            selection_reward_available = True

            selection_reward_v1 = float(
                selection_reward
            )


        reward_v5 = float(
            breakdown.total
        )

        info[
            "selection_reward_v1"
        ] = float(
            selection_reward_v1
        )

        info[
            "selection_reward_available"
        ] = bool(
            selection_reward_available
        )

        info[
            "reward_version"
        ] = REWARD_V5_VERSION

        info[
            "reward_v5_breakdown"
        ] = {
            "step":
                float(
                    breakdown.step
                ),

            "tet_growth":
                float(
                    breakdown.tet_growth
                ),

            "revert":
                float(
                    breakdown.revert
                ),

            "convergence":
                float(
                    breakdown.convergence
                ),

            "quality_available":
                bool(
                    breakdown.quality_available
                ),

            "utility":
                float(
                    breakdown.utility
                ),

            "terminal":
                float(
                    breakdown.terminal
                ),

            "total":
                float(
                    breakdown.total
                ),
        }

        info[
            "resource_guard"
        ] = (
            self._collector_resource_guard_record(
                rich_resource_abort
            )
        )

        info.pop(
            "resource_abort",
            None,
        )

        info[
            "finalization_outcome"
        ] = (
            self._collector_finalization_record_v5(
                outcome_record,
                attempted=
                    finalization_attempted,
            )
        )

        info[
            "terminal_quality"
        ] = (
            self._collector_terminal_quality_record(
                quality_record
            )
        )

        return (
            observation,
            reward_v5,
            terminated,
            truncated,
            info,
        )
