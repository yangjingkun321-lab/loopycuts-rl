from __future__ import annotations

from dataclasses import fields

import gymnasium as gym


from rewards.reward_v3 import (
    DEFAULT_REWARD_V3_WEIGHTS,
    OUTCOME_RESOURCE_ABORT,
    REWARD_V3_VERSION,
    RewardV3Weights,
    compute_reward_v3,
)

from rewards.transition_metrics import (
    TransitionMetrics,
)


class FinalRewardWrapperV3(
    gym.Wrapper
):
    """
    Resource-aware Reward V3 wrapper.

    Expected order:

        FinalRewardWrapperV3(
            FinalizationEvalWrapper(
                LoopyCutsEnv(...)
            )
        )

    Normal LoopyCuts transitions:
        numerically identical to Reward V2.

    ResourceGuard terminal:
        RESOURCE_ABORT
        reward = -4
        no fabricated geometric TransitionMetrics.
    """

    def __init__(
        self,
        env,
        weights: RewardV3Weights = (
            DEFAULT_REWARD_V3_WEIGHTS
        ),
    ):
        super().__init__(
            env
        )

        self.weights = weights

        self._initial_actionable_count = None


    # ------------------------------------------------------------------

    @staticmethod
    def _metrics_from_dict(
        data,
    ):
        if not isinstance(
            data,
            dict,
        ):
            raise RuntimeError(
                "info['transition_metrics'] "
                "must be a dict"
            )

        names = {
            field.name
            for field in fields(
                TransitionMetrics
            )
        }

        missing = (
            names
            -
            set(
                data
            )
        )

        if missing:
            raise RuntimeError(
                "transition_metrics is missing fields: "
                +
                ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        return TransitionMetrics(
            **{
                name:
                    data[
                        name
                    ]
                for name in names
            }
        )


    # ------------------------------------------------------------------

    @staticmethod
    def _collector_finalization_record(
        outcome_record,
    ):
        """
        Fixed scalar schema for Tianshou ReplayBuffer.
        """

        if outcome_record is None:
            return {
                "outcome":
                    "NONE",

                "outcome_code":
                    0,

                "attempted":
                    False,

                "completed":
                    False,

                "crashed":
                    False,

                "return_code":
                    0,

                "signal_number":
                    0,

                "signal_name":
                    "",

                "final_hex":
                    -1,

                "final_total_polys":
                    -1,

                "full_hex":
                    -1,
            }

        if not isinstance(
            outcome_record,
            dict,
        ):
            raise RuntimeError(
                "finalization_outcome must be "
                "None or a dict"
            )

        if (
            "outcome"
            not in outcome_record
        ):
            raise RuntimeError(
                "finalization_outcome is missing "
                "'outcome'"
            )

        outcome = str(
            outcome_record[
                "outcome"
            ]
        )

        outcome_codes = {
            "FULL_HEX":
                1,

            "NON_FULL_HEX":
                2,

            "FINALIZATION_CRASH":
                3,

            "RESOURCE_ABORT":
                4,
        }

        if outcome not in outcome_codes:
            raise RuntimeError(
                "Unknown terminal outcome: "
                f"{outcome!r}"
            )

        def int_or_default(
            key,
            default,
        ):
            value = (
                outcome_record.get(
                    key
                )
            )

            if value is None:
                return int(
                    default
                )

            return int(
                value
            )

        signal_name = (
            outcome_record.get(
                "signal_name"
            )
        )

        if signal_name is None:
            signal_name = ""

        # RESOURCE_ABORT explicitly skipped FINALIZE_EVAL.
        attempted = (
            outcome
            !=
            OUTCOME_RESOURCE_ABORT
        )

        return {
            "outcome":
                outcome,

            "outcome_code":
                outcome_codes[
                    outcome
                ],

            "attempted":
                bool(
                    attempted
                ),

            "completed":
                bool(
                    outcome_record.get(
                        "completed",
                        False,
                    )
                ),

            "crashed":
                bool(
                    outcome_record.get(
                        "crashed",
                        False,
                    )
                ),

            "return_code":
                int_or_default(
                    "return_code",
                    0,
                ),

            "signal_number":
                int_or_default(
                    "signal_number",
                    0,
                ),

            "signal_name":
                str(
                    signal_name
                ),

            "final_hex":
                int_or_default(
                    "final_hex",
                    -1,
                ),

            "final_total_polys":
                int_or_default(
                    "final_total_polys",
                    -1,
                ),

            "full_hex":
                int_or_default(
                    "full_hex",
                    -1,
                ),
        }


    # ------------------------------------------------------------------

    @staticmethod
    def _collector_resource_guard_record(
        rich_record,
    ):
        """
        Fixed scalar resource telemetry schema on EVERY transition.

        This prevents Tianshou from synthesizing NaNs because the
        rich resource_abort dictionary only exists on abort steps.
        """

        if rich_record is None:
            return {
                "triggered":
                    False,

                "phase":
                    "",

                "guard_state":
                    "",

                "action":
                    -1,

                "swap_used_bytes":
                    0,

                "swap_total_bytes":
                    0,

                "swap_free_bytes":
                    0,

                "mem_available_bytes":
                    0,

                "python_rss_bytes":
                    0,

                "python_swap_bytes":
                    0,

                "cpp_rss_bytes":
                    0,

                "cpp_swap_bytes":
                    0,
            }

        if not isinstance(
            rich_record,
            dict,
        ):
            raise RuntimeError(
                "resource_abort must be a dict"
            )

        if (
            rich_record.get(
                "outcome"
            )
            !=
            OUTCOME_RESOURCE_ABORT
        ):
            raise RuntimeError(
                "Unknown resource_abort outcome"
            )

        return {
            "triggered":
                True,

            "phase":
                str(
                    rich_record.get(
                        "phase",
                        "",
                    )
                ),

            "guard_state":
                str(
                    rich_record.get(
                        "guard_state",
                        "",
                    )
                ),

            "action":
                int(
                    rich_record.get(
                        "action",
                        -1,
                    )
                ),

            "swap_used_bytes":
                int(
                    rich_record.get(
                        "swap_used_bytes",
                        0,
                    )
                ),

            "swap_total_bytes":
                int(
                    rich_record.get(
                        "swap_total_bytes",
                        0,
                    )
                ),

            "swap_free_bytes":
                int(
                    rich_record.get(
                        "swap_free_bytes",
                        0,
                    )
                ),

            "mem_available_bytes":
                int(
                    rich_record.get(
                        "mem_available_bytes",
                        0,
                    )
                ),

            "python_rss_bytes":
                int(
                    rich_record.get(
                        "python_rss_bytes",
                        0,
                    )
                ),

            "python_swap_bytes":
                int(
                    rich_record.get(
                        "python_swap_bytes",
                        0,
                    )
                ),

            "cpp_rss_bytes":
                int(
                    rich_record.get(
                        "cpp_rss_bytes",
                        0,
                    )
                ),

            "cpp_swap_bytes":
                int(
                    rich_record.get(
                        "cpp_swap_bytes",
                        0,
                    )
                ),
        }


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
        ] = REWARD_V3_VERSION

        info[
            "selection_reward_version"
        ] = "selection_v1"

        info[
            "selection_reward_available"
        ] = False

        info[
            "finalization_outcome"
        ] = (
            self._collector_finalization_record(
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

        # Never expose the sparse rich record to Tianshou.
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
                "FinalRewardWrapperV3.step() called "
                "before reset()"
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
                "Reward V3 does not define artificial "
                "truncation semantics"
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

        terminal_outcome = None

        if terminated:
            if not isinstance(
                outcome_record,
                dict,
            ):
                raise RuntimeError(
                    "Terminal Reward V3 transition "
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
            #
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
                        "STEP RESOURCE_ABORT inner reward must be "
                        "the zero placeholder"
                    )

                selection_reward_available = False
                selection_reward_v1 = 0.0

            # --------------------------------------------------------
            # FINALIZE_EVAL abort:
            #
            # the final selection STEP completed normally, therefore
            # its real transition_metrics and Selection Reward V1
            # remain valid audit data.
            #
            # Nevertheless Reward V3 for the terminal RL transition is
            # overridden to exact RESOURCE_ABORT = -4.
            # --------------------------------------------------------

            elif (
                resource_abort_phase
                ==
                "FINALIZE_EVAL"
            ):
                if (
                    "transition_metrics"
                    not in info
                ):
                    raise RuntimeError(
                        "FINALIZE_EVAL RESOURCE_ABORT must preserve "
                        "the genuine final STEP transition_metrics"
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

            # RESOURCE_ABORT deliberately receives no dense geometric
            # shaping, regardless of whether it originated in STEP or
            # in FINALIZE_EVAL.
            breakdown = (
                compute_reward_v3(
                    metrics=
                        None,

                    initial_actionable_count=
                        self._initial_actionable_count,

                    terminal_outcome=
                        terminal_outcome,

                    resource_abort=
                        True,

                    weights=
                        self.weights,
                )
            )


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

            breakdown = (
                compute_reward_v3(
                    metrics=
                        metrics,

                    initial_actionable_count=
                        self._initial_actionable_count,

                    terminal_outcome=
                        terminal_outcome,

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


        reward_v3 = float(
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
        ] = REWARD_V3_VERSION

        info[
            "reward_v3_breakdown"
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

        # Replace rich/sparse records with fixed collector schemas.
        info.pop(
            "resource_abort",
            None,
        )

        info[
            "finalization_outcome"
        ] = (
            self._collector_finalization_record(
                outcome_record
            )
        )

        return (
            observation,
            reward_v3,
            terminated,
            truncated,
            info,
        )
