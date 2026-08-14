from __future__ import annotations

from dataclasses import fields

import gymnasium as gym


from rewards.reward_v2 import (
    DEFAULT_REWARD_V2_WEIGHTS,
    RewardV2Weights,
    compute_reward_v2,
)

from rewards.transition_metrics import (
    TransitionMetrics,
)


class FinalRewardWrapper(
    gym.Wrapper
):
    """
    Convert the wrapped environment's Selection Reward V1 into
    Final-aware Reward V2.

    Expected wrapping order:

        FinalRewardWrapper(
            FinalizationEvalWrapper(
                LoopyCutsEnv(...)
            )
        )

    Responsibilities:

        LoopyCutsEnv:
            real Stage-2 transition
            Observation V1
            Selection Reward V1

        FinalizationEvalWrapper:
            real terminal FINALIZE_EVAL outcome

        FinalRewardWrapper:
            recompute the reward as Reward V2

    Reward V2 deliberately removes Selection Reward V1's
    selection_success / terminal_failure proxy terminal reward.
    """

    def __init__(
        self,
        env,
        weights: RewardV2Weights = (
            DEFAULT_REWARD_V2_WEIGHTS
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
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        kwargs = {
            name:
                data[
                    name
                ]
            for name in names
        }

        return TransitionMetrics(
            **kwargs
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _collector_finalization_record(
        outcome_record,
    ):
        """
        Convert the rich finalization result into a fixed-schema,
        NaN-free record suitable for Tianshou Batch/ReplayBuffer.

        Why this is needed:

            non-terminal:
                FinalizationEvalWrapper exposes
                finalization_outcome = None

            terminal:
                it exposes a rich dict containing optional/nested
                final_result, final_state and log_tail fields.

        Tianshou stacks info dictionaries across transitions.
        Mixing None and terminal-only nested dictionaries causes
        missing values to be represented as NaN.

        This compact representation deliberately contains the same
        scalar keys on EVERY transition.
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
        }

        if outcome not in outcome_codes:
            raise RuntimeError(
                "Unknown finalization outcome: "
                f"{outcome!r}"
            )

        def int_or_default(
            key,
            default,
        ):
            value = outcome_record.get(
                key
            )

            if value is None:
                return int(
                    default
                )

            return int(
                value
            )

        signal_name = outcome_record.get(
            "signal_name"
        )

        if signal_name is None:
            signal_name = ""

        return {
            "outcome":
                outcome,

            "outcome_code":
                outcome_codes[
                    outcome
                ],

            "attempted":
                True,

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

    def reset(
        self,
        **kwargs,
    ):
        observation, info = (
            self.env.reset(
                **kwargs
            )
        )

        #
        # At reset, C++ ACTIONS is the authoritative initial
        # actionable set.
        #
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
        ] = "final_v2"

        info[
            "selection_reward_version"
        ] = "selection_v1"

        #
        # Keep the collector-facing info schema identical from reset
        # onward. The inner FinalizationEvalWrapper may use None here,
        # but the outer training wrapper never exposes that None to
        # Tianshou.
        #
        info[
            "finalization_outcome"
        ] = (
            self._collector_finalization_record(
                None
            )
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
                "FinalRewardWrapper.step() called "
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
                "Reward V2 does not currently define "
                "artificial truncation semantics"
            )

        info = dict(
            info
        )

        if (
            "transition_metrics"
            not in info
        ):
            raise RuntimeError(
                "Wrapped environment did not expose "
                "transition_metrics"
            )

        metrics = (
            self._metrics_from_dict(
                info[
                    "transition_metrics"
                ]
            )
        )

        finalization_outcome = None

        if terminated:
            outcome_record = (
                info.get(
                    "finalization_outcome"
                )
            )

            if not isinstance(
                outcome_record,
                dict,
            ):
                raise RuntimeError(
                    "Terminal Reward V2 transition requires "
                    "FinalizationEvalWrapper outcome"
                )

            if (
                "outcome"
                not in
                outcome_record
            ):
                raise RuntimeError(
                    "finalization_outcome is missing "
                    "'outcome'"
                )

            finalization_outcome = str(
                outcome_record[
                    "outcome"
                ]
            )

        else:
            #
            # A non-terminal transition must not already carry a real
            # finalization result.
            #
            outcome_record = (
                info.get(
                    "finalization_outcome"
                )
            )

            if (
                outcome_record
                is not None
            ):
                raise RuntimeError(
                    "Non-terminal transition unexpectedly "
                    "contains finalization outcome"
                )

        breakdown = (
            compute_reward_v2(
                metrics=metrics,
                initial_actionable_count=(
                    self._initial_actionable_count
                ),
                finalization_outcome=(
                    finalization_outcome
                ),
                weights=self.weights,
            )
        )

        reward_v2 = float(
            breakdown.total
        )

        #
        # Preserve V1 for auditing; expose V2 as the actual Gym reward.
        #
        info[
            "selection_reward_v1"
        ] = float(
            selection_reward
        )

        info[
            "reward_version"
        ] = "final_v2"

        info[
            "reward_v2_breakdown"
        ] = {
            "step":
                breakdown.step,

            "tet_growth":
                breakdown.tet_growth,

            "revert":
                breakdown.revert,

            "convergence":
                breakdown.convergence,

            "finalization":
                breakdown.finalization,

            "total":
                breakdown.total,
        }

        #
        # IMPORTANT:
        #
        # Do this only AFTER Reward V2 has consumed the rich terminal
        # outcome. The returned Gym info must use one fixed schema on
        # every transition so Tianshou does not synthesize NaNs for
        # missing/None terminal-only fields.
        #
        info[
            "finalization_outcome"
        ] = (
            self._collector_finalization_record(
                outcome_record
            )
        )

        return (
            observation,
            reward_v2,
            terminated,
            truncated,
            info,
        )
