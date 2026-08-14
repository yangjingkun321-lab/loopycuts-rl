from __future__ import annotations

import gymnasium as gym


from finalization.outcome import (
    evaluate_terminal_finalization,
)


class FinalizationEvalWrapper(
    gym.Wrapper
):
    """
    Explicit finalization-aware wrapper around LoopyCutsEnv.

    The wrapped LoopyCutsEnv remains the Stage-2 selection MDP.

    On the transition that reaches selection terminal:

        1. preserve the genuine selection-terminal observation;
        2. preserve Selection Reward V1 unchanged;
        3. execute no-save FINALIZE_EVAL;
        4. attach the real finalization outcome to info.

    D3 deliberately DOES NOT modify reward.

    Final-outcome reward design belongs to Phase 2D-D4.
    """

    def __init__(
        self,
        env,
    ):
        super().__init__(
            env
        )

        self._episode_finalized = False

    def reset(
        self,
        **kwargs,
    ):
        self._episode_finalized = False

        observation, info = (
            self.env.reset(
                **kwargs
            )
        )

        info = dict(
            info
        )

        info[
            "finalization_attempted"
        ] = False

        info[
            "finalization_outcome"
        ] = None

        return (
            observation,
            info,
        )

    def step(
        self,
        action,
    ):
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(
            action
        )

        info = dict(
            info
        )

        info[
            "finalization_attempted"
        ] = False

        info[
            "finalization_outcome"
        ] = None

        if (
            terminated
            and
            not truncated
        ):
            if self._episode_finalized:
                raise RuntimeError(
                    "Terminal episode has already "
                    "entered FINALIZE_EVAL"
                )

            self._episode_finalized = True

            #
            # IMPORTANT:
            #
            # evaluate_terminal_finalization() mutates C++ state,
            # but we deliberately return the observation already
            # produced by LoopyCutsEnv BEFORE finalization.
            #
            # Therefore obs_next remains the genuine selection-terminal
            # observation with its real all-False action mask.
            #
            outcome = (
                evaluate_terminal_finalization(
                    self.unwrapped.client
                )
            )

            info[
                "finalization_attempted"
            ] = True

            info[
                "finalization_outcome"
            ] = outcome.to_dict()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
