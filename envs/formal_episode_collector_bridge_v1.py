from __future__ import annotations

import copy

import gymnasium as gym


FORMAL_EPISODE_COLLECTOR_BRIDGE_VERSION = (
    "loopycuts_formal_episode_collector_bridge_v1"
)


class FormalEpisodeCollectorBridgeV1(
    gym.Wrapper
):
    """
    Tianshou Collector compatibility bridge for formal Stage-II.

    Formal Stage-II creates one vector environment for exactly one
    LoopyCuts model episode and closes it immediately after that
    episode finishes.

    Tianshou automatically calls reset() after a terminal transition
    before collect() returns.  A real reset at that point would launch
    another volumetric_cutter for the just-finished model, even though
    that new process will never be used.

    Therefore suppress exactly ONE automatic reset after EVERY native
    terminal transition:

        terminal transition
            -> Collector calls reset()
            -> return cached genuine initial observation/info
            -> DO NOT launch another C++ process
            -> collect() returns
            -> formal episode collector is closed

    This applies to:

        FULL_HEX
        NON_FULL_HEX
        FINALIZATION_CRASH
        RESOURCE_ABORT

    Calling step() after the suppressed reset is a programming error.
    """

    def __init__(
        self,
        env,
    ):
        super().__init__(
            env
        )

        self._cached_reset_observation = None
        self._cached_reset_info = None

        self._suppress_next_reset = False
        self._after_suppressed_reset = False

        self.suppressed_reset_count = 0


    @property
    def suppress_next_reset(
        self,
    ) -> bool:
        return bool(
            self._suppress_next_reset
        )


    @property
    def after_suppressed_reset(
        self,
    ) -> bool:
        return bool(
            self._after_suppressed_reset
        )


    def reset(
        self,
        **kwargs,
    ):
        # ============================================================
        # Consume exactly one Collector autoreset after terminal.
        # ============================================================

        if self._suppress_next_reset:
            if (
                self._cached_reset_observation
                is None
                or
                self._cached_reset_info
                is None
            ):
                raise RuntimeError(
                    "Terminal autoreset suppression has no "
                    "cached genuine reset state"
                )

            self._suppress_next_reset = False
            self._after_suppressed_reset = True

            self.suppressed_reset_count += 1

            observation = copy.deepcopy(
                self._cached_reset_observation
            )

            info = copy.deepcopy(
                self._cached_reset_info
            )

            info = dict(
                info
            )

            info[
                "formal_episode_collector_autoreset_suppressed"
            ] = True

            return (
                observation,
                info,
            )


        # ============================================================
        # Genuine initial environment reset.
        # ============================================================

        observation, info = (
            self.env.reset(
                **kwargs
            )
        )

        self._cached_reset_observation = (
            copy.deepcopy(
                observation
            )
        )

        self._cached_reset_info = (
            copy.deepcopy(
                info
            )
        )

        self._after_suppressed_reset = False

        return (
            observation,
            info,
        )


    def step(
        self,
        action,
    ):
        if self._after_suppressed_reset:
            raise RuntimeError(
                "Collector attempted to step after the "
                "formal terminal autoreset was suppressed. "
                "The current collector must be closed and "
                "a new model episode must be created."
            )

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

        if truncated:
            raise RuntimeError(
                "Formal Stage-II does not permit artificial "
                "truncation at an episode boundary"
            )


        # ============================================================
        # Every REAL terminal transition ends this formal model
        # collector.  Suppress Tianshou's subsequent autoreset.
        # ============================================================

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
                    "Formal terminal transition lacks "
                    "terminal outcome record"
                )

            outcome = (
                outcome_record.get(
                    "outcome"
                )
            )

            if outcome not in {
                "FULL_HEX",
                "NON_FULL_HEX",
                "FINALIZATION_CRASH",
                "RESOURCE_ABORT",
            }:
                raise RuntimeError(
                    "Unknown formal terminal outcome: "
                    f"{outcome!r}"
                )

            resource_guard = (
                info.get(
                    "resource_guard"
                )
            )

            resource_abort = False

            if isinstance(
                resource_guard,
                dict,
            ):
                resource_abort = bool(
                    resource_guard.get(
                        "triggered",
                        False,
                    )
                )

            if (
                resource_abort
                !=
                (
                    outcome
                    ==
                    "RESOURCE_ABORT"
                )
            ):
                raise RuntimeError(
                    "ResourceGuard trigger and terminal "
                    "outcome are inconsistent"
                )

            self._suppress_next_reset = True

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
