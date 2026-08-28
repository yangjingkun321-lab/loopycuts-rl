from __future__ import annotations

from pathlib import Path

import gymnasium as gym


from bridge.cpp_client import (
    RLServerResourceAbort,
)

from finalization.outcome import (
    OUTCOME_RESOURCE_ABORT,
)

from finalization.quality_outcome_v1 import (
    evaluate_terminal_quality_finalization,
)


class FinalizationQualityWrapperV1(
    gym.Wrapper
):
    """
    V5 quality-aware terminal finalization layer.

    Wrapped LoopyCutsEnv remains the Stage-2 selection MDP.

    Normal non-terminal STEP:
        transparent.

    Genuine selection terminal:
        run

            FINALIZE_QUALITY <quality_ref>

        on the same persistent C++ episode.

    RESOURCE_ABORT:
        remains one terminal RL transition and never fabricates
        another action/transition.
    """

    def __init__(
        self,
        env,
        *,
        quality_ref_path,
        expected_model: str,
    ):
        super().__init__(
            env
        )

        quality_ref_path = Path(
            quality_ref_path
        ).resolve()

        if not quality_ref_path.is_file():
            raise FileNotFoundError(
                quality_ref_path
            )

        if any(
            ch.isspace()
            for ch in str(
                quality_ref_path
            )
        ):
            raise ValueError(
                "quality_ref_path must not contain whitespace"
            )

        expected_model = str(
            expected_model
        )

        if not expected_model:
            raise ValueError(
                "expected_model must be non-empty"
            )

        self.quality_ref_path = (
            quality_ref_path
        )

        self.expected_model = (
            expected_model
        )

        self._episode_finalized = False


    # ------------------------------------------------------------------

    @staticmethod
    def _resource_outcome_record(
        resource_abort,
    ):
        if not isinstance(
            resource_abort,
            dict,
        ):
            raise RuntimeError(
                "resource_abort must be a dict"
            )

        if (
            resource_abort.get(
                "outcome"
            )
            !=
            OUTCOME_RESOURCE_ABORT
        ):
            raise RuntimeError(
                "Unknown resource_abort outcome"
            )

        return_code = (
            resource_abort.get(
                "return_code"
            )
        )

        signal_number = None
        signal_name = None

        if (
            isinstance(
                return_code,
                int,
            )
            and
            return_code < 0
        ):
            signal_number = (
                -return_code
            )

            if signal_number == 9:
                signal_name = (
                    "SIGKILL"
                )

        return {
            "outcome":
                OUTCOME_RESOURCE_ABORT,

            "completed":
                False,

            "crashed":
                False,

            "return_code":
                return_code,

            "signal_number":
                signal_number,

            "signal_name":
                signal_name,

            "final_hex":
                None,

            "final_total_polys":
                None,

            "full_hex":
                None,

            "final_result":
                None,

            "final_state":
                None,

            "terminal_quality":
                None,

            "log_tail":
                (),
        }


    # ------------------------------------------------------------------

    @staticmethod
    def _resource_abort_from_finalize_exception(
        *,
        exc,
        action,
    ):
        snapshot = (
            exc.snapshot
        )

        cpp_memory = (
            snapshot.cpp_memory
        )

        cpp_rss_bytes = 0
        cpp_swap_bytes = 0

        if cpp_memory is not None:
            cpp_rss_bytes = int(
                cpp_memory.rss_bytes
            )

            cpp_swap_bytes = int(
                cpp_memory.swap_bytes
            )

        return_code = (
            None
            if exc.return_code is None
            else
            int(
                exc.return_code
            )
        )

        return {
            "outcome":
                OUTCOME_RESOURCE_ABORT,

            "phase":
                "FINALIZE_QUALITY",

            "guard_state":
                str(
                    exc.guard_state
                ),

            "action":
                int(
                    action
                ),

            "return_code":
                return_code,

            "swap_used_bytes":
                int(
                    snapshot.swap_used_bytes
                ),

            "swap_total_bytes":
                int(
                    snapshot.swap_total_bytes
                ),

            "swap_free_bytes":
                int(
                    snapshot.swap_free_bytes
                ),

            "mem_available_bytes":
                int(
                    snapshot.mem_available_bytes
                ),

            "python_rss_bytes":
                int(
                    snapshot
                    .python_memory
                    .rss_bytes
                ),

            "python_swap_bytes":
                int(
                    snapshot
                    .python_memory
                    .swap_bytes
                ),

            "cpp_rss_bytes":
                cpp_rss_bytes,

            "cpp_swap_bytes":
                cpp_swap_bytes,
        }


    # ------------------------------------------------------------------

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

        info[
            "terminal_quality"
        ] = None

        return (
            observation,
            info,
        )


    # ------------------------------------------------------------------

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

        info[
            "terminal_quality"
        ] = None

        if (
            terminated
            and
            not truncated
        ):
            if self._episode_finalized:
                raise RuntimeError(
                    "Terminal episode has already "
                    "entered FINALIZE_QUALITY"
                )

            self._episode_finalized = True

            resource_abort = (
                info.get(
                    "resource_abort"
                )
            )

            # ========================================================
            # STEP ResourceGuard abort.
            #
            # No valid terminal geometry exists, therefore do NOT
            # enter FINALIZE_QUALITY.
            # ========================================================

            if resource_abort is not None:
                info[
                    "finalization_attempted"
                ] = False

                info[
                    "finalization_outcome"
                ] = (
                    self._resource_outcome_record(
                        resource_abort
                    )
                )

            # ========================================================
            # Genuine selection terminal.
            # ========================================================

            else:
                try:
                    outcome = (
                        evaluate_terminal_quality_finalization(
                            self.unwrapped.client,

                            quality_ref_path=
                                self.quality_ref_path,

                            expected_model=
                                self.expected_model,
                        )
                    )

                except RLServerResourceAbort as exc:
                    if (
                        str(
                            exc.phase
                        )
                        !=
                        "FINALIZE_QUALITY"
                    ):
                        raise

                    resource_abort = (
                        self._resource_abort_from_finalize_exception(
                            exc=
                                exc,

                            action=
                                action,
                        )
                    )

                    info[
                        "resource_abort"
                    ] = (
                        resource_abort
                    )

                    info[
                        "finalization_attempted"
                    ] = True

                    info[
                        "finalization_outcome"
                    ] = (
                        self._resource_outcome_record(
                            resource_abort
                        )
                    )

                else:
                    info[
                        "finalization_attempted"
                    ] = True

                    info[
                        "finalization_outcome"
                    ] = (
                        outcome.to_dict()
                    )

                    if (
                        outcome.terminal_quality
                        is not None
                    ):
                        info[
                            "terminal_quality"
                        ] = (
                            outcome
                            .terminal_quality
                            .to_dict()
                        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
