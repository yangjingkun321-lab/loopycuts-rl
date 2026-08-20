from __future__ import annotations

import gymnasium as gym


from bridge.cpp_client import (
    RLServerResourceAbort,
)


from finalization.outcome import (
    OUTCOME_RESOURCE_ABORT,
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

            resource_abort = (
                info.get(
                    "resource_abort"
                )
            )

            if resource_abort is not None:
                # ----------------------------------------------------
                # ResourceGuard already terminated the C++ server.
                #
                # This is a genuine terminal RL transition, but it is
                # NOT a selection-terminal geometry state and must NOT
                # enter FINALIZE_EVAL.
                # ----------------------------------------------------

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

                info[
                    "finalization_attempted"
                ] = False

                info[
                    "finalization_outcome"
                ] = {
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

                    "log_tail":
                        (),
                }

            else:
                #
                # IMPORTANT:
                #
                # The terminal Stage-2 STEP has already completed and
                # produced genuine transition geometry at this point.
                #
                # FINALIZE_EVAL is post-STEP evaluation. If its
                # independent system-swap fuse fires, this SAME final
                # selection transition becomes RESOURCE_ABORT.
                #
                # No additional RL action/transition is fabricated.
                #
                try:
                    outcome = (
                        evaluate_terminal_finalization(
                            self.unwrapped.client
                        )
                    )

                except RLServerResourceAbort as exc:
                    if (
                        str(
                            exc.phase
                        )
                        !=
                        "FINALIZE_EVAL"
                    ):
                        raise

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
                        else int(
                            exc.return_code
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

                    info[
                        "resource_abort"
                    ] = {
                        "outcome":
                            OUTCOME_RESOURCE_ABORT,

                        "phase":
                            "FINALIZE_EVAL",

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

                    info[
                        "finalization_attempted"
                    ] = True

                    info[
                        "finalization_outcome"
                    ] = {
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

                        "log_tail":
                            (),
                    }

                else:
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
