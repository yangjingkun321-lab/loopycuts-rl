from bridge.cpp_client import (
    RLServerProcessError,
)


def run_episode(
    client,
    policy,
    output_dir=None,
    finalize=True,
    max_steps=10000,
):
    """
    Execute one complete LoopyCuts loop-selection episode.

    The C++ GlobalState remains alive for the entire
    Stage-2 episode.

    Stage-2:
        The policy selects legal loops until terminal=True.

    Finalization:
        Every terminal Stage-2 episode may enter the
        original LoopyCuts finalization pipeline,
        regardless of selection_success.

    A C++ process crash specifically during FINALIZE
    is recorded as an evaluation outcome instead of
    terminating the whole Python experiment.
    """

    if hasattr(
        policy,
        "reset",
    ):
        policy.reset()

    trajectory = []

    # ------------------------------------------------------------
    # Stage-2 loop-selection phase
    # ------------------------------------------------------------

    while not client.state[
        "terminal"
    ]:
        actions = list(
            client.actions
        )

        if not actions:
            raise RuntimeError(
                "Environment is non-terminal "
                "but has no legal actions"
            )

        action = policy.select(
            client.state,
            actions,
        )

        if action not in actions:
            raise RuntimeError(
                f"Policy selected illegal "
                f"action {action}. "
                f"Legal actions are: {actions}"
            )

        (
            step_result,
            state,
            next_actions,
        ) = client.step(
            action
        )

        trajectory.append(
            {
                "action":
                    action,

                "step_result":
                    step_result,

                "state":
                    dict(state),

                "next_actions":
                    list(next_actions),
            }
        )

        if (
            len(trajectory)
            >= max_steps
        ):
            raise RuntimeError(
                "Episode exceeded "
                f"max_steps={max_steps}"
            )

    # ------------------------------------------------------------
    # Preserve Stage-2 terminal state BEFORE finalization.
    # ------------------------------------------------------------

    selection_state = dict(
        client.state
    )

    result = {
        "trajectory":
            trajectory,

        "num_steps":
            len(trajectory),

        # --------------------------------------------------------
        # Stage-2 selection result
        # --------------------------------------------------------

        "selection_success":
            selection_state[
                "selection_success"
            ],

        "converged":
            selection_state[
                "converged"
            ],

        "terminal":
            selection_state[
                "terminal"
            ],

        "selection_state":
            selection_state,

        # --------------------------------------------------------
        # Finalization result
        # --------------------------------------------------------

        "finalization_attempted":
            False,

        "finalization_completed":
            False,

        "finalization_process_terminated":
            False,

        "finalization_crashed":
            False,

        "finalization_returncode":
            None,

        "finalization_signal":
            None,

        "finalization_signal_name":
            None,

        "finalization_error":
            None,

        "finalization_log_tail":
            [],

        "final_result":
            None,

        "final_state":
            None,
    }

    # ------------------------------------------------------------
    # Finalization handoff
    # ------------------------------------------------------------

    if finalize:

        if output_dir is None:
            raise ValueError(
                "output_dir is required "
                "when finalize=True"
            )

        if not selection_state[
            "terminal"
        ]:
            raise RuntimeError(
                "Cannot finalize a "
                "non-terminal Stage-2 episode"
            )

        result[
            "finalization_attempted"
        ] = True

        try:
            (
                final_result,
                final_state,
            ) = client.finalize(
                output_dir
            )

        except RLServerProcessError as exc:
            #
            # IMPORTANT:
            #
            # Catch ONLY an unexpected C++ process
            # termination during FINALIZE.
            #
            # Protocol errors and ordinary Python bugs
            # are deliberately NOT swallowed.
            #
            result[
                "finalization_process_terminated"
            ] = True

            result[
                "finalization_returncode"
            ] = exc.return_code

            result[
                "finalization_signal"
            ] = exc.signal_number

            result[
                "finalization_signal_name"
            ] = exc.signal_name

            result[
                "finalization_error"
            ] = str(exc)

            #
            # Non-zero return code means the process
            # itself failed.
            #
            result[
                "finalization_crashed"
            ] = (
                exc.return_code
                not in (None, 0)
            )

            #
            # Preserve the last part of the actual C++
            # output. This should include assertion /
            # abort diagnostics for BracketInches.
            #
            result[
                "finalization_log_tail"
            ] = exc.lines[-50:]

        else:
            result[
                "finalization_completed"
            ] = True

            result[
                "final_result"
            ] = final_result

            result[
                "final_state"
            ] = final_state

    return result