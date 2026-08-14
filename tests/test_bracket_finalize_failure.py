import sys
from pathlib import Path


PROJECT_ROOT = Path(
    "/home/yjk/codes/loopycuts_rl"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from bridge.cpp_client import (
    LoopyCutsClient,
)

from evaluation.episode_runner import (
    run_episode,
)

from policies.simple import (
    OriginalOrderPolicy,
)


EXECUTABLE = (
    "/home/yjk/codes/LoopyCuts/"
    "volumetric_cutter/"
    "volumetric_cutter"
)

MESH = (
    "/home/yjk/codes/LoopyCuts/"
    "test_data/BracketInches/"
    "BracketInches_rem_rem_splitted.obj"
)

LOOPS = (
    "/home/yjk/codes/LoopyCuts/"
    "test_data/BracketInches/"
    "BracketInches_rem_rem_loop.txt"
)

OUTPUT = Path(
    "/home/yjk/loopycuts_test/"
    "semantics_original/"
    "BracketInches/"
    "rl_finalize_failure_test"
)


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy = (
        OriginalOrderPolicy()
    )

    with LoopyCutsClient(
        executable=EXECUTABLE,
        mesh_file=MESH,
        loop_file=LOOPS,
        echo_logs=False,
    ) as client:

        result = run_episode(
            client=client,
            policy=policy,
            output_dir=OUTPUT,
            finalize=True,
        )

    actions = [
        item["action"]
        for item
        in result["trajectory"]
    ]

    print()
    print(
        "===================================="
    )

    print(
        "BRACKET FINALIZATION TEST"
    )

    print(
        "===================================="
    )

    print(
        "num_steps:",
        result["num_steps"],
    )

    print(
        "actions:",
        actions,
    )

    print(
        "terminal:",
        result["terminal"],
    )

    print(
        "converged:",
        result["converged"],
    )

    print(
        "selection_success:",
        result["selection_success"],
    )

    print(
        "finalization_attempted:",
        result[
            "finalization_attempted"
        ],
    )

    print(
        "finalization_completed:",
        result[
            "finalization_completed"
        ],
    )

    print(
        "finalization_process_terminated:",
        result[
            "finalization_process_terminated"
        ],
    )

    print(
        "finalization_crashed:",
        result[
            "finalization_crashed"
        ],
    )

    print(
        "returncode:",
        result[
            "finalization_returncode"
        ],
    )

    print(
        "signal:",
        result[
            "finalization_signal"
        ],
    )

    print(
        "signal_name:",
        result[
            "finalization_signal_name"
        ],
    )

    print(
        "final_result:",
        result[
            "final_result"
        ],
    )

    print()
    print(
        "===== FINALIZATION LOG TAIL ====="
    )

    for line in result[
        "finalization_log_tail"
    ]:
        print(line)

    #
    # Stage-2 semantics must match the verified
    # original forward-only main traversal.
    #
    expected_actions = (
        list(range(0, 29))
        +
        list(range(81, 90))
    )

    assert actions == (
        expected_actions
    )

    assert result[
        "num_steps"
    ] == 38

    assert result[
        "terminal"
    ] == 1

    assert result[
        "converged"
    ] == 0

    assert result[
        "selection_success"
    ] == 0

    assert result[
        "finalization_attempted"
    ] is True

    #
    # Do NOT assert that finalization must crash.
    #
    # If it completes successfully, that is also
    # scientifically useful information.
    #
    print()
    print(
        "PASS: Stage-2 failure was handed "
        "to FINALIZE without aborting the "
        "Python evaluation process."
    )


if __name__ == "__main__":
    main()