from __future__ import annotations

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
    RLServerProcessError,
)

from evaluation.episode_runner import (
    run_episode,
)


EXECUTABLE = (
    "/home/yjk/codes/LoopyCuts/"
    "volumetric_cutter/"
    "volumetric_cutter"
)


class MinLegalPolicy:
    def reset(
        self,
    ):
        pass

    def select(
        self,
        state,
        actions,
    ):
        if not actions:
            raise RuntimeError(
                "No legal action"
            )

        return min(
            actions
        )


CASES = {
    "cylinder_original": {
        "mesh": (
            "/home/yjk/loopycuts_inputs/"
            "cylinder_plate_clean/"
            "cylinder_plate_rem_splitted.obj"
        ),

        "loops": (
            "/home/yjk/loopycuts_inputs/"
            "cylinder_plate_clean/"
            "cylinder_plate_rem_loop.txt"
        ),

        "expected_steps":
            4,

        "expected_selection_success":
            1,

        "expected_outcome":
            "FULL_HEX",

        "expected_hex":
            88,

        "expected_total":
            88,
    },

    "deckel_original": {
        "mesh": (
            "/home/yjk/codes/LoopyCuts/"
            "test_data/deckel/"
            "deckel_rem_splitted.obj"
        ),

        "loops": (
            "/home/yjk/codes/LoopyCuts/"
            "test_data/deckel/"
            "deckel_rem_loop.txt"
        ),

        "expected_steps":
            23,

        "expected_selection_success":
            1,

        "expected_outcome":
            "NON_FULL_HEX",

        "expected_hex":
            512,

        "expected_total":
            518,
    },

    "bracket_original": {
        "mesh": (
            "/home/yjk/codes/LoopyCuts/"
            "test_data/BracketInches/"
            "BracketInches_rem_rem_splitted.obj"
        ),

        "loops": (
            "/home/yjk/codes/LoopyCuts/"
            "test_data/BracketInches/"
            "BracketInches_rem_rem_loop.txt"
        ),

        "expected_steps":
            38,

        "expected_selection_success":
            0,

        "expected_outcome":
            "FINALIZATION_CRASH",

        "expected_hex":
            None,

        "expected_total":
            None,
    },
}


def run_case(
    case_name,
    spec,
):
    print()
    print(
        "=" * 72
    )

    print(
        "CASE:",
        case_name
    )

    print(
        "=" * 72
    )

    policy = MinLegalPolicy()

    client = LoopyCutsClient(
        executable=EXECUTABLE,
        mesh_file=spec[
            "mesh"
        ],
        loop_file=spec[
            "loops"
        ],
        echo_logs=False,
    )

    try:
        #
        # Selection only.
        #
        selection = run_episode(
            client=client,
            policy=policy,
            finalize=False,
        )

        assert (
            selection[
                "num_steps"
            ]
            ==
            spec[
                "expected_steps"
            ]
        )

        assert int(
            selection[
                "selection_success"
            ]
        ) == int(
            spec[
                "expected_selection_success"
            ]
        )

        assert int(
            selection[
                "terminal"
            ]
        ) == 1

        print(
            "selection steps:",
            selection[
                "num_steps"
            ]
        )

        print(
            "selection success:",
            selection[
                "selection_success"
            ]
        )

        #
        # No-save real finalization.
        #
        try:
            (
                final_result,
                final_state,
            ) = client.finalize_eval()

        except RLServerProcessError as exc:
            if (
                spec[
                    "expected_outcome"
                ]
                !=
                "FINALIZATION_CRASH"
            ):
                raise

            print(
                "outcome:",
                "FINALIZATION_CRASH"
            )

            print(
                "returncode:",
                exc.return_code
            )

            print(
                "signal:",
                exc.signal_name
            )

            assert (
                exc.phase
                ==
                "FINALIZE_EVAL"
            )

            assert (
                exc.return_code
                ==
                -6
            )

            assert (
                exc.signal_number
                ==
                6
            )

            assert (
                exc.signal_name
                ==
                "SIGABRT"
            )

            return

        #
        # If finalization completed but crash was expected,
        # ground-truth equivalence failed.
        #
        if (
            spec[
                "expected_outcome"
            ]
            ==
            "FINALIZATION_CRASH"
        ):
            raise AssertionError(
                f"{case_name}: FINALIZE_EVAL completed, "
                "but D1 ground truth is SIGABRT"
            )

        assert (
            final_state[
                "finalized"
            ]
            ==
            1
        )

        assert (
            client.actions
            ==
            []
        )

        nh = int(
            final_result[
                "hex"
            ]
        )

        npolys = int(
            final_result[
                "total_polys"
            ]
        )

        full_hex = int(
            final_result[
                "full_hex"
            ]
        )

        outcome = (
            "FULL_HEX"
            if full_hex
            else
            "NON_FULL_HEX"
        )

        print(
            "outcome:",
            outcome
        )

        print(
            "hex:",
            nh
        )

        print(
            "total_polys:",
            npolys
        )

        print(
            "full_hex:",
            full_hex
        )

        assert (
            outcome
            ==
            spec[
                "expected_outcome"
            ]
        )

        assert (
            nh
            ==
            spec[
                "expected_hex"
            ]
        )

        assert (
            npolys
            ==
            spec[
                "expected_total"
            ]
        )

        if outcome == "FULL_HEX":
            assert full_hex == 1
        else:
            assert full_hex == 0

    finally:
        client.close()


def main():
    for (
        case_name,
        spec,
    ) in CASES.items():
        run_case(
            case_name,
            spec,
        )

    print()
    print(
        "=" * 72
    )

    print(
        "PASS: FINALIZE_EVAL reproduces "
        "D1 ground-truth FULL_HEX, "
        "NON_FULL_HEX, and FINALIZATION_CRASH outcomes."
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
