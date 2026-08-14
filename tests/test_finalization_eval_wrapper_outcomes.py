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


from envs.finalization_eval_wrapper import (
    FinalizationEvalWrapper,
)

from envs.loopycuts_env import (
    LoopyCutsEnv,
)


EXECUTABLE = (
    "/home/yjk/codes/LoopyCuts/"
    "volumetric_cutter/"
    "volumetric_cutter"
)


CASES = {
    "deckel": {
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

        "expected_outcome":
            "NON_FULL_HEX",

        "expected_hex":
            512,

        "expected_total":
            518,
    },

    "bracket": {
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

        "expected_outcome":
            "FINALIZATION_CRASH",

        "expected_hex":
            None,

        "expected_total":
            None,
    },
}


def run_case(
    name,
    spec,
):
    base_env = LoopyCutsEnv(
        executable=EXECUTABLE,
        mesh_file=spec[
            "mesh"
        ],
        loop_file=spec[
            "loops"
        ],
        echo_logs=False,
    )

    env = FinalizationEvalWrapper(
        base_env
    )

    try:
        observation, info = (
            env.reset(
                seed=123
            )
        )

        steps = 0

        while not int(
            env.unwrapped.current_state[
                "terminal"
            ]
        ):
            action = min(
                env.unwrapped.legal_actions
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            steps += 1

        assert (
            steps
            ==
            spec[
                "expected_steps"
            ]
        )

        assert terminated is True
        assert truncated is False

        #
        # Selection-terminal observation remains real.
        #
        assert (
            int(
                observation[
                    "mask"
                ].sum()
            )
            ==
            0
        )

        outcome = (
            info[
                "finalization_outcome"
            ]
        )

        print()
        print(
            name,
            "steps:",
            steps,
        )

        print(
            name,
            "outcome:",
            outcome[
                "outcome"
            ],
        )

        print(
            name,
            "signal:",
            outcome[
                "signal_name"
            ],
        )

        print(
            name,
            "hex:",
            outcome[
                "final_hex"
            ],
        )

        print(
            name,
            "total:",
            outcome[
                "final_total_polys"
            ],
        )

        assert (
            outcome[
                "outcome"
            ]
            ==
            spec[
                "expected_outcome"
            ]
        )

        assert (
            outcome[
                "final_hex"
            ]
            ==
            spec[
                "expected_hex"
            ]
        )

        assert (
            outcome[
                "final_total_polys"
            ]
            ==
            spec[
                "expected_total"
            ]
        )

        if (
            name
            ==
            "bracket"
        ):
            assert (
                outcome[
                    "crashed"
                ]
                is True
            )

            assert (
                outcome[
                    "return_code"
                ]
                ==
                -6
            )

            assert (
                outcome[
                    "signal_number"
                ]
                ==
                6
            )

            assert (
                outcome[
                    "signal_name"
                ]
                ==
                "SIGABRT"
            )

    finally:
        env.close()


def main():
    for (
        name,
        spec,
    ) in CASES.items():
        run_case(
            name,
            spec,
        )

    print()

    print(
        "PASS: terminal wrapper exposes "
        "NON_FULL_HEX and FINALIZATION_CRASH "
        "without changing selection-terminal semantics."
    )


if __name__ == "__main__":
    main()
