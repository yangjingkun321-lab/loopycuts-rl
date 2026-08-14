from __future__ import annotations

import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    "/home/yjk/codes/loopycuts_rl"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from envs.final_reward_wrapper import (
    FinalRewardWrapper,
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

        "expected_outcome":
            "FULL_HEX",

        "expected_finalization":
            3.0,

        "expected_terminal_v2":
            2.9065586441455137,

        "expected_return":
            2.4865046601979746,

        #
        # V1 +3 proxy is replaced by real +3 FULL_HEX,
        # so this terminal transition is numerically unchanged.
        #
        "expected_v2_minus_v1_terminal":
            0.0,
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

        "expected_outcome":
            "NON_FULL_HEX",

        "expected_finalization":
            -3.0,

        "expected_terminal_v2":
            -3.0511591131298803,

        "expected_return":
            -4.957490718331989,

        #
        # V1 +3 proxy -> V2 -3 NON_FULL_HEX.
        #
        "expected_v2_minus_v1_terminal":
            -6.0,
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

        "expected_outcome":
            "FINALIZATION_CRASH",

        "expected_finalization":
            -4.0,

        "expected_terminal_v2":
            -4.137326320642485,

        "expected_return":
            -7.708048447038928,

        #
        # V1 terminal failure -3 -> real crash -4.
        #
        "expected_v2_minus_v1_terminal":
            -1.0,
    },
}


def run_case(
    name,
    spec,
):
    print()
    print(
        "=" * 76
    )

    print(
        "CASE:",
        name
    )

    print(
        "=" * 76
    )

    env = FinalRewardWrapper(
        FinalizationEvalWrapper(
            LoopyCutsEnv(
                executable=EXECUTABLE,
                mesh_file=spec[
                    "mesh"
                ],
                loop_file=spec[
                    "loops"
                ],
                echo_logs=False,
            )
        )
    )

    try:
        observation, info = (
            env.reset(
                seed=123
            )
        )

        assert (
            info[
                "reward_version"
            ]
            ==
            "final_v2"
        )

        rewards = []
        steps = 0
        final_info = None

        while True:
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

            rewards.append(
                float(
                    reward
                )
            )

            assert truncated is False

            assert (
                info[
                    "reward_version"
                ]
                ==
                "final_v2"
            )

            if terminated:
                final_info = info
                break

        if final_info is None:
            raise RuntimeError(
                f"{name}: missing terminal info"
            )

        total = float(
            sum(
                rewards
            )
        )

        outcome = (
            final_info[
                "finalization_outcome"
            ]
        )

        breakdown = (
            final_info[
                "reward_v2_breakdown"
            ]
        )

        selection_v1_terminal = float(
            final_info[
                "selection_reward_v1"
            ]
        )

        terminal_v2 = float(
            breakdown[
                "total"
            ]
        )

        delta_terminal = (
            terminal_v2
            -
            selection_v1_terminal
        )

        print(
            "steps:",
            steps
        )

        print(
            "outcome:",
            outcome[
                "outcome"
            ]
        )

        print(
            "finalization component:",
            breakdown[
                "finalization"
            ]
        )

        print(
            "selection V1 terminal:",
            selection_v1_terminal
        )

        print(
            "V2 terminal:",
            terminal_v2
        )

        print(
            "V2 - V1 terminal:",
            delta_terminal
        )

        print(
            "episode V2 return:",
            total
        )

        print(
            "terminal mask:",
            int(
                observation[
                    "mask"
                ].sum()
            )
        )

        assert (
            steps
            ==
            spec[
                "expected_steps"
            ]
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

        assert math.isclose(
            float(
                breakdown[
                    "finalization"
                ]
            ),
            spec[
                "expected_finalization"
            ],
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        assert math.isclose(
            terminal_v2,
            spec[
                "expected_terminal_v2"
            ],
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        assert math.isclose(
            total,
            spec[
                "expected_return"
            ],
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        assert math.isclose(
            delta_terminal,
            spec[
                "expected_v2_minus_v1_terminal"
            ],
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        #
        # Gym transition still terminates on the real last
        # Stage-2 action.
        #
        assert terminated is True

        #
        # Finalization does not replace obs_next with a
        # post-finalization pseudo-state.
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

        #
        # Outcome-specific consistency.
        #
        if (
            spec[
                "expected_outcome"
            ]
            ==
            "FULL_HEX"
        ):
            assert (
                outcome[
                    "completed"
                ]
                is True
            )

            assert (
                outcome[
                    "crashed"
                ]
                is False
            )

            assert (
                outcome[
                    "full_hex"
                ]
                ==
                1
            )

        elif (
            spec[
                "expected_outcome"
            ]
            ==
            "NON_FULL_HEX"
        ):
            assert (
                outcome[
                    "completed"
                ]
                is True
            )

            assert (
                outcome[
                    "crashed"
                ]
                is False
            )

            assert (
                outcome[
                    "full_hex"
                ]
                ==
                0
            )

        elif (
            spec[
                "expected_outcome"
            ]
            ==
            "FINALIZATION_CRASH"
        ):
            assert (
                outcome[
                    "completed"
                ]
                is False
            )

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

        else:
            raise RuntimeError(
                f"Unexpected expected outcome: "
                f"{spec['expected_outcome']}"
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
        "=" * 76
    )

    print(
        "PASS: online Final-aware Reward V2 matches "
        "offline balanced ground truth for FULL_HEX, "
        "NON_FULL_HEX, and FINALIZATION_CRASH."
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":
    main()
