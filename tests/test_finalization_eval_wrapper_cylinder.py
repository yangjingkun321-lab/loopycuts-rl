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

MESH = (
    "/home/yjk/loopycuts_inputs/"
    "cylinder_plate_clean/"
    "cylinder_plate_rem_splitted.obj"
)

LOOPS = (
    "/home/yjk/loopycuts_inputs/"
    "cylinder_plate_clean/"
    "cylinder_plate_rem_loop.txt"
)


def main():
    base_env = LoopyCutsEnv(
        executable=EXECUTABLE,
        mesh_file=MESH,
        loop_file=LOOPS,
        echo_logs=False,
    )

    env = FinalizationEvalWrapper(
        base_env
    )

    try:
        observation, info = env.reset(
            seed=123
        )

        assert (
            info[
                "finalization_attempted"
            ]
            is False
        )

        rewards = []

        for action in (
            0,
            1,
            2,
            3,
        ):
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            rewards.append(
                float(
                    reward
                )
            )

            if action != 3:
                assert terminated is False

                assert (
                    info[
                        "finalization_attempted"
                    ]
                    is False
                )

                assert (
                    info[
                        "finalization_outcome"
                    ]
                    is None
                )

        assert terminated is True
        assert truncated is False

        #
        # Reward V1 is deliberately unchanged by D3.
        #
        assert math.isclose(
            sum(
                rewards
            ),
            2.4865046601979746,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        #
        # Returned next observation is still the genuine
        # selection-terminal state.
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

        assert (
            float(
                observation[
                    "obs"
                ][
                    "global"
                ][4]
            )
            ==
            1.0
        )

        assert (
            float(
                observation[
                    "obs"
                ][
                    "global"
                ][5]
            )
            ==
            1.0
        )

        assert (
            info[
                "finalization_attempted"
            ]
            is True
        )

        outcome = (
            info[
                "finalization_outcome"
            ]
        )

        print(
            "selection return:",
            sum(
                rewards
            )
        )

        print(
            "terminal mask:",
            int(
                observation[
                    "mask"
                ].sum()
            )
        )

        print(
            "final outcome:",
            outcome
        )

        assert (
            outcome[
                "outcome"
            ]
            ==
            "FULL_HEX"
        )

        assert (
            outcome[
                "final_hex"
            ]
            ==
            88
        )

        assert (
            outcome[
                "final_total_polys"
            ]
            ==
            88
        )

        assert (
            outcome[
                "full_hex"
            ]
            ==
            1
        )

        print()

        print(
            "PASS: finalization wrapper preserves "
            "Cylinder Selection Reward V1 and "
            "selection-terminal observation while "
            "exposing FULL_HEX outcome."
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
