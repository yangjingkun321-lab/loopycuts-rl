import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(
    "/home/yjk/codes/loopycuts_rl"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from envs.loopycuts_env import LoopyCutsEnv


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

LOOP_FILE = (
    "/home/yjk/loopycuts_inputs/"
    "cylinder_plate_clean/"
    "cylinder_plate_rem_loop.txt"
)


def main():
    env = LoopyCutsEnv(
        executable=EXECUTABLE,
        mesh_file=MESH,
        loop_file=LOOP_FILE,
        echo_logs=False,
    )

    try:
        # ============================================================
        # RESET
        # ============================================================

        observation, info = (
            env.reset(
                seed=123
            )
        )

        print(
            "Initial legal actions:",
            env.legal_actions,
        )

        print(
            "Initial mask count:",
            int(
                observation[
                    "mask"
                ].sum()
            ),
        )

        assert (
            env.observation_space.contains(
                observation
            )
        )

        assert (
            env.action_space.n
            ==
            331
        )

        assert (
            env.legal_actions
            ==
            tuple(
                range(65)
            )
        )

        assert int(
            observation[
                "mask"
            ].sum()
        ) == 65

        assert (
            info[
                "reward_is_placeholder"
            ]
            is False
        )

        assert (
            info[
                "num_executed"
            ]
            ==
            0
        )

        assert (
            env.executed_loop_ids
            ==
            set()
        )

        # ============================================================
        # ORIGINAL CYLINDER SEQUENCE
        # ============================================================

        expected_actions = [
            0,
            1,
            2,
            3,
        ]

        for (
            step_index,
            action,
        ) in enumerate(
            expected_actions,
            start=1,
        ):
            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            print()

            print(
                f"step={step_index} "
                f"action={action} "
                f"status="
                f"{info['step_result']['status']} "
                f"terminated={terminated} "
                f"truncated={truncated} "
                f"mask_count="
                f"{int(next_observation['mask'].sum())}"
            )

            assert (
                env.observation_space.contains(
                    next_observation
                )
            )

            assert math.isclose(
                float(
                    reward
                ),
                float(
                    info[
                        "reward_breakdown"
                    ][
                        "total"
                    ]
                ),
                rel_tol=0.0,
                abs_tol=1e-15,
            )

            assert (
                info[
                    "reward_is_placeholder"
                ]
                is False
            )

            assert truncated is False

            assert action in (
                env.executed_loop_ids
            )

            #
            # The selected action is no longer legal.
            #
            assert not bool(
                next_observation[
                    "mask"
                ][
                    action
                ]
            )

            #
            # executed feature = 1.
            #
            assert float(
                next_observation[
                    "obs"
                ][
                    "loops"
                ][
                    action,
                    11,
                ]
            ) == 1.0

            if action != 3:
                assert terminated is False

            else:
                assert terminated is True

            observation = (
                next_observation
            )

        # ============================================================
        # TERMINAL CYLINDER
        # ============================================================

        assert int(
            env.current_state[
                "terminal"
            ]
        ) == 1

        assert int(
            env.current_state[
                "converged"
            ]
        ) == 1

        assert int(
            env.current_state[
                "selection_success"
            ]
        ) == 1

        assert (
            env.legal_actions
            ==
            ()
        )

        assert not bool(
            observation[
                "mask"
            ].any()
        )

        assert (
            env.executed_loop_ids
            ==
            {
                0,
                1,
                2,
                3,
            }
        )

        # ============================================================
        # step() AFTER TERMINAL MUST FAIL
        # ============================================================

        try:
            env.step(
                0
            )

        except RuntimeError:
            pass

        else:
            raise AssertionError(
                "step() after terminal "
                "did not raise RuntimeError"
            )

        # ============================================================
        # SECOND RESET
        #
        # Must create a completely fresh C++ Stage-2 episode.
        # ============================================================

        observation, info = (
            env.reset(
                seed=456
            )
        )

        print()

        print(
            "After second reset:"
        )

        print(
            "legal count:",
            len(
                env.legal_actions
            ),
        )

        print(
            "executed:",
            env.executed_loop_ids,
        )

        assert (
            env.observation_space.contains(
                observation
            )
        )

        assert (
            env.executed_loop_ids
            ==
            set()
        )

        assert (
            env.legal_actions
            ==
            tuple(
                range(65)
            )
        )

        assert int(
            env.current_state[
                "step"
            ]
        ) == 0

        assert int(
            env.current_state[
                "terminal"
            ]
        ) == 0

        assert int(
            observation[
                "mask"
            ].sum()
        ) == 65

        #
        # No real loop is marked executed after reset.
        #
        assert np.array_equal(
            observation[
                "obs"
            ][
                "loops"
            ][
                :91,
                11,
            ],
            np.zeros(
                91,
                dtype=np.float32,
            ),
        )

        print()

        print(
            "PASS: LoopyCutsEnv Cylinder "
            "reset/step/terminal/reset lifecycle "
            "is correct."
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
