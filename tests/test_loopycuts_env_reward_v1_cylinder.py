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


EXPECTED_REWARDS = [
    -0.14250819235093484,
    -0.17114110262030834,
    -0.106404689,
    2.906558644,
]


def main():
    env = LoopyCutsEnv(
        executable=EXECUTABLE,
        mesh_file=MESH,
        loop_file=LOOP_FILE,
        echo_logs=False,
    )

    try:
        observation, info = env.reset(
            seed=123
        )

        assert (
            info[
                "reward_is_placeholder"
            ]
            is False
        )

        assert (
            info[
                "reward_version"
            ]
            ==
            "selection_v1"
        )

        actual_rewards = []

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

            actual_rewards.append(
                reward
            )

            breakdown = (
                info[
                    "reward_breakdown"
                ]
            )

            metrics = (
                info[
                    "transition_metrics"
                ]
            )

            print(
                f"step={metrics['step']} "
                f"id={action} "
                f"status={metrics['status']} "
                f"step_r={breakdown['step']:+.6f} "
                f"tet_r={breakdown['tet_growth']:+.6f} "
                f"revert_r={breakdown['revert']:+.3f} "
                f"conv_r={breakdown['convergence']:+.3f} "
                f"terminal_r={breakdown['terminal']:+.3f} "
                f"reward={reward:+.12f}"
            )

            assert math.isclose(
                reward,
                breakdown[
                    "total"
                ],
                rel_tol=0.0,
                abs_tol=1e-15,
            )

            assert truncated is False

        episode_return = sum(
            actual_rewards
        )

        print()
        print(
            "episode return:",
            episode_return,
        )

        assert math.isclose(
            episode_return,
            2.4865046601979746,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        assert int(
            env.current_state[
                "terminal"
            ]
        ) == 1

        assert int(
            env.current_state[
                "selection_success"
            ]
        ) == 1

        #
        # First convergence itself receives no convergence bonus.
        #
        final_breakdown = (
            info[
                "reward_breakdown"
            ]
        )

        assert (
            final_breakdown[
                "convergence"
            ]
            ==
            0.0
        )

        assert (
            final_breakdown[
                "terminal"
            ]
            ==
            3.0
        )

        print()

        print(
            "PASS: LoopyCutsEnv emits "
            "Selection Reward V1 for Cylinder."
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
