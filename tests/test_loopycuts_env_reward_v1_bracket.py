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
    "/home/yjk/codes/LoopyCuts/"
    "test_data/BracketInches/"
    "BracketInches_rem_rem_splitted.obj"
)

LOOP_FILE = (
    "/home/yjk/codes/LoopyCuts/"
    "test_data/BracketInches/"
    "BracketInches_rem_rem_loop.txt"
)


EXPECTED_ACTIONS = (
    list(
        range(
            29
        )
    )
    +
    list(
        range(
            81,
            90,
        )
    )
)


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

        rewards = []

        loop87_breakdown = None
        loop89_breakdown = None

        while not int(
            env.current_state[
                "terminal"
            ]
        ):
            action = min(
                env.legal_actions
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

            rewards.append(
                reward
            )

            breakdown = (
                info[
                    "reward_breakdown"
                ]
            )

            if action == 87:
                loop87_breakdown = dict(
                    breakdown
                )

            if action == 89:
                loop89_breakdown = dict(
                    breakdown
                )

        assert (
            list(
                env.executed_loop_ids
            )
            is not None
        )

        assert (
            loop87_breakdown
            is not None
        )

        assert (
            loop89_breakdown
            is not None
        )

        # ============================================================
        # Convergence destruction.
        # ============================================================

        assert (
            loop87_breakdown[
                "convergence"
            ]
            ==
            -1.0
        )

        assert (
            loop87_breakdown[
                "terminal"
            ]
            ==
            0.0
        )

        # ============================================================
        # Terminal failure on reverted loop 89.
        # ============================================================

        assert (
            loop89_breakdown[
                "revert"
            ]
            ==
            -0.10
        )

        assert (
            loop89_breakdown[
                "terminal"
            ]
            ==
            -3.0
        )

        episode_return = sum(
            rewards
        )

        print(
            "steps:",
            len(
                rewards
            ),
        )

        print(
            "loop87 breakdown:",
            loop87_breakdown,
        )

        print(
            "loop89 breakdown:",
            loop89_breakdown,
        )

        print(
            "episode return:",
            episode_return,
        )

        assert len(
            rewards
        ) == 38

        assert math.isclose(
            episode_return,
            -6.70804844703893,
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
                "converged"
            ]
        ) == 0

        assert int(
            env.current_state[
                "selection_success"
            ]
        ) == 0

        print()

        print(
            "PASS: LoopyCutsEnv emits "
            "Selection Reward V1 for "
            "Bracket terminal failure."
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
