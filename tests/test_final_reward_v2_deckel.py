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

MESH = (
    "/home/yjk/codes/LoopyCuts/"
    "test_data/deckel/"
    "deckel_rem_splitted.obj"
)

LOOPS = (
    "/home/yjk/codes/LoopyCuts/"
    "test_data/deckel/"
    "deckel_rem_loop.txt"
)


EXPECTED_V2 = (
    -4.957490718331989
)


def main():
    env = FinalRewardWrapper(
        FinalizationEvalWrapper(
            LoopyCutsEnv(
                executable=EXECUTABLE,
                mesh_file=MESH,
                loop_file=LOOPS,
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

        final_info = None

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

            assert truncated is False

            rewards.append(
                float(
                    reward
                )
            )

            if terminated:
                final_info = info

        total = float(
            sum(
                rewards
            )
        )

        if final_info is None:
            raise RuntimeError(
                "Deckel episode terminated without "
                "terminal info"
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

        print(
            "steps:",
            len(
                rewards
            )
        )

        print(
            "outcome:",
            outcome[
                "outcome"
            ]
        )

        print(
            "finalization reward:",
            breakdown[
                "finalization"
            ]
        )

        print(
            "terminal V2 reward:",
            breakdown[
                "total"
            ]
        )

        print(
            "episode V2 return:",
            total
        )

        print(
            "terminal Selection V1 reward:",
            final_info[
                "selection_reward_v1"
            ]
        )

        assert (
            len(
                rewards
            )
            ==
            23
        )

        assert (
            outcome[
                "outcome"
            ]
            ==
            "NON_FULL_HEX"
        )

        assert (
            outcome[
                "final_hex"
            ]
            ==
            512
        )

        assert (
            outcome[
                "final_total_polys"
            ]
            ==
            518
        )

        assert (
            breakdown[
                "finalization"
            ]
            ==
            -3.0
        )

        assert math.isclose(
            total,
            EXPECTED_V2,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        #
        # The real terminal transition must remain terminal and
        # preserve the genuine all-False selection action mask.
        #
        assert terminated is True

        assert (
            int(
                observation[
                    "mask"
                ].sum()
            )
            ==
            0
        )

        print()

        print(
            "PASS: Deckel online Reward V2 replaces "
            "selection-success proxy reward with real "
            "NON_FULL_HEX terminal outcome."
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
