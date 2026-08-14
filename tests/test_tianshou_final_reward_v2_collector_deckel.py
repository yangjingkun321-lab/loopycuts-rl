from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tianshou.data import (
    Batch,
    Collector,
    VectorReplayBuffer,
)

from tianshou.algorithm.random import (
    MARLRandomDiscreteMaskedOffPolicyAlgorithm,
)


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


OFFLINE_SCORED = Path(
    "/home/yjk/loopycuts_test/"
    "reward_v2_offline_audit/"
    "reward_v2_candidate_transition_scored.csv"
)


EXPECTED_ACTIONS = (
    list(
        range(
            22
        )
    )
    +
    [
        73,
    ]
)

EXPECTED_RETURN = (
    -4.957490718331989
)

EXPECTED_FINAL_REWARD = (
    -3.0511591131298803
)


class MinLegalMaskedPolicy(
    MARLRandomDiscreteMaskedOffPolicyAlgorithm.Policy
):
    """
    Deterministically select the minimum authoritative
    legal original LoopyCuts loop ID.

    This isolates:

        FinalRewardWrapper
        -> FinalizationEvalWrapper
        -> LoopyCutsEnv
        -> Tianshou Collector
        -> VectorReplayBuffer

    from learned-policy behavior.
    """

    def forward(
        self,
        batch,
        state=None,
        **kwargs,
    ):
        mask = np.asarray(
            batch.obs.mask,
            dtype=np.bool_,
        )

        if mask.ndim != 2:
            raise RuntimeError(
                "Expected batched mask "
                f"(B, 331), got {mask.shape}"
            )

        if mask.shape[
            1
        ] != 331:
            raise RuntimeError(
                "Expected action dimension 331, "
                f"got {mask.shape[1]}"
            )

        #
        # Collector must never ask the policy to act on the
        # terminal all-False observation.
        #
        if not bool(
            mask.any(
                axis=1
            ).all()
        ):
            raise RuntimeError(
                "Policy received terminal all-False mask"
            )

        actions = np.argmax(
            mask,
            axis=1,
        ).astype(
            np.int64
        )

        rows = np.arange(
            mask.shape[
                0
            ]
        )

        if not bool(
            mask[
                rows,
                actions,
            ].all()
        ):
            raise RuntimeError(
                "Policy selected illegal action"
            )

        return Batch(
            act=actions
        )


def load_offline_expected_rewards():
    import pandas as pd

    if not OFFLINE_SCORED.is_file():
        raise FileNotFoundError(
            OFFLINE_SCORED
        )

    df = pd.read_csv(
        OFFLINE_SCORED
    )

    df = (
        df[
            (
                df[
                    "profile"
                ]
                ==
                "balanced"
            )
            &
            (
                df[
                    "case"
                ]
                ==
                "deckel_original"
            )
        ]
        .sort_values(
            "step"
        )
    )

    if len(
        df
    ) != 23:
        raise RuntimeError(
            "Expected exactly 23 balanced "
            "Deckel offline transitions, "
            f"got {len(df)}"
        )

    expected_actions = (
        df[
            "loop_id"
        ]
        .astype(
            np.int64
        )
        .to_numpy()
    )

    expected_rewards = (
        df[
            "reward_v2"
        ]
        .astype(
            np.float64
        )
        .to_numpy()
    )

    return (
        expected_actions,
        expected_rewards,
    )


def main():
    (
        offline_actions,
        offline_rewards,
    ) = (
        load_offline_expected_rewards()
    )

    assert (
        offline_actions.tolist()
        ==
        EXPECTED_ACTIONS
    )

    assert np.isclose(
        float(
            offline_rewards.sum()
        ),
        EXPECTED_RETURN,
        rtol=0.0,
        atol=1e-12,
    )

    assert np.isclose(
        float(
            offline_rewards[
                -1
            ]
        ),
        EXPECTED_FINAL_REWARD,
        rtol=0.0,
        atol=1e-12,
    )

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

    policy = MinLegalMaskedPolicy(
        action_space=(
            env.action_space
        )
    )

    #
    # Observation V1 is relatively large, but Deckel has
    # only 23 transitions in the deterministic trajectory.
    #
    buffer = VectorReplayBuffer(
        total_size=32,
        buffer_num=1,
    )

    collector = Collector(
        policy=policy,
        env=env,
        buffer=buffer,
        exploration_noise=False,
        raise_on_nan_in_buffer=True,
    )

    try:
        stats = collector.collect(
            n_episode=1,
            reset_before_collect=True,
        )

        print(
            "Collected episodes:",
            stats.n_collected_episodes,
        )

        print(
            "Collected steps:",
            stats.n_collected_steps,
        )

        print(
            "Episode lengths:",
            stats.lens,
        )

        print(
            "Episode returns:",
            stats.returns,
        )

        assert (
            stats.n_collected_episodes
            ==
            1
        )

        assert (
            stats.n_collected_steps
            ==
            23
        )

        assert len(
            buffer
        ) == 23

        #
        # Tianshou's aggregate return may pass through its own
        # array dtype, so use a realistic numerical tolerance.
        #
        np.testing.assert_allclose(
            np.asarray(
                stats.returns,
                dtype=np.float64,
            ),
            np.asarray(
                [
                    EXPECTED_RETURN,
                ],
                dtype=np.float64,
            ),
            rtol=0.0,
            atol=1e-6,
        )

        indices = (
            buffer.sample_indices(
                0
            )
        )

        if len(
            indices
        ) != 23:
            raise RuntimeError(
                "ReplayBuffer did not contain "
                "exactly 23 valid transitions"
            )

        batch = buffer[
            indices
        ]

        actions = np.asarray(
            batch.act,
            dtype=np.int64,
        ).reshape(
            -1
        )

        rewards = np.asarray(
            batch.rew,
            dtype=np.float32,
        ).reshape(
            -1
        )

        terminated = np.asarray(
            batch.terminated,
            dtype=np.bool_,
        ).reshape(
            -1
        )

        truncated = np.asarray(
            batch.truncated,
            dtype=np.bool_,
        ).reshape(
            -1
        )

        obs_mask = np.asarray(
            batch.obs.mask,
            dtype=np.bool_,
        )

        next_mask = np.asarray(
            batch.obs_next.mask,
            dtype=np.bool_,
        )

        print()
        print(
            "Actions:",
            actions.tolist(),
        )

        print(
            "Rewards:",
            rewards.tolist(),
        )

        print(
            "Replay reward sum:",
            float(
                rewards.sum(
                    dtype=np.float64
                )
            ),
        )

        print(
            "Terminal reward:",
            float(
                rewards[
                    -1
                ]
            ),
        )

        print(
            "Terminated:",
            terminated.tolist(),
        )

        print(
            "Truncated:",
            truncated.tolist(),
        )

        print(
            "Final obs mask count:",
            int(
                obs_mask[
                    -1
                ].sum()
            ),
        )

        print(
            "Final obs_next mask count:",
            int(
                next_mask[
                    -1
                ].sum()
            ),
        )

        # ============================================================
        # Exact deterministic trajectory.
        # ============================================================

        assert (
            actions.tolist()
            ==
            EXPECTED_ACTIONS
        )

        np.testing.assert_array_equal(
            actions,
            offline_actions,
        )

        # ============================================================
        # Reward V2 transmission.
        #
        # ReplayBuffer stores float32, so compare against the
        # independently generated offline float64 ground truth
        # after converting it to the same representation.
        # ============================================================

        expected_rewards_f32 = (
            offline_rewards.astype(
                np.float32
            )
        )

        np.testing.assert_allclose(
            rewards,
            expected_rewards_f32,
            rtol=0.0,
            atol=2e-7,
        )

        assert np.isclose(
            float(
                rewards[
                    -1
                ]
            ),
            EXPECTED_FINAL_REWARD,
            rtol=0.0,
            atol=2e-7,
        )

        assert np.isclose(
            float(
                rewards.sum(
                    dtype=np.float64
                )
            ),
            EXPECTED_RETURN,
            rtol=0.0,
            atol=5e-7,
        )

        # ============================================================
        # Terminal semantics.
        # ============================================================

        expected_terminated = (
            [
                False,
            ]
            *
            22
            +
            [
                True,
            ]
        )

        assert (
            terminated.tolist()
            ==
            expected_terminated
        )

        assert not bool(
            truncated.any()
        )

        #
        # Every action must be legal in its own pre-action state.
        #
        assert obs_mask.shape == (
            23,
            331,
        )

        assert next_mask.shape == (
            23,
            331,
        )

        rows = np.arange(
            23
        )

        assert bool(
            obs_mask[
                rows,
                actions,
            ].all()
        )

        #
        # The final replay transition must contain the genuine
        # selection-terminal observation, not a post-finalization
        # pseudo-state and not a dummy action.
        #
        assert not bool(
            next_mask[
                -1
            ].any()
        )

        print()
        print(
            "PASS: Tianshou Collector -> ReplayBuffer "
            "stores Deckel Final-aware Reward V2 "
            "component-for-component against offline "
            "balanced ground truth, including the "
            "NON_FULL_HEX terminal consequence."
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
