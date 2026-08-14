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

LOOP_FILE = (
    "/home/yjk/loopycuts_inputs/"
    "cylinder_plate_clean/"
    "cylinder_plate_rem_loop.txt"
)


class MinLegalMaskedPolicy(
    MARLRandomDiscreteMaskedOffPolicyAlgorithm.Policy
):
    """
    Deterministic Collector smoke-test policy.

    It deliberately has no learned parameters.

    For every batched observation:
        choose the minimum original loop ID whose
        authoritative C++ action mask is True.

    This isolates:
        LoopyCutsEnv
        -> Tianshou batching
        -> policy mask consumption
        -> Collector
        -> ReplayBuffer

    from any RL-learning behavior.
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
                "Expected batched action mask with shape "
                "(batch_size, 331), "
                f"got {mask.shape}"
            )

        if mask.shape[1] != 331:
            raise RuntimeError(
                "Expected action dimension 331, "
                f"got {mask.shape[1]}"
            )

        #
        # Policy should never be asked to select an action
        # from a terminal all-False observation.
        #
        if not np.all(
            mask.any(
                axis=1
            )
        ):
            raise RuntimeError(
                "Collector asked policy to act on "
                "an all-False terminal mask"
            )

        #
        # np.argmax on bool chooses the first True entry,
        # i.e. the minimum currently legal original loop ID.
        #
        actions = np.argmax(
            mask,
            axis=1,
        ).astype(
            np.int64
        )

        #
        # Defensive legality check.
        #
        rows = np.arange(
            mask.shape[0]
        )

        if not np.all(
            mask[
                rows,
                actions,
            ]
        ):
            raise RuntimeError(
                "Policy produced an illegal action"
            )

        return Batch(
            act=actions
        )


def main():
    env = LoopyCutsEnv(
        executable=EXECUTABLE,
        mesh_file=MESH,
        loop_file=LOOP_FILE,
        echo_logs=False,
    )

    policy = MinLegalMaskedPolicy(
        action_space=env.action_space
    )

    #
    # Keep this intentionally tiny.
    #
    # Cylinder original-order episode has only four transitions.
    # A small VectorReplayBuffer also avoids allocating unnecessary
    # storage for our relatively large nested observation.
    #
    buffer = VectorReplayBuffer(
        total_size=16,
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
        # ============================================================
        # Collect exactly one complete Cylinder episode.
        # ============================================================

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
            4
        )

        np.testing.assert_allclose(
            np.asarray(
                stats.returns,
                dtype=np.float64,
            ),
            np.asarray(
                [
                    2.4865046601979746,
                ],
                dtype=np.float64,
            ),
            rtol=0.0,
            atol=2e-7,
        )

        assert len(buffer) == 4

        # ============================================================
        # Retrieve exactly the valid transitions.
        # ============================================================

        indices = (
            buffer.sample_indices(
                0
            )
        )

        print(
            "Buffer indices:",
            indices,
        )

        assert len(indices) == 4

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
            "Terminated:",
            terminated.tolist(),
        )

        print(
            "Truncated:",
            truncated.tolist(),
        )

        print(
            "obs mask counts:",
            obs_mask.sum(
                axis=1
            ).tolist(),
        )

        print(
            "obs_next mask counts:",
            next_mask.sum(
                axis=1
            ).tolist(),
        )

        # ============================================================
        # Deterministic original-order Cylinder regression.
        # ============================================================

        assert actions.tolist() == [
            0,
            1,
            2,
            3,
        ]

        expected_rewards = np.asarray(
            [
                -0.14250819235093484,
                -0.17114110262030834,
                -0.10640468897629546,
                2.9065586441455135,
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            rewards,
            expected_rewards,
            rtol=0.0,
            atol=2e-7,
        )

        assert np.isclose(
            float(
                rewards.sum(
                    dtype=np.float64
                )
            ),
            2.4865046601979746,
            rtol=0.0,
            atol=2e-7,
        )

        assert terminated.tolist() == [
            False,
            False,
            False,
            True,
        ]

        assert not bool(
            truncated.any()
        )

        # ============================================================
        # Current-state mask before each action.
        # ============================================================

        assert obs_mask.shape == (
            4,
            331,
        )

        assert (
            obs_mask.sum(
                axis=1
            ).tolist()
            ==
            [
                65,
                64,
                63,
                62,
            ]
        )

        #
        # Every collected action must have been legal in its
        # corresponding pre-action observation.
        #
        rows = np.arange(
            4
        )

        assert bool(
            obs_mask[
                rows,
                actions,
            ].all()
        )

        # ============================================================
        # Post-step masks.
        #
        # Final transition must preserve the REAL environment mask:
        #
        #     all False
        #
        # No terminal dummy-action fallback belongs in ReplayBuffer.
        # ============================================================

        assert next_mask.shape == (
            4,
            331,
        )

        assert (
            next_mask.sum(
                axis=1
            ).tolist()
            ==
            [
                64,
                63,
                62,
                0,
            ]
        )

        assert not bool(
            next_mask[
                -1
            ].any()
        )

        # ============================================================
        # Verify Tianshou preserved the complete nested Observation V1.
        # ============================================================

        obs_global = np.asarray(
            batch.obs.obs["global"],
            dtype=np.float32,
        )

        obs_loops = np.asarray(
            batch.obs.obs.loops,
            dtype=np.float32,
        )

        obs_exists = np.asarray(
            batch.obs.obs.exists,
            dtype=np.bool_,
        )

        next_global = np.asarray(
            batch.obs_next.obs["global"],
            dtype=np.float32,
        )

        next_loops = np.asarray(
            batch.obs_next.obs.loops,
            dtype=np.float32,
        )

        next_exists = np.asarray(
            batch.obs_next.obs.exists,
            dtype=np.bool_,
        )

        assert obs_global.shape == (
            4,
            16,
        )

        assert obs_loops.shape == (
            4,
            331,
            14,
        )

        assert obs_exists.shape == (
            4,
            331,
        )

        assert next_global.shape == (
            4,
            16,
        )

        assert next_loops.shape == (
            4,
            331,
            14,
        )

        assert next_exists.shape == (
            4,
            331,
        )

        assert np.isfinite(
            obs_global
        ).all()

        assert np.isfinite(
            obs_loops
        ).all()

        assert np.isfinite(
            next_global
        ).all()

        assert np.isfinite(
            next_loops
        ).all()

        # ============================================================
        # Terminal next observation must still contain the genuine
        # Cylinder terminal semantic state:
        #
        # global[4] = converged
        # global[5] = regular_phase_closed
        # ============================================================

        terminal_next_global = (
            next_global[
                -1
            ]
        )

        assert float(
            terminal_next_global[
                4
            ]
        ) == 1.0

        assert float(
            terminal_next_global[
                5
            ]
        ) == 1.0

        # ============================================================
        # Exists is independent of action legality throughout.
        # Cylinder always has exactly 91 serialized loops.
        # ============================================================

        assert (
            obs_exists.sum(
                axis=1
            ).tolist()
            ==
            [
                91,
                91,
                91,
                91,
            ]
        )

        assert (
            next_exists.sum(
                axis=1
            ).tolist()
            ==
            [
                91,
                91,
                91,
                91,
            ]
        )

        print()

        print(
            "PASS: Tianshou Collector -> "
            "LoopyCutsEnv -> VectorReplayBuffer "
            "preserves nested Observation V1, "
            "legal original loop IDs, and "
            "terminal all-False obs_next mask."
        )

    finally:
        collector.close()


if __name__ == "__main__":
    main()
