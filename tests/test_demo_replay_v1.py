import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from imitation.demo_replay import (
    load_demo_episode_into_replay,
)

from imitation.demo_v1 import (
    copy_observation,
    save_episode,
)

from observation.builder import (
    GLOBAL_DIM,
    LOOP_FEATURE_DIM,
    MAX_LOOPS,
)


def make_observation(
    *,
    step,
    legal_actions,
):
    mask = np.zeros(
        MAX_LOOPS,
        dtype=np.bool_,
    )

    if legal_actions:
        mask[
            np.asarray(
                legal_actions,
                dtype=np.int64,
            )
        ] = True

    global_features = np.zeros(
        GLOBAL_DIM,
        dtype=np.float32,
    )

    global_features[
        0
    ] = np.float32(
        step
    )

    loop_features = np.zeros(
        (
            MAX_LOOPS,
            LOOP_FEATURE_DIM,
        ),
        dtype=np.float32,
    )

    loop_features[
        :5,
        0,
    ] = np.arange(
        5,
        dtype=np.float32,
    )

    exists = np.zeros(
        MAX_LOOPS,
        dtype=np.bool_,
    )

    exists[
        :5
    ] = True

    return copy_observation(
        {
            "obs": {
                "global":
                    global_features,

                "loops":
                    loop_features,

                "exists":
                    exists,
            },

            "mask":
                mask,
        }
    )


def main():
    observations = [
        make_observation(
            step=0,
            legal_actions=[
                2,
                4,
            ],
        ),

        make_observation(
            step=1,
            legal_actions=[
                4,
            ],
        ),

        make_observation(
            step=2,
            legal_actions=[],
        ),
    ]

    actions = np.asarray(
        [
            2,
            4,
        ],
        dtype=np.int64,
    )

    rewards = np.asarray(
        [
            -0.25,
            3.0,
        ],
        dtype=np.float32,
    )

    terminated = np.asarray(
        [
            False,
            True,
        ],
        dtype=np.bool_,
    )

    truncated = np.asarray(
        [
            False,
            False,
        ],
        dtype=np.bool_,
    )

    with TemporaryDirectory() as tmp:
        result = save_episode(
            output_dir=
                Path(
                    tmp
                ),

            model=
                "synthetic",

            split=
                "train",

            mesh_file=
                Path(
                    "/tmp/fake.obj"
                ),

            loop_file=
                Path(
                    "/tmp/fake_loop.txt"
                ),

            source_git_commit=
                "synthetic",

            observations=
                observations,

            actions=
                actions,

            rewards=
                rewards,

            terminated=
                terminated,

            truncated=
                truncated,

            audit_records=[
                {
                    "step":
                        0,
                    "action":
                        2,
                },
                {
                    "step":
                        1,
                    "action":
                        4,
                },
            ],

            finalization_outcome={
                "outcome":
                    "FULL_HEX",

                "attempted":
                    True,

                "completed":
                    True,

                "crashed":
                    False,
            },

            initial_actionable=
                2,
        )

        (
            buffer,
            metadata,
        ) = (
            load_demo_episode_into_replay(
                npz_path=
                    result[
                        "npz"
                    ],

                metadata_path=
                    result[
                        "metadata"
                    ],
            )
        )

        assert len(
            buffer
        ) == 2

        indices = (
            buffer.sample_indices(
                0
            )
        )

        assert (
            indices.tolist()
            ==
            [
                0,
                1,
            ]
        )

        batch = buffer[
            indices
        ]

        loaded_actions = np.asarray(
            batch.act,
            dtype=np.int64,
        ).reshape(
            -1
        )

        loaded_rewards = np.asarray(
            batch.rew,
            dtype=np.float32,
        ).reshape(
            -1
        )

        loaded_terminated = np.asarray(
            batch.terminated,
            dtype=np.bool_,
        ).reshape(
            -1
        )

        loaded_truncated = np.asarray(
            batch.truncated,
            dtype=np.bool_,
        ).reshape(
            -1
        )

        loaded_done = np.asarray(
            batch.done,
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

        obs_global = np.asarray(
            batch.obs.obs["global"],
            dtype=np.float32,
        )

        next_global = np.asarray(
            batch.obs_next.obs["global"],
            dtype=np.float32,
        )

        assert (
            loaded_actions.tolist()
            ==
            [
                2,
                4,
            ]
        )

        np.testing.assert_array_equal(
            loaded_rewards,
            rewards,
        )

        assert (
            loaded_terminated.tolist()
            ==
            [
                False,
                True,
            ]
        )

        assert (
            loaded_truncated.tolist()
            ==
            [
                False,
                False,
            ]
        )

        assert (
            loaded_done.tolist()
            ==
            [
                False,
                True,
            ]
        )

        assert (
            obs_mask.shape
            ==
            (
                2,
                MAX_LOOPS,
            )
        )

        assert (
            next_mask.shape
            ==
            (
                2,
                MAX_LOOPS,
            )
        )

        rows = np.arange(
            2
        )

        assert bool(
            obs_mask[
                rows,
                loaded_actions,
            ].all()
        )

        assert not bool(
            next_mask[
                -1
            ].any()
        )

        assert (
            obs_global.shape
            ==
            (
                2,
                GLOBAL_DIM,
            )
        )

        assert (
            next_global.shape
            ==
            (
                2,
                GLOBAL_DIM,
            )
        )

        assert np.isclose(
            float(
                obs_global[
                    0,
                    0,
                ]
            ),
            0.0,
        )

        assert np.isclose(
            float(
                next_global[
                    0,
                    0,
                ]
            ),
            1.0,
        )

        assert (
            metadata[
                "model"
            ]
            ==
            "synthetic"
        )

    print(
        "PASS: Demonstration V1 -> "
        "Tianshou ReplayBuffer"
    )


if __name__ == "__main__":
    main()
