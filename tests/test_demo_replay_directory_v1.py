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
    load_demo_directory,
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


def save_synthetic_episode(
    *,
    output_dir,
    model,
    actions,
):
    observations = []

    for step, action in enumerate(
        actions
    ):
        observations.append(
            make_observation(
                step=step,
                legal_actions=[
                    action,
                ],
            )
        )

    observations.append(
        make_observation(
            step=len(
                actions
            ),
            legal_actions=[],
        )
    )

    rewards = (
        [
            -0.1,
        ]
        *
        (
            len(
                actions
            )
            -
            1
        )
        +
        [
            3.0,
        ]
    )

    terminated = (
        [
            False,
        ]
        *
        (
            len(
                actions
            )
            -
            1
        )
        +
        [
            True,
        ]
    )

    truncated = [
        False,
    ] * len(
        actions
    )

    audit_records = [
        {
            "step":
                step,

            "action":
                int(
                    action
                ),
        }
        for step, action
        in enumerate(
            actions
        )
    ]

    return save_episode(
        output_dir=
            output_dir,

        model=
            model,

        split=
            "train",

        mesh_file=
            Path(
                f"/tmp/{model}.obj"
            ),

        loop_file=
            Path(
                f"/tmp/{model}_loop.txt"
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

        audit_records=
            audit_records,

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
            1,
    )


def main():
    with TemporaryDirectory() as tmp:
        root = Path(
            tmp
        )

        save_synthetic_episode(
            output_dir=
                root
                /
                "01_model_a",

            model=
                "model_a",

            actions=[
                2,
                4,
            ],
        )

        save_synthetic_episode(
            output_dir=
                root
                /
                "02_model_b",

            model=
                "model_b",

            actions=[
                1,
            ],
        )

        (
            buffer,
            records,
        ) = load_demo_directory(
            root=
                root
        )

        assert len(
            buffer
        ) == 3

        assert len(
            records
        ) == 2

        assert [
            record[
                "model"
            ]
            for record
            in records
        ] == [
            "model_a",
            "model_b",
        ]

        assert [
            record[
                "num_steps"
            ]
            for record
            in records
        ] == [
            2,
            1,
        ]

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
                2,
            ]
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

        done = np.asarray(
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

        assert (
            actions.tolist()
            ==
            [
                2,
                4,
                1,
            ]
        )

        assert (
            terminated.tolist()
            ==
            [
                False,
                True,
                True,
            ]
        )

        assert (
            done.tolist()
            ==
            [
                False,
                True,
                True,
            ]
        )

        assert not bool(
            truncated.any()
        )

        rows = np.arange(
            len(
                actions
            )
        )

        assert bool(
            obs_mask[
                rows,
                actions,
            ].all()
        )

        # End of episode A.
        assert not bool(
            next_mask[
                1
            ].any()
        )

        # End of episode B.
        assert not bool(
            next_mask[
                2
            ].any()
        )

        # Tianshou temporal navigation must not cross
        # from episode A into episode B.
        previous_of_episode_b = (
            buffer.prev(
                np.asarray(
                    [
                        2,
                    ],
                    dtype=np.int64,
                )
            )
        )

        assert (
            previous_of_episode_b.tolist()
            ==
            [
                2,
            ]
        )

        next_of_episode_a_terminal = (
            buffer.next(
                np.asarray(
                    [
                        1,
                    ],
                    dtype=np.int64,
                )
            )
        )

        assert (
            next_of_episode_a_terminal.tolist()
            ==
            [
                1,
            ]
        )

    print(
        "PASS: multi-episode "
        "Demonstration V1 directory -> D_demo "
        "preserves episode boundaries"
    )


if __name__ == "__main__":
    main()
