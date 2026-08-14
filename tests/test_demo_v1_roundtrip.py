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

from imitation.demo_v1 import (
    copy_observation,
    load_episode,
    save_episode,
)

from observation.builder import (
    GLOBAL_DIM,
    LOOP_FEATURE_DIM,
    MAX_LOOPS,
)


def make_observation(
    *,
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

    exists = np.zeros(
        MAX_LOOPS,
        dtype=np.bool_,
    )

    exists[
        :5
    ] = True

    return {
        "obs": {
            "global":
                np.zeros(
                    GLOBAL_DIM,
                    dtype=np.float32,
                ),

            "loops":
                np.zeros(
                    (
                        MAX_LOOPS,
                        LOOP_FEATURE_DIM,
                    ),
                    dtype=np.float32,
                ),

            "exists":
                exists,
        },

        "mask":
            mask,
    }


def main():
    observations = [
        copy_observation(
            make_observation(
                legal_actions=[
                    2,
                    4,
                ]
            )
        ),

        copy_observation(
            make_observation(
                legal_actions=[
                    4,
                ]
            )
        ),

        copy_observation(
            make_observation(
                legal_actions=[]
            )
        ),
    ]

    actions = [
        2,
        4,
    ]

    rewards = [
        -0.1,
        3.0,
    ]

    terminated = [
        False,
        True,
    ]

    truncated = [
        False,
        False,
    ]

    audit_records = [
        {
            "step":
                0,

            "action":
                2,

            "status":
                "COMMITTED",
        },

        {
            "step":
                1,

            "action":
                4,

            "status":
                "COMMITTED",
        },
    ]

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
                2,
        )

        loaded = load_episode(
            result[
                "npz"
            ]
        )

        assert (
            loaded[
                "obs_global"
            ].shape
            ==
            (
                3,
                GLOBAL_DIM,
            )
        )

        assert (
            loaded[
                "obs_loops"
            ].shape
            ==
            (
                3,
                MAX_LOOPS,
                LOOP_FEATURE_DIM,
            )
        )

        assert (
            loaded[
                "obs_exists"
            ].shape
            ==
            (
                3,
                MAX_LOOPS,
            )
        )

        assert (
            loaded[
                "obs_mask"
            ].shape
            ==
            (
                3,
                MAX_LOOPS,
            )
        )

        assert (
            loaded[
                "actions"
            ].tolist()
            ==
            [
                2,
                4,
            ]
        )

        assert (
            loaded[
                "terminated"
            ].tolist()
            ==
            [
                False,
                True,
            ]
        )

        assert (
            loaded[
                "truncated"
            ].tolist()
            ==
            [
                False,
                False,
            ]
        )

        assert not np.any(
            loaded[
                "obs_mask"
            ][
                -1
            ]
        )

        assert (
            result[
                "record"
            ][
                "schema_version"
            ]
            ==
            "loopycuts_demo_episode_v1"
        )

        assert (
            result[
                "record"
            ][
                "quality_class"
            ]
            ==
            "UNCLASSIFIED"
        )

    print(
        "PASS: Demo V1 round-trip"
    )


if __name__ == "__main__":
    main()
