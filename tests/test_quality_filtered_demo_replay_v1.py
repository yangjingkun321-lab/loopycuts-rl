import csv
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
    load_auxiliary_demo_replay,
    load_main_demo_replay,
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


QUALITY_FIELDS = [
    "quality_version",
    "model",
    "split",
    "raw_demo_status",
    "integrity_status",
    "demo_num_steps",
    "demo_outcome",
    "original_profile_status",
    "original_selection_success",
    "original_outcome",
    "demo_baseline_trajectory_match",
    "quality_role",
    "quality_reason",
    "main_demo_replay_eligible",
    "strong_bc_eligible",
    "auxiliary_rl_eligible",
    "demo_npz_file",
    "demo_metadata_file",
]


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

    return copy_observation(
        {
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
                    np.ones(
                        MAX_LOOPS,
                        dtype=np.bool_,
                    ),
            },

            "mask":
                mask,
        }
    )


def create_episode(
    *,
    raw_root,
    model,
    actions,
    outcome,
):
    observations = []

    for action in actions:
        observations.append(
            make_observation(
                legal_actions=[
                    action,
                ]
            )
        )

    observations.append(
        make_observation(
            legal_actions=[]
        )
    )

    num_steps = len(
        actions
    )

    rewards = (
        [
            -0.1,
        ]
        *
        (
            num_steps
            -
            1
        )
        +
        [
            (
                3.0
                if outcome == "FULL_HEX"
                else -3.0
            ),
        ]
    )

    terminated = (
        [
            False,
        ]
        *
        (
            num_steps
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
    ] * num_steps

    result = save_episode(
        output_dir=
            raw_root
            /
            model,

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

        audit_records=[
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
        ],

        finalization_outcome={
            "outcome":
                outcome,

            "attempted":
                True,

            "completed":
                True,

            "crashed":
                False,
        },

        initial_actionable=
            len(
                actions
            ),
    )

    return result


def make_quality_row(
    *,
    model,
    num_steps,
    outcome,
    role,
    main,
    strong_bc,
    auxiliary,
    result,
):
    return {
        "quality_version":
            "demo_quality_v1",

        "model":
            model,

        "split":
            "train",

        "raw_demo_status":
            "COLLECTED",

        "integrity_status":
            "VERIFIED",

        "demo_num_steps":
            str(
                num_steps
            ),

        "demo_outcome":
            outcome,

        "original_profile_status":
            "COMPLETE",

        "original_selection_success":
            (
                "1"
                if outcome == "FULL_HEX"
                else "0"
            ),

        "original_outcome":
            outcome,

        "demo_baseline_trajectory_match":
            "MATCH",

        "quality_role":
            role,

        "quality_reason":
            "SYNTHETIC",

        "main_demo_replay_eligible":
            str(
                int(
                    main
                )
            ),

        "strong_bc_eligible":
            str(
                int(
                    strong_bc
                )
            ),

        "auxiliary_rl_eligible":
            str(
                int(
                    auxiliary
                )
            ),

        "demo_npz_file":
            str(
                result[
                    "npz"
                ]
            ),

        "demo_metadata_file":
            str(
                result[
                    "metadata"
                ]
            ),
    }


def main():
    with TemporaryDirectory() as tmp:
        root = Path(
            tmp
        )

        raw_root = (
            root
            /
            "raw"
        )

        quality_path = (
            root
            /
            "demo_quality_v1.csv"
        )

        core_a = create_episode(
            raw_root=
                raw_root,

            model=
                "core_a",

            actions=[
                2,
                4,
            ],

            outcome=
                "FULL_HEX",
        )

        core_b = create_episode(
            raw_root=
                raw_root,

            model=
                "core_b",

            actions=[
                1,
            ],

            outcome=
                "FULL_HEX",
        )

        auxiliary = create_episode(
            raw_root=
                raw_root,

            model=
                "auxiliary",

            actions=[
                3,
                5,
            ],

            outcome=
                "NON_FULL_HEX",
        )

        rows = [
            make_quality_row(
                model=
                    "core_a",

                num_steps=
                    2,

                outcome=
                    "FULL_HEX",

                role=
                    "BC_CORE",

                main=
                    True,

                strong_bc=
                    True,

                auxiliary=
                    False,

                result=
                    core_a,
            ),

            make_quality_row(
                model=
                    "core_b",

                num_steps=
                    1,

                outcome=
                    "FULL_HEX",

                role=
                    "BC_CORE",

                main=
                    True,

                strong_bc=
                    True,

                auxiliary=
                    False,

                result=
                    core_b,
            ),

            make_quality_row(
                model=
                    "auxiliary",

                num_steps=
                    2,

                outcome=
                    "NON_FULL_HEX",

                role=
                    "RL_AUXILIARY",

                main=
                    False,

                strong_bc=
                    False,

                auxiliary=
                    True,

                result=
                    auxiliary,
            ),
        ]

        with quality_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=
                    QUALITY_FIELDS,

                lineterminator=
                    "\n",
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

        (
            main_buffer,
            main_records,
            main_provenance,
        ) = load_main_demo_replay(
            raw_root=
                raw_root,

            quality_manifest=
                quality_path,
        )

        assert len(
            main_buffer
        ) == 3

        assert (
            main_provenance[
                "models"
            ]
            ==
            [
                "core_a",
                "core_b",
            ]
        )

        assert all(
            record[
                "quality_role"
            ]
            ==
            "BC_CORE"
            for record
            in main_records
        )

        main_batch = main_buffer[
            main_buffer.sample_indices(
                0
            )
        ]

        assert (
            np.asarray(
                main_batch.act,
                dtype=np.int64,
            )
            .reshape(
                -1
            )
            .tolist()
            ==
            [
                2,
                4,
                1,
            ]
        )

        assert (
            np.asarray(
                main_batch.done,
                dtype=np.bool_,
            )
            .reshape(
                -1
            )
            .tolist()
            ==
            [
                False,
                True,
                True,
            ]
        )

        (
            aux_buffer,
            aux_records,
            aux_provenance,
        ) = load_auxiliary_demo_replay(
            raw_root=
                raw_root,

            quality_manifest=
                quality_path,
        )

        assert len(
            aux_buffer
        ) == 2

        assert (
            aux_provenance[
                "models"
            ]
            ==
            [
                "auxiliary",
            ]
        )

        assert (
            aux_records[
                0
            ][
                "quality_role"
            ]
            ==
            "RL_AUXILIARY"
        )

        aux_actions = (
            np.asarray(
                aux_buffer[
                    aux_buffer.sample_indices(
                        0
                    )
                ].act,
                dtype=np.int64,
            )
            .reshape(
                -1
            )
            .tolist()
        )

        assert (
            aux_actions
            ==
            [
                3,
                5,
            ]
        )

    print(
        "PASS: Demonstration Quality V1 "
        "filters main D_demo from "
        "RL_AUXILIARY replay"
    )


if __name__ == "__main__":
    main()
