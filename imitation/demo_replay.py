from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from tianshou.data import (
    Batch,
    ReplayBuffer,
)

from imitation.demo_v1 import (
    OBSERVATION_VERSION,
    REWARD_VERSION,
    SCHEMA_VERSION,
    TEACHER_VERSION,
    load_episode,
    sha256_file,
)


class DemoReplayError(
    RuntimeError
):
    pass


def _observation_batch(
    *,
    global_features,
    loop_features,
    exists,
    mask,
):
    """
    Convert one stored Observation V1 state into the exact nested
    structure expected by the LoopyCuts Tianshou policy path:

        observation
        ├── obs
        │   ├── global
        │   ├── loops
        │   └── exists
        └── mask

    "global" is a Python keyword, so it must be stored as a
    dictionary/Batch key rather than used as a keyword argument.
    """

    inner_observation = Batch(
        {
            "global":
                np.asarray(
                    global_features,
                    dtype=np.float32,
                ),

            "loops":
                np.asarray(
                    loop_features,
                    dtype=np.float32,
                ),

            "exists":
                np.asarray(
                    exists,
                    dtype=np.bool_,
                ),
        }
    )

    return Batch(
        obs=
            inner_observation,

        mask=
            np.asarray(
                mask,
                dtype=np.bool_,
            ),
    )

def _validate_metadata(
    *,
    metadata_path: Path,
    npz_path: Path,
):
    metadata_path = Path(
        metadata_path
    )

    npz_path = Path(
        npz_path
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            metadata_path
        )

    if not npz_path.is_file():
        raise FileNotFoundError(
            npz_path
        )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    required = {
        "schema_version",
        "observation_version",
        "reward_version",
        "teacher_version",
        "model",
        "split",
        "num_steps",
        "npz_file",
        "npz_sha256",
        "quality_class",
        "finalization_outcome",
    }

    missing = (
        required
        -
        set(metadata)
    )

    if missing:
        raise DemoReplayError(
            "Demonstration metadata is missing fields: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )

    expected_versions = {
        "schema_version":
            SCHEMA_VERSION,

        "observation_version":
            OBSERVATION_VERSION,

        "reward_version":
            REWARD_VERSION,

        "teacher_version":
            TEACHER_VERSION,
    }

    for (
        key,
        expected,
    ) in expected_versions.items():
        actual = metadata[
            key
        ]

        if actual != expected:
            raise DemoReplayError(
                f"{key} mismatch: "
                f"expected={expected!r}, "
                f"actual={actual!r}"
            )

    if (
        metadata[
            "npz_file"
        ]
        !=
        npz_path.name
    ):
        raise DemoReplayError(
            "Metadata npz_file does not match "
            "the supplied NPZ path"
        )

    actual_sha256 = (
        sha256_file(
            npz_path
        )
    )

    if (
        actual_sha256
        !=
        metadata[
            "npz_sha256"
        ]
    ):
        raise DemoReplayError(
            "Demonstration NPZ checksum mismatch"
        )

    return metadata


def add_episode_to_buffer(
    *,
    buffer: ReplayBuffer,
    data: dict,
):
    """
    Add exactly one complete Demo V1 episode to an existing
    Tianshou ReplayBuffer.

    The stored expert action is simply `batch.act`.
    No duplicate expert-action field is required.
    """

    num_steps = int(
        data[
            "actions"
        ].shape[
            0
        ]
    )

    if (
        buffer.maxsize
        <
        len(buffer)
        +
        num_steps
    ):
        raise DemoReplayError(
            "ReplayBuffer is too small for this "
            "demonstration episode; refusing to "
            "overwrite earlier demonstrations"
        )

    for step in range(
        num_steps
    ):
        obs = _observation_batch(
            global_features=
                data[
                    "obs_global"
                ][
                    step
                ],

            loop_features=
                data[
                    "obs_loops"
                ][
                    step
                ],

            exists=
                data[
                    "obs_exists"
                ][
                    step
                ],

            mask=
                data[
                    "obs_mask"
                ][
                    step
                ],
        )

        obs_next = (
            _observation_batch(
                global_features=
                    data[
                        "obs_global"
                    ][
                        step
                        +
                        1
                    ],

                loop_features=
                    data[
                        "obs_loops"
                    ][
                        step
                        +
                        1
                    ],

                exists=
                    data[
                        "obs_exists"
                    ][
                        step
                        +
                        1
                    ],

                mask=
                    data[
                        "obs_mask"
                    ][
                        step
                        +
                        1
                    ],
            )
        )

        transition = Batch(
            obs=
                obs,

            act=np.int64(
                data[
                    "actions"
                ][
                    step
                ]
            ),

            rew=np.float32(
                data[
                    "rewards"
                ][
                    step
                ]
            ),

            terminated=np.bool_(
                data[
                    "terminated"
                ][
                    step
                ]
            ),

            truncated=np.bool_(
                data[
                    "truncated"
                ][
                    step
                ]
            ),

            obs_next=
                obs_next,
        )

        buffer.add(
            transition
        )

    return num_steps


def load_demo_episode_into_replay(
    *,
    npz_path: Path,
    metadata_path: Path,
    buffer: ReplayBuffer | None = None,
):
    """
    Validate one Demonstration V1 artifact and load it into a
    Tianshou ReplayBuffer.

    If no buffer is supplied, create an exactly-sized ReplayBuffer.
    """

    npz_path = Path(
        npz_path
    )

    metadata_path = Path(
        metadata_path
    )

    metadata = (
        _validate_metadata(
            metadata_path=
                metadata_path,

            npz_path=
                npz_path,
        )
    )

    data = load_episode(
        npz_path
    )

    num_steps = int(
        data[
            "actions"
        ].shape[
            0
        ]
    )

    if (
        int(
            metadata[
                "num_steps"
            ]
        )
        !=
        num_steps
    ):
        raise DemoReplayError(
            "Metadata num_steps does not match "
            "the NPZ transition count"
        )

    if buffer is None:
        buffer = ReplayBuffer(
            size=
                num_steps
        )

    added = (
        add_episode_to_buffer(
            buffer=
                buffer,

            data=
                data,
        )
    )

    if added != num_steps:
        raise DemoReplayError(
            "Unexpected demonstration "
            "transition count"
        )

    return (
        buffer,
        metadata,
    )


def load_demo_directory(
    *,
    root: Path,
):
    """
    Load every Demonstration V1 episode under a directory tree into
    one shared D_demo ReplayBuffer.

    Expected artifacts are paired as:

        *_original_demo_v1.npz
        *_original_demo_v1.json

    This function deliberately does not apply quality filtering yet.
    quality_class remains metadata for the next phase.
    """

    root = Path(
        root
    )

    if not root.is_dir():
        raise NotADirectoryError(
            root
        )

    npz_paths = sorted(
        root.rglob(
            "*_original_demo_v1.npz"
        )
    )

    if not npz_paths:
        raise DemoReplayError(
            f"No Demonstration V1 NPZ files found under {root}"
        )

    episodes = []

    total_steps = 0

    for npz_path in npz_paths:
        metadata_path = (
            npz_path.with_suffix(
                ".json"
            )
        )

        metadata = (
            _validate_metadata(
                metadata_path=
                    metadata_path,

                npz_path=
                    npz_path,
            )
        )

        data = load_episode(
            npz_path
        )

        num_steps = int(
            data[
                "actions"
            ].shape[
                0
            ]
        )

        if (
            int(
                metadata[
                    "num_steps"
            ]
            )
            !=
            num_steps
        ):
            raise DemoReplayError(
                f"{npz_path}: metadata/NPZ "
                "step-count mismatch"
            )

        episodes.append(
            (
                npz_path,
                metadata,
                data,
            )
        )

        total_steps += (
            num_steps
        )

    if total_steps <= 0:
        raise DemoReplayError(
            "Demonstration directory contains "
            "no transitions"
        )

    buffer = ReplayBuffer(
        size=
            total_steps
    )

    records = []

    for (
        npz_path,
        metadata,
        data,
    ) in episodes:
        added = (
            add_episode_to_buffer(
                buffer=
                    buffer,

                data=
                    data,
            )
        )

        records.append(
            {
                "model":
                    metadata[
                        "model"
                    ],

                "split":
                    metadata[
                        "split"
                    ],

                "quality_class":
                    metadata[
                        "quality_class"
                    ],

                "outcome":
                    metadata[
                        "finalization_outcome"
                    ][
                        "outcome"
                    ],

                "num_steps":
                    added,

                "npz_path":
                    str(
                        npz_path
                    ),
            }
        )

    if (
        len(
            buffer
        )
        !=
        total_steps
    ):
        raise DemoReplayError(
            "Final ReplayBuffer length does "
            "not match total demonstration "
            "transition count"
        )

    return (
        buffer,
        records,
    )


# ================================================================
# Demonstration Quality V1 filtered replay loading.
#
# IMPORTANT:
#
# `load_demo_directory()` above remains the raw-data audit loader.
# It intentionally loads every raw demonstration found under the
# supplied directory.
#
# Formal training must instead use the quality-filtered entry
# points below.
# ================================================================


DEMO_QUALITY_VERSION = (
    "demo_quality_v1"
)

MAIN_DEMO_ELIGIBILITY_FIELD = (
    "main_demo_replay_eligible"
)

AUXILIARY_DEMO_ELIGIBILITY_FIELD = (
    "auxiliary_rl_eligible"
)


def _load_quality_manifest_rows(
    *,
    quality_manifest: Path,
):
    quality_manifest = Path(
        quality_manifest
    )

    if not quality_manifest.is_file():
        raise FileNotFoundError(
            quality_manifest
        )

    with quality_manifest.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = [
            dict(row)
            for row
            in csv.DictReader(f)
        ]

    if not rows:
        raise DemoReplayError(
            "Demonstration quality manifest "
            "contains no rows"
        )

    required = {
        "quality_version",
        "model",
        "split",
        "raw_demo_status",
        "integrity_status",
        "demo_num_steps",
        "demo_outcome",
        "demo_baseline_trajectory_match",
        "quality_role",
        "main_demo_replay_eligible",
        "strong_bc_eligible",
        "auxiliary_rl_eligible",
        "demo_npz_file",
        "demo_metadata_file",
    }

    missing = (
        required
        -
        set(
            rows[
                0
            ]
        )
    )

    if missing:
        raise DemoReplayError(
            "Demonstration quality manifest "
            "is missing columns: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )

    models = [
        row[
            "model"
        ]
        for row
        in rows
    ]

    if len(
        models
    ) != len(
        set(
            models
        )
    ):
        raise DemoReplayError(
            "Demonstration quality manifest "
            "contains duplicate models"
        )

    for row in rows:
        if (
            row[
                "quality_version"
            ]
            !=
            DEMO_QUALITY_VERSION
        ):
            raise DemoReplayError(
                "Unsupported Demonstration "
                "Quality version for "
                f"model={row['model']}: "
                f"{row['quality_version']!r}"
            )

        if (
            row[
                "split"
            ]
            !=
            "train"
        ):
            raise DemoReplayError(
                "Formal D_demo may only be "
                "constructed from split=train; "
                f"got model={row['model']} "
                f"split={row['split']!r}"
            )

        for field in [
            "main_demo_replay_eligible",
            "strong_bc_eligible",
            "auxiliary_rl_eligible",
        ]:
            if (
                row[
                    field
                ]
                not in {
                    "0",
                    "1",
                }
            ):
                raise DemoReplayError(
                    f"{row['model']}: invalid "
                    f"{field}={row[field]!r}"
                )

    return rows


def _validate_quality_eligible_row(
    *,
    row,
    eligibility_field,
):
    model = row[
        "model"
    ]

    if (
        row[
            "raw_demo_status"
        ]
        !=
        "COLLECTED"
    ):
        raise DemoReplayError(
            f"{model}: eligible demonstration "
            "is not COLLECTED"
        )

    if (
        row[
            "integrity_status"
        ]
        !=
        "VERIFIED"
    ):
        raise DemoReplayError(
            f"{model}: eligible demonstration "
            "does not have VERIFIED integrity"
        )

    if (
        row[
            "demo_baseline_trajectory_match"
        ]
        !=
        "MATCH"
    ):
        raise DemoReplayError(
            f"{model}: eligible demonstration "
            "does not MATCH the frozen "
            "Original baseline"
        )

    if (
        eligibility_field
        ==
        MAIN_DEMO_ELIGIBILITY_FIELD
    ):
        if (
            row[
                "quality_role"
            ]
            !=
            "BC_CORE"
        ):
            raise DemoReplayError(
                f"{model}: main D_demo "
                "eligibility requires "
                "quality_role=BC_CORE"
            )

        if (
            row[
                "strong_bc_eligible"
            ]
            !=
            "1"
        ):
            raise DemoReplayError(
                f"{model}: main D_demo "
                "eligibility requires "
                "strong_bc_eligible=1"
            )

        if (
            row[
                "auxiliary_rl_eligible"
            ]
            !=
            "0"
        ):
            raise DemoReplayError(
                f"{model}: BC_CORE row must "
                "not also be auxiliary RL "
                "eligible"
            )

    elif (
        eligibility_field
        ==
        AUXILIARY_DEMO_ELIGIBILITY_FIELD
    ):
        if (
            row[
                "quality_role"
            ]
            !=
            "RL_AUXILIARY"
        ):
            raise DemoReplayError(
                f"{model}: auxiliary replay "
                "eligibility requires "
                "quality_role=RL_AUXILIARY"
            )

        if (
            row[
                "main_demo_replay_eligible"
            ]
            !=
            "0"
        ):
            raise DemoReplayError(
                f"{model}: RL_AUXILIARY row "
                "must not be main D_demo "
                "eligible"
            )

        if (
            row[
                "strong_bc_eligible"
            ]
            !=
            "0"
        ):
            raise DemoReplayError(
                f"{model}: RL_AUXILIARY row "
                "must not be strong-BC "
                "eligible"
            )

    else:
        raise DemoReplayError(
            "Unsupported eligibility field: "
            f"{eligibility_field!r}"
        )


def load_quality_filtered_demo_replay(
    *,
    raw_root: Path,
    quality_manifest: Path,
    eligibility_field: str,
    random_seed: int = 42,
):
    """
    Construct a Tianshou ReplayBuffer using only demonstrations
    explicitly allowed by Demonstration Quality V1.

    This is the formal training loader.

    It does NOT mutate raw Demonstration V1 metadata.
    It does NOT infer quality from outcome at training time.
    The frozen quality manifest is authoritative.
    """

    raw_root = Path(
        raw_root
    )

    quality_manifest = Path(
        quality_manifest
    )

    if not raw_root.is_dir():
        raise NotADirectoryError(
            raw_root
        )

    if (
        eligibility_field
        not in {
            MAIN_DEMO_ELIGIBILITY_FIELD,
            AUXILIARY_DEMO_ELIGIBILITY_FIELD,
        }
    ):
        raise DemoReplayError(
            "Unsupported demonstration "
            "eligibility field: "
            f"{eligibility_field!r}"
        )

    rows = (
        _load_quality_manifest_rows(
            quality_manifest=
                quality_manifest,
        )
    )

    selected_rows = [
        row
        for row
        in rows
        if (
            row[
                eligibility_field
            ]
            ==
            "1"
        )
    ]

    if not selected_rows:
        raise DemoReplayError(
            "No demonstrations are eligible "
            f"for {eligibility_field}"
        )

    episodes = []

    total_steps = 0

    resolved_root = (
        raw_root.resolve()
    )

    for row in selected_rows:
        _validate_quality_eligible_row(
            row=
                row,

            eligibility_field=
                eligibility_field,
        )

        model = row[
            "model"
        ]

        expected_dir = (
            raw_root
            /
            model
        )

        expected_stem = (
            f"{model}_"
            "original_demo_v1"
        )

        expected_npz_path = (
            expected_dir
            /
            f"{expected_stem}.npz"
        )

        expected_metadata_path = (
            expected_dir
            /
            f"{expected_stem}.json"
        )

        row_npz_path = Path(
            row[
                "demo_npz_file"
            ]
        )

        row_metadata_path = Path(
            row[
                "demo_metadata_file"
            ]
        )

        if (
            row_npz_path.resolve()
            !=
            expected_npz_path.resolve()
        ):
            raise DemoReplayError(
                f"{model}: quality manifest "
                "NPZ path does not match "
                "the requested raw root"
            )

        if (
            row_metadata_path.resolve()
            !=
            expected_metadata_path.resolve()
        ):
            raise DemoReplayError(
                f"{model}: quality manifest "
                "metadata path does not match "
                "the requested raw root"
            )

        # Refuse artifacts outside the requested raw dataset root.
        for artifact in [
            expected_npz_path,
            expected_metadata_path,
        ]:
            try:
                artifact.resolve().relative_to(
                    resolved_root
                )

            except ValueError as exc:
                raise DemoReplayError(
                    f"{model}: demonstration "
                    "artifact escapes raw_root"
                ) from exc

        metadata = (
            _validate_metadata(
                metadata_path=
                    expected_metadata_path,

                npz_path=
                    expected_npz_path,
            )
        )

        data = load_episode(
            expected_npz_path
        )

        num_steps = int(
            data[
                "actions"
            ].shape[
                0
            ]
        )

        if (
            num_steps
            !=
            int(
                row[
                    "demo_num_steps"
                ]
            )
        ):
            raise DemoReplayError(
                f"{model}: quality manifest/"
                "NPZ step-count mismatch"
            )

        if (
            num_steps
            !=
            int(
                metadata[
                    "num_steps"
                ]
            )
        ):
            raise DemoReplayError(
                f"{model}: raw metadata/NPZ "
                "step-count mismatch"
            )

        if (
            metadata[
                "model"
            ]
            !=
            model
        ):
            raise DemoReplayError(
                f"{model}: raw metadata "
                "model mismatch"
            )

        if (
            metadata[
                "split"
            ]
            !=
            "train"
        ):
            raise DemoReplayError(
                f"{model}: raw demonstration "
                "is not split=train"
            )

        if (
            metadata[
                "quality_class"
            ]
            !=
            "UNCLASSIFIED"
        ):
            raise DemoReplayError(
                f"{model}: raw Demonstration V1 "
                "metadata was unexpectedly "
                "modified; quality must remain "
                "external"
            )

        raw_outcome = (
            metadata[
                "finalization_outcome"
            ][
                "outcome"
            ]
        )

        if (
            raw_outcome
            !=
            row[
                "demo_outcome"
            ]
        ):
            raise DemoReplayError(
                f"{model}: quality manifest/"
                "raw demonstration outcome "
                "mismatch"
            )

        episodes.append(
            (
                row,
                expected_npz_path,
                metadata,
                data,
            )
        )

        total_steps += (
            num_steps
        )

    if total_steps <= 0:
        raise DemoReplayError(
            "Quality-filtered demonstration "
            "set contains no transitions"
        )

    buffer = ReplayBuffer(
        size=
            total_steps,

        random_seed=
            random_seed,
    )

    records = []

    for (
        row,
        npz_path,
        metadata,
        data,
    ) in episodes:
        added = (
            add_episode_to_buffer(
                buffer=
                    buffer,

                data=
                    data,
            )
        )

        records.append(
            {
                "model":
                    row[
                        "model"
                    ],

                "split":
                    row[
                        "split"
                    ],

                "quality_role":
                    row[
                        "quality_role"
                    ],

                "eligibility_field":
                    eligibility_field,

                "outcome":
                    row[
                        "demo_outcome"
                    ],

                "num_steps":
                    added,

                "npz_path":
                    str(
                        npz_path
                    ),
            }
        )

    if (
        len(
            buffer
        )
        !=
        total_steps
    ):
        raise DemoReplayError(
            "Quality-filtered ReplayBuffer "
            "length does not match expected "
            "transition count"
        )

    provenance = {
        "quality_version":
            DEMO_QUALITY_VERSION,

        "quality_manifest":
            str(
                quality_manifest.resolve()
            ),

        "quality_manifest_sha256":
            sha256_file(
                quality_manifest
            ),

        "raw_root":
            str(
                raw_root.resolve()
            ),

        "eligibility_field":
            eligibility_field,

        "episodes":
            len(
                records
            ),

        "transitions":
            len(
                buffer
            ),

        "models":
            [
                record[
                    "model"
                ]
                for record
                in records
            ],

        "random_seed":
            int(
                random_seed
            ),
    }

    return (
        buffer,
        records,
        provenance,
    )


def load_main_demo_replay(
    *,
    raw_root: Path,
    quality_manifest: Path,
    random_seed: int = 42,
):
    """
    Formal Stage-I / Stage-II main D_demo.

    Only Demonstration Quality V1 rows with:

        main_demo_replay_eligible == 1

    are admitted.
    """

    return load_quality_filtered_demo_replay(
        raw_root=
            raw_root,

        quality_manifest=
            quality_manifest,

        eligibility_field=
            MAIN_DEMO_ELIGIBILITY_FIELD,

        random_seed=
            random_seed,
    )


def load_auxiliary_demo_replay(
    *,
    raw_root: Path,
    quality_manifest: Path,
    random_seed: int = 42,
):
    """
    Separate replay for RL_AUXILIARY demonstrations.

    It is deliberately NOT part of the formal main D_demo.
    """

    return load_quality_filtered_demo_replay(
        raw_root=
            raw_root,

        quality_manifest=
            quality_manifest,

        eligibility_field=
            AUXILIARY_DEMO_ELIGIBILITY_FIELD,

        random_seed=
            random_seed,
    )
