from __future__ import annotations

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
