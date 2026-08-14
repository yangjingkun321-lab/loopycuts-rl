from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from observation.builder import (
    GLOBAL_DIM,
    LOOP_FEATURE_DIM,
    MAX_LOOPS,
)


SCHEMA_VERSION = (
    "loopycuts_demo_episode_v1"
)

OBSERVATION_VERSION = (
    "observation_v1"
)

REWARD_VERSION = (
    "reward_v2"
)

TEACHER_VERSION = (
    "original_runtime_policy_v1"
)


class DemoValidationError(
    ValueError
):
    pass


def copy_observation(
    observation,
):
    """
    Make an owned, fixed-dtype copy of frozen Observation V1.

    Returned layout:

        {
            "global": float32[16],
            "loops":  float32[331, 14],
            "exists": bool[331],
            "mask":   bool[331],
        }
    """

    if not isinstance(
        observation,
        dict,
    ):
        raise DemoValidationError(
            "Observation must be a dict"
        )

    if (
        "obs" not in observation
        or
        "mask" not in observation
    ):
        raise DemoValidationError(
            "Observation must contain "
            "'obs' and 'mask'"
        )

    inner = observation[
        "obs"
    ]

    if not isinstance(
        inner,
        dict,
    ):
        raise DemoValidationError(
            "Observation['obs'] must "
            "be a dict"
        )

    required_inner = {
        "global",
        "loops",
        "exists",
    }

    missing = (
        required_inner
        -
        set(
            inner
        )
    )

    if missing:
        raise DemoValidationError(
            "Observation['obs'] is "
            "missing fields: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )

    global_features = (
        np.asarray(
            inner[
                "global"
            ],
            dtype=np.float32,
        )
        .copy()
    )

    loop_features = (
        np.asarray(
            inner[
                "loops"
            ],
            dtype=np.float32,
        )
        .copy()
    )

    exists = (
        np.asarray(
            inner[
                "exists"
            ],
            dtype=np.bool_,
        )
        .copy()
    )

    mask = (
        np.asarray(
            observation[
                "mask"
            ],
            dtype=np.bool_,
        )
        .copy()
    )

    if (
        global_features.shape
        !=
        (
            GLOBAL_DIM,
        )
    ):
        raise DemoValidationError(
            "Invalid global observation "
            f"shape: "
            f"{global_features.shape}; "
            f"expected ({GLOBAL_DIM},)"
        )

    if (
        loop_features.shape
        !=
        (
            MAX_LOOPS,
            LOOP_FEATURE_DIM,
        )
    ):
        raise DemoValidationError(
            "Invalid loop observation "
            f"shape: "
            f"{loop_features.shape}; "
            "expected "
            f"({MAX_LOOPS}, "
            f"{LOOP_FEATURE_DIM})"
        )

    if (
        exists.shape
        !=
        (
            MAX_LOOPS,
        )
    ):
        raise DemoValidationError(
            "Invalid exists shape: "
            f"{exists.shape}"
        )

    if (
        mask.shape
        !=
        (
            MAX_LOOPS,
        )
    ):
        raise DemoValidationError(
            "Invalid action-mask shape: "
            f"{mask.shape}"
        )

    return {
        "global":
            global_features,

        "loops":
            loop_features,

        "exists":
            exists,

        "mask":
            mask,
    }


def sha256_file(
    path: Path,
):
    path = Path(
        path
    )

    digest = (
        hashlib.sha256()
    )

    with path.open(
        "rb"
    ) as f:
        while True:
            block = f.read(
                1024
                *
                1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return (
        digest.hexdigest()
    )


def validate_episode(
    *,
    observations,
    actions,
    rewards,
    terminated,
    truncated,
):
    """
    Demo Episode V1 layout:

        observations: T + 1
        actions:      T
        rewards:      T
        terminated:   T
        truncated:    T

    Transition t:

        observations[t]
        action[t]
        reward[t]
        observations[t + 1]
    """

    num_steps = len(
        actions
    )

    if (
        num_steps
        <=
        0
    ):
        raise DemoValidationError(
            "Demo episode contains "
            "no transitions"
        )

    if (
        len(
            observations
        )
        !=
        num_steps
        +
        1
    ):
        raise DemoValidationError(
            "Observation sequence "
            "must contain T+1 entries"
        )

    sequence_fields = (
        (
            "rewards",
            rewards,
        ),
        (
            "terminated",
            terminated,
        ),
        (
            "truncated",
            truncated,
        ),
    )

    for (
        name,
        values,
    ) in sequence_fields:
        if (
            len(
                values
            )
            !=
            num_steps
        ):
            raise DemoValidationError(
                f"{name} must contain "
                "exactly T entries"
            )

    for step in range(
        num_steps
    ):
        action = int(
            actions[
                step
            ]
        )

        if not (
            0
            <=
            action
            <
            MAX_LOOPS
        ):
            raise DemoValidationError(
                "Invalid action at "
                f"step={step}: "
                f"{action}"
            )

        observation = (
            observations[
                step
            ]
        )

        mask = (
            observation[
                "mask"
            ]
        )

        if not bool(
            mask[
                action
            ]
        ):
            raise DemoValidationError(
                "Teacher selected an "
                "illegal action at "
                f"step={step}: "
                f"action={action}"
            )

    if any(
        bool(
            value
        )
        for value
        in truncated
    ):
        raise DemoValidationError(
            "Demo V1 does not allow "
            "artificial truncation"
        )

    if not bool(
        terminated[
            -1
        ]
    ):
        raise DemoValidationError(
            "Final transition must "
            "be terminal"
        )

    if any(
        bool(
            value
        )
        for value
        in terminated[
            :-1
        ]
    ):
        raise DemoValidationError(
            "Only the final transition "
            "may be terminal"
        )

    final_mask = (
        observations[
            -1
        ][
            "mask"
        ]
    )

    if np.any(
        final_mask
    ):
        raise DemoValidationError(
            "Frozen terminal "
            "observation must have "
            "an all-False action mask"
        )


def save_episode(
    *,
    output_dir: Path,
    model: str,
    split: str,
    mesh_file: Path,
    loop_file: Path,
    source_git_commit: str,
    observations,
    actions,
    rewards,
    terminated,
    truncated,
    audit_records,
    finalization_outcome,
    initial_actionable: int,
):
    validate_episode(
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
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"{model}_"
        "original_demo_v1"
    )

    npz_path = (
        output_dir
        /
        f"{stem}.npz"
    )

    metadata_path = (
        output_dir
        /
        f"{stem}.json"
    )

    audit_path = (
        output_dir
        /
        f"{stem}.jsonl"
    )

    obs_global = (
        np.stack(
            [
                observation[
                    "global"
                ]
                for observation
                in observations
            ],
            axis=0,
        )
        .astype(
            np.float32,
            copy=False,
        )
    )

    obs_loops = (
        np.stack(
            [
                observation[
                    "loops"
                ]
                for observation
                in observations
            ],
            axis=0,
        )
        .astype(
            np.float32,
            copy=False,
        )
    )

    obs_exists = (
        np.stack(
            [
                observation[
                    "exists"
                ]
                for observation
                in observations
            ],
            axis=0,
        )
        .astype(
            np.bool_,
            copy=False,
        )
    )

    obs_mask = (
        np.stack(
            [
                observation[
                    "mask"
                ]
                for observation
                in observations
            ],
            axis=0,
        )
        .astype(
            np.bool_,
            copy=False,
        )
    )

    actions_array = (
        np.asarray(
            actions,
            dtype=np.int64,
        )
    )

    rewards_array = (
        np.asarray(
            rewards,
            dtype=np.float32,
        )
    )

    terminated_array = (
        np.asarray(
            terminated,
            dtype=np.bool_,
        )
    )

    truncated_array = (
        np.asarray(
            truncated,
            dtype=np.bool_,
        )
    )

    temporary_npz = (
        output_dir
        /
        (
            "."
            +
            npz_path.name
            +
            ".tmp.npz"
        )
    )

    np.savez_compressed(
        temporary_npz,

        obs_global=
            obs_global,

        obs_loops=
            obs_loops,

        obs_exists=
            obs_exists,

        obs_mask=
            obs_mask,

        actions=
            actions_array,

        rewards=
            rewards_array,

        terminated=
            terminated_array,

        truncated=
            truncated_array,
    )

    temporary_npz.replace(
        npz_path
    )

    with audit_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for record in audit_records:
            f.write(
                json.dumps(
                    record,
                    sort_keys=True,
                )
            )

            f.write(
                "\n"
            )

    metadata = {
        "schema_version":
            SCHEMA_VERSION,

        "observation_version":
            OBSERVATION_VERSION,

        "reward_version":
            REWARD_VERSION,

        "teacher_version":
            TEACHER_VERSION,

        "teacher_policy":
            (
                "OriginalOrderPolicy: "
                "min(current authoritative "
                "C++ legal loop IDs)"
            ),

        "model":
            str(
                model
            ),

        "split":
            str(
                split
            ),

        "mesh_file":
            str(
                mesh_file
            ),

        "loop_file":
            str(
                loop_file
            ),

        "source_git_commit":
            str(
                source_git_commit
            ),

        "num_steps":
            int(
                len(
                    actions
                )
            ),

        "initial_actionable":
            int(
                initial_actionable
            ),

        "total_return":
            float(
                np.sum(
                    rewards_array,
                    dtype=np.float64,
                )
            ),

        "finalization_outcome":
            finalization_outcome,

        # Filtering is deliberately
        # a later phase.
        "quality_class":
            "UNCLASSIFIED",

        "npz_file":
            npz_path.name,

        "audit_file":
            audit_path.name,

        "npz_sha256":
            sha256_file(
                npz_path
            ),

        "audit_sha256":
            sha256_file(
                audit_path
            ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    return {
        "npz":
            npz_path,

        "metadata":
            metadata_path,

        "audit":
            audit_path,

        "record":
            metadata,
    }


def load_episode(
    npz_path: Path,
):
    """
    Load and structurally validate
    Demo Episode V1 arrays.

    Conversion to Tianshou Batch /
    ReplayBuffer is intentionally
    handled by a later module.
    """

    npz_path = Path(
        npz_path
    )

    with np.load(
        npz_path,
        allow_pickle=False,
    ) as data:
        result = {
            name:
                np.array(
                    data[
                        name
                    ],
                    copy=True,
                )
            for name
            in data.files
        }

    required = {
        "obs_global",
        "obs_loops",
        "obs_exists",
        "obs_mask",
        "actions",
        "rewards",
        "terminated",
        "truncated",
    }

    missing = (
        required
        -
        set(
            result
        )
    )

    if missing:
        raise DemoValidationError(
            "Demo NPZ is missing "
            "arrays: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )

    num_steps = int(
        result[
            "actions"
        ].shape[
            0
        ]
    )

    if (
        result[
            "obs_global"
        ].shape
        !=
        (
            num_steps
            +
            1,
            GLOBAL_DIM,
        )
    ):
        raise DemoValidationError(
            "Invalid obs_global shape: "
            f"{result['obs_global'].shape}"
        )

    if (
        result[
            "obs_loops"
        ].shape
        !=
        (
            num_steps
            +
            1,
            MAX_LOOPS,
            LOOP_FEATURE_DIM,
        )
    ):
        raise DemoValidationError(
            "Invalid obs_loops shape: "
            f"{result['obs_loops'].shape}"
        )

    if (
        result[
            "obs_exists"
        ].shape
        !=
        (
            num_steps
            +
            1,
            MAX_LOOPS,
        )
    ):
        raise DemoValidationError(
            "Invalid obs_exists shape"
        )

    if (
        result[
            "obs_mask"
        ].shape
        !=
        (
            num_steps
            +
            1,
            MAX_LOOPS,
        )
    ):
        raise DemoValidationError(
            "Invalid obs_mask shape"
        )

    one_dimensional = (
        "actions",
        "rewards",
        "terminated",
        "truncated",
    )

    for name in one_dimensional:
        if (
            result[
                name
            ].shape
            !=
            (
                num_steps,
            )
        ):
            raise DemoValidationError(
                f"Invalid {name} shape: "
                f"{result[name].shape}"
            )

    for step in range(
        num_steps
    ):
        action = int(
            result[
                "actions"
            ][
                step
            ]
        )

        if not bool(
            result[
                "obs_mask"
            ][
                step,
                action,
            ]
        ):
            raise DemoValidationError(
                "Stored action is illegal "
                "under stored action mask: "
                f"step={step}, "
                f"action={action}"
            )

    if not bool(
        result[
            "terminated"
        ][
            -1
        ]
    ):
        raise DemoValidationError(
            "Stored episode does not "
            "terminate on its final step"
        )

    if np.any(
        result[
            "obs_mask"
        ][
            -1
        ]
    ):
        raise DemoValidationError(
            "Stored terminal observation "
            "mask is not all False"
        )

    if np.any(
        result[
            "truncated"
        ]
    ):
        raise DemoValidationError(
            "Stored Demo V1 episode "
            "contains truncation"
        )

    return result
