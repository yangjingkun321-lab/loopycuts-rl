from __future__ import annotations

import copy
import hashlib
import json
import os

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path
from typing import Any


import numpy as np
import torch


from training.formal_checkpoint_v1 import (
    FORMAL_CHECKPOINT_VERSION,
    current_git_head,
    repository_is_dirty,
    sha256_file,
)

from training.formal_training_v1 import (
    FORMAL_STAGE2_ONLINE_VERSION,
    FORMAL_TRAINER_CORE_VERSION,

    FormalStage2StateV1,
    FormalTrainingCoreV1,
)

from training.protocol_v1 import (
    PROTOCOL_VERSION,

    PROJECT_BC_WEIGHT,

    PROJECT_FORMAL_TRAINING_SEEDS,

    PROJECT_MAIN_DEMO_EPISODES,
    PROJECT_MAIN_DEMO_TRANSITIONS,

    PROJECT_STAGE1_GRADIENT_STEPS,
    PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS,

    PROJECT_STAGE2_EXPLORATION_EPSILON,
    PROJECT_STAGE2_SAMPLES_PER_BUFFER,
    PROJECT_STAGE2_EXPO_REPLAY_CAPACITY,
    PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS,

    PROJECT_STAGE2_MODEL_COUNT,
    PROJECT_STAGE2_MODEL_SAMPLING,

    PROJECT_STAGE2_CURRICULUM_VERSION,
    PROJECT_STAGE2_CURRICULUM_WARMUP_ENV_STEPS,
    PROJECT_STAGE2_CURRICULUM_WARMUP_MAX_STRATUM,
    PROJECT_STAGE2_CURRICULUM_WARMUP_MODEL_COUNT,
    PROJECT_STAGE2_CURRICULUM_FULL_MODEL_COUNT,
    PROJECT_STAGE2_CURRICULUM_PHASE_SELECTION,
    PROJECT_STAGE2_CURRICULUM_BOUNDARY_POLICY,
)


FORMAL_RUN_ARTIFACTS_VERSION = (
    "loopycuts_formal_run_artifacts_v2_curriculum"
)

FORMAL_RUN_MANIFEST_SCHEMA = (
    "loopycuts_formal_run_manifest_v2_curriculum"
)

FORMAL_RUN_EVENT_SCHEMA = (
    "loopycuts_formal_run_event_v2_curriculum"
)

RUN_MANIFEST_FILENAME = (
    "run_manifest_v2.json"
)

EVENT_LOG_FILENAME = (
    "events_v2.jsonl"
)


class FormalRunArtifactError(
    RuntimeError
):
    pass


# ======================================================================
# JSON canonicalization.
# ======================================================================


def to_jsonable(
    value: Any,
):
    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        torch.Tensor,
    ):
        return (
            value
            .detach()
            .cpu()
            .tolist()
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ):
                to_jsonable(
                    item
                )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            to_jsonable(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ) or value is None:
        return value

    raise FormalRunArtifactError(
        "Value is not JSON serializable under formal artifact policy: "
        f"{type(value)!r}"
    )


def canonical_json_bytes(
    value,
):
    return (
        json.dumps(
            to_jsonable(
                value
            ),
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            ensure_ascii=False,
            allow_nan=False,
        )
        .encode(
            "utf-8"
        )
    )


def sha256_json(
    value,
):
    return (
        hashlib.sha256(
            canonical_json_bytes(
                value
            )
        )
        .hexdigest()
    )


# ======================================================================
# Durable filesystem primitives.
# ======================================================================


def _fsync_directory(
    directory: Path,
):
    directory = Path(
        directory
    )

    fd = os.open(
        str(
            directory
        ),
        os.O_DIRECTORY,
    )

    try:
        os.fsync(
            fd
        )

    finally:
        os.close(
            fd
        )


def atomic_write_json(
    *,
    path: Path,
    payload,
):
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        +
        f".tmp.{os.getpid()}"
    )

    if temporary.exists():
        temporary.unlink()

    encoded = (
        json.dumps(
            to_jsonable(
                payload
            ),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )

    try:
        with temporary.open(
            "wb"
        ) as f:
            f.write(
                encoded
            )

            f.flush()

            os.fsync(
                f.fileno()
            )

        os.replace(
            temporary,
            path,
        )

        _fsync_directory(
            path.parent
        )

    finally:
        if temporary.exists():
            temporary.unlink()


# ======================================================================
# Run manifest.
# ======================================================================


def build_formal_run_manifest(
    core: FormalTrainingCoreV1,
):
    seed = int(
        core.seed
    )

    if (
        seed
        not in
        PROJECT_FORMAL_TRAINING_SEEDS
    ):
        raise FormalRunArtifactError(
            f"Unexpected formal seed: {seed}"
        )

    return {
        "schema_version":
            FORMAL_RUN_MANIFEST_SCHEMA,

        "artifacts_version":
            FORMAL_RUN_ARTIFACTS_VERSION,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "seed":
            seed,

        "repository": {
            "head":
                current_git_head(),

            "dirty":
                repository_is_dirty(),
        },

        "software_contract": {
            "protocol_version":
                PROTOCOL_VERSION,

            "trainer_core_version":
                FORMAL_TRAINER_CORE_VERSION,

            "stage2_online_version":
                FORMAL_STAGE2_ONLINE_VERSION,

            "checkpoint_version":
                FORMAL_CHECKPOINT_VERSION,
        },

        "input_provenance": {
            "manifest_path":
                core.input_provenance[
                    "manifest_path"
                ],

            "manifest_sha256":
                core.input_provenance[
                    "manifest_sha256"
                ],

            "train49_models":
                core.input_provenance[
                    "train49_models"
                ],

            "train49_aggregate_sha256":
                core.input_provenance[
                    "train49_aggregate_sha256"
                ],

            "selected_bc_weight":
                core.input_provenance[
                    "selected_bc_weight"
                ],

            "bc_weight_selection_sha256":
                core.input_provenance[
                    "bc_weight_selection_sha256"
                ],
        },

        "runtime":
            copy.deepcopy(
                core.runtime
            ),

        "formal_training": {
            "lambda_bc":
                float(
                    PROJECT_BC_WEIGHT
                ),

            "demo_episodes":
                int(
                    PROJECT_MAIN_DEMO_EPISODES
                ),

            "demo_transitions":
                int(
                    PROJECT_MAIN_DEMO_TRANSITIONS
                ),

            "stage1_gradient_updates":
                int(
                    PROJECT_STAGE1_GRADIENT_STEPS
                ),

            "stage1_sampled_demo_transitions":
                int(
                    PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS
                ),

            "stage2_model_count":
                int(
                    PROJECT_STAGE2_MODEL_COUNT
                ),

            "stage2_model_sampling":
                PROJECT_STAGE2_MODEL_SAMPLING,

            "stage2_curriculum_version":
                PROJECT_STAGE2_CURRICULUM_VERSION,

            "stage2_curriculum_warmup_env_steps":
                int(
                    PROJECT_STAGE2_CURRICULUM_WARMUP_ENV_STEPS
                ),

            "stage2_curriculum_warmup_max_stratum":
                int(
                    PROJECT_STAGE2_CURRICULUM_WARMUP_MAX_STRATUM
                ),

            "stage2_curriculum_warmup_model_count":
                int(
                    PROJECT_STAGE2_CURRICULUM_WARMUP_MODEL_COUNT
                ),

            "stage2_curriculum_full_model_count":
                int(
                    PROJECT_STAGE2_CURRICULUM_FULL_MODEL_COUNT
                ),

            "stage2_curriculum_phase_selection":
                PROJECT_STAGE2_CURRICULUM_PHASE_SELECTION,

            "stage2_curriculum_boundary_policy":
                PROJECT_STAGE2_CURRICULUM_BOUNDARY_POLICY,

            "stage2_epsilon":
                float(
                    PROJECT_STAGE2_EXPLORATION_EPSILON
                ),

            "stage2_demo_samples_per_update":
                int(
                    PROJECT_STAGE2_SAMPLES_PER_BUFFER
                ),

            "stage2_expo_samples_per_update":
                int(
                    PROJECT_STAGE2_SAMPLES_PER_BUFFER
                ),

            "stage2_expo_capacity":
                int(
                    PROJECT_STAGE2_EXPO_REPLAY_CAPACITY
                ),

            "stage2_environment_budget":
                int(
                    PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
                ),
        },
    }


def create_formal_run_artifacts(
    *,
    run_directory: Path,
    core: FormalTrainingCoreV1,
    require_clean_git: bool = True,
):
    run_directory = Path(
        run_directory
    ).resolve()

    manifest_path = (
        run_directory
        /
        RUN_MANIFEST_FILENAME
    )

    event_log_path = (
        run_directory
        /
        EVENT_LOG_FILENAME
    )

    if (
        require_clean_git
        and
        repository_is_dirty()
    ):
        raise FormalRunArtifactError(
            "Formal run artifacts require a clean Git worktree"
        )

    if (
        manifest_path.exists()
        or
        event_log_path.exists()
    ):
        raise FormalRunArtifactError(
            "Refusing to overwrite an existing formal run artifact set"
        )

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = (
        build_formal_run_manifest(
            core
        )
    )

    atomic_write_json(
        path=
            manifest_path,

        payload=
            manifest,
    )

    with event_log_path.open(
        "xb"
    ) as f:
        f.flush()

        os.fsync(
            f.fileno()
        )

    _fsync_directory(
        run_directory
    )

    return {
        "run_directory":
            str(
                run_directory
            ),

        "manifest_path":
            str(
                manifest_path
            ),

        "manifest_sha256":
            sha256_file(
                manifest_path
            ),

        "event_log_path":
            str(
                event_log_path
            ),
    }


def load_and_validate_run_manifest(
    *,
    run_directory: Path,
    core: FormalTrainingCoreV1,
    strict_git: bool = True,
):
    run_directory = Path(
        run_directory
    ).resolve()

    manifest_path = (
        run_directory
        /
        RUN_MANIFEST_FILENAME
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            manifest_path
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get(
            "schema_version"
        )
        !=
        FORMAL_RUN_MANIFEST_SCHEMA
    ):
        raise FormalRunArtifactError(
            "Formal run manifest schema mismatch"
        )

    if (
        manifest.get(
            "artifacts_version"
        )
        !=
        FORMAL_RUN_ARTIFACTS_VERSION
    ):
        raise FormalRunArtifactError(
            "Formal run artifact version mismatch"
        )

    if (
        int(
            manifest[
                "seed"
            ]
        )
        !=
        int(
            core.seed
        )
    ):
        raise FormalRunArtifactError(
            "Formal run seed mismatch"
        )

    expected_input = manifest[
        "input_provenance"
    ]

    if (
        expected_input[
            "manifest_sha256"
        ]
        !=
        core.input_provenance[
            "manifest_sha256"
        ]
    ):
        raise FormalRunArtifactError(
            "Formal run input-provenance mismatch"
        )

    if (
        expected_input[
            "train49_aggregate_sha256"
        ]
        !=
        core.input_provenance[
            "train49_aggregate_sha256"
        ]
    ):
        raise FormalRunArtifactError(
            "Formal run Train49 provenance mismatch"
        )

    if (
        float(
            expected_input[
                "selected_bc_weight"
            ]
        )
        !=
        float(
            PROJECT_BC_WEIGHT
        )
    ):
        raise FormalRunArtifactError(
            "Formal run lambda_BC mismatch"
        )

    if strict_git:
        if (
            manifest[
                "repository"
            ][
                "dirty"
            ]
        ):
            raise FormalRunArtifactError(
                "Formal run manifest was created from a dirty worktree"
            )

        if (
            manifest[
                "repository"
            ][
                "head"
            ]
            !=
            current_git_head()
        ):
            raise FormalRunArtifactError(
                "Formal run Git HEAD mismatch"
            )

        if repository_is_dirty():
            raise FormalRunArtifactError(
                "Formal run resume requires a clean Git worktree"
            )

    return manifest


# ======================================================================
# Hash-chained JSONL event log.
# ======================================================================


def _event_digest(
    event_without_digest,
):
    return sha256_json(
        event_without_digest
    )


def read_formal_events(
    *,
    event_log_path: Path,
):
    event_log_path = Path(
        event_log_path
    )

    if not event_log_path.is_file():
        raise FileNotFoundError(
            event_log_path
        )

    events = []

    expected_previous = None

    with event_log_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, raw_line in enumerate(
            f,
            start=1,
        ):
            if not raw_line.endswith(
                "\n"
            ):
                raise FormalRunArtifactError(
                    "Formal event log has an incomplete trailing line "
                    f"at line {line_number}"
                )

            line = raw_line.strip()

            if not line:
                raise FormalRunArtifactError(
                    f"Formal event log contains an empty line at {line_number}"
                )

            try:
                event = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise FormalRunArtifactError(
                    f"Invalid formal event JSON at line {line_number}"
                ) from exc

            if (
                event.get(
                    "schema_version"
                )
                !=
                FORMAL_RUN_EVENT_SCHEMA
            ):
                raise FormalRunArtifactError(
                    f"Formal event schema mismatch at line {line_number}"
                )

            if (
                int(
                    event.get(
                        "sequence",
                        -1,
                    )
                )
                !=
                line_number
            ):
                raise FormalRunArtifactError(
                    f"Formal event sequence mismatch at line {line_number}"
                )

            if (
                event.get(
                    "prev_event_sha256"
                )
                !=
                expected_previous
            ):
                raise FormalRunArtifactError(
                    f"Formal event hash-chain predecessor mismatch at line {line_number}"
                )

            stored_digest = event.get(
                "event_sha256"
            )

            if not isinstance(
                stored_digest,
                str,
            ):
                raise FormalRunArtifactError(
                    f"Formal event lacks SHA256 at line {line_number}"
                )

            digest_input = dict(
                event
            )

            digest_input.pop(
                "event_sha256",
                None,
            )

            observed_digest = (
                _event_digest(
                    digest_input
                )
            )

            if (
                stored_digest
                !=
                observed_digest
            ):
                raise FormalRunArtifactError(
                    f"Formal event SHA256 mismatch at line {line_number}"
                )

            events.append(
                event
            )

            expected_previous = (
                stored_digest
            )

    return events


def append_formal_event(
    *,
    event_log_path: Path,
    seed: int,
    event_type: str,
    payload,
):
    event_log_path = Path(
        event_log_path
    )

    events = read_formal_events(
        event_log_path=
            event_log_path
    )

    sequence = (
        len(
            events
        )
        +
        1
    )

    previous_digest = (
        events[
            -1
        ][
            "event_sha256"
        ]
        if events
        else
        None
    )

    event_without_digest = {
        "schema_version":
            FORMAL_RUN_EVENT_SCHEMA,

        "sequence":
            sequence,

        "seed":
            int(
                seed
            ),

        "event_type":
            str(
                event_type
            ),

        "prev_event_sha256":
            previous_digest,

        "payload":
            to_jsonable(
                payload
            ),
    }

    event = dict(
        event_without_digest
    )

    event[
        "event_sha256"
    ] = (
        _event_digest(
            event_without_digest
        )
    )

    encoded = (
        json.dumps(
            event,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            ensure_ascii=False,
            allow_nan=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )

    with event_log_path.open(
        "ab"
    ) as f:
        f.write(
            encoded
        )

        f.flush()

        os.fsync(
            f.fileno()
        )

    return event


# ======================================================================
# Typed events.
# ======================================================================


def record_stage1_complete(
    *,
    run_directory: Path,
    core: FormalTrainingCoreV1,
    stage1_result,
):
    event_log_path = (
        Path(
            run_directory
        )
        /
        EVENT_LOG_FILENAME
    )

    payload = {
        "stage":
            "STAGE_I",

        "gradient_updates":
            int(
                core.stage1_updates_completed
            ),

        "sampled_demo_transitions":
            int(
                core.stage1_sampled_demo_transitions
            ),

        "alpha_after_stage1":
            float(
                stage1_result[
                    "alpha_after_stage1"
                ]
            ),

        "elapsed_seconds":
            float(
                stage1_result[
                    "elapsed_seconds"
                ]
            ),

        "final_training_stats":
            copy.deepcopy(
                stage1_result[
                    "final_training_stats"
                ]
            ),
    }

    existing = [
        event
        for event in
        read_formal_events(
            event_log_path=
                event_log_path
        )
        if (
            event[
                "event_type"
            ]
            ==
            "STAGE1_COMPLETE"
        )
    ]

    if len(
        existing
    ) > 1:
        raise FormalRunArtifactError(
            "Formal event log contains multiple Stage-I completion events"
        )

    if existing:
        existing_payload = copy.deepcopy(
            existing[
                0
            ][
                "payload"
            ]
        )

        replay_payload = to_jsonable(
            payload
        )

        # elapsed_seconds is runtime telemetry, not part of the
        # deterministic training trajectory.  A Stage-I replay after
        # an interruption may legitimately take a different wall-clock
        # duration while producing exactly the same model/optimizer/RNG
        # state.
        existing_payload.pop(
            "elapsed_seconds",
            None,
        )

        replay_payload.pop(
            "elapsed_seconds",
            None,
        )

        if (
            existing_payload
            !=
            replay_payload
        ):
            raise FormalRunArtifactError(
                "Replayed Stage-I completion differs from existing artifact"
            )

        return {
            "event":
                existing[
                    0
                ],

            "replayed":
                True,
        }

    event = append_formal_event(
        event_log_path=
            event_log_path,

        seed=
            core.seed,

        event_type=
            "STAGE1_COMPLETE",

        payload=
            payload,
    )

    return {
        "event":
            event,

        "replayed":
            False,
    }


def record_stage2_episode(
    *,
    run_directory: Path,
    core: FormalTrainingCoreV1,
    record,
):
    event_log_path = (
        Path(
            run_directory
        )
        /
        EVENT_LOG_FILENAME
    )

    episode_index = int(
        record[
            "episode_index"
        ]
    )

    if episode_index <= 0:
        raise FormalRunArtifactError(
            "Stage-II episode index must be positive"
        )

    existing = [
        event
        for event in
        read_formal_events(
            event_log_path=
                event_log_path
        )
        if (
            event[
                "event_type"
            ]
            ==
            "STAGE2_EPISODE"
        )
    ]

    for expected_index, event in enumerate(
        existing,
        start=1,
    ):
        observed_index = int(
            event[
                "payload"
            ][
                "record"
            ][
                "episode_index"
            ]
        )

        if (
            observed_index
            !=
            expected_index
        ):
            raise FormalRunArtifactError(
                "Stage-II event log episode sequence is not contiguous"
            )

    payload = {
        "record":
            copy.deepcopy(
                record
            ),
    }

    if (
        episode_index
        <=
        len(
            existing
        )
    ):
        existing_event = existing[
            episode_index
            -
            1
        ]

        if (
            existing_event[
                "payload"
            ]
            !=
            to_jsonable(
                payload
            )
        ):
            raise FormalRunArtifactError(
                "Replayed Stage-II episode differs from existing formal artifact: "
                f"episode={episode_index}"
            )

        return {
            "event":
                existing_event,

            "replayed":
                True,
        }

    if (
        episode_index
        !=
        len(
            existing
        )
        +
        1
    ):
        raise FormalRunArtifactError(
            "Stage-II formal event log would contain an episode gap"
        )

    event = append_formal_event(
        event_log_path=
            event_log_path,

        seed=
            core.seed,

        event_type=
            "STAGE2_EPISODE",

        payload=
            payload,
    )

    return {
        "event":
            event,

        "replayed":
            False,
    }


def record_checkpoint(
    *,
    run_directory: Path,
    core: FormalTrainingCoreV1,
    stage2_state: FormalStage2StateV1 | None,
    checkpoint_result,
):
    event_log_path = (
        Path(
            run_directory
        )
        /
        EVENT_LOG_FILENAME
    )

    payload = {
        "checkpoint": {
            "path":
                checkpoint_result[
                    "path"
                ],

            "bytes":
                int(
                    checkpoint_result[
                        "bytes"
                    ]
                ),

            "sha256":
                checkpoint_result[
                    "sha256"
                ],

            "stage":
                checkpoint_result[
                    "stage"
                ],

            "seed":
                int(
                    checkpoint_result[
                        "seed"
                    ]
                ),
        },

        "core": {
            "stage":
                core.stage,

            "stage1_updates_completed":
                int(
                    core.stage1_updates_completed
                ),

            "stage1_sampled_demo_transitions":
                int(
                    core.stage1_sampled_demo_transitions
                ),
        },

        "stage2":
            (
                None
                if stage2_state is None
                else
                {
                    "total_environment_steps":
                        int(
                            stage2_state.total_environment_steps
                        ),

                    "total_gradient_updates":
                        int(
                            stage2_state.total_gradient_updates
                        ),

                    "episode_attempts":
                        int(
                            stage2_state.episode_attempts
                        ),

                    "completed_episodes":
                        int(
                            stage2_state.completed_episodes
                        ),

                    "expo_buffer_size":
                        len(
                            stage2_state.expo_buffer
                        ),
                }
            ),
    }

    events = read_formal_events(
        event_log_path=
            event_log_path
    )

    # Avoid writing the exact same checkpoint event twice.
    if (
        events
        and
        events[
            -1
        ][
            "event_type"
        ]
        ==
        "CHECKPOINT"
        and
        events[
            -1
        ][
            "payload"
        ]
        ==
        to_jsonable(
            payload
        )
    ):
        return {
            "event":
                events[
                    -1
                ],

            "replayed":
                True,
        }

    event = append_formal_event(
        event_log_path=
            event_log_path,

        seed=
            core.seed,

        event_type=
            "CHECKPOINT",

        payload=
            payload,
    )

    return {
        "event":
            event,

        "replayed":
            False,
    }


def record_run_complete(
    *,
    run_directory: Path,
    core: FormalTrainingCoreV1,
    stage2_state: FormalStage2StateV1,
    final_checkpoint_result,
):
    event_log_path = (
        Path(
            run_directory
        )
        /
        EVENT_LOG_FILENAME
    )

    payload = {
        "stage":
            core.stage,

        "total_environment_steps":
            int(
                stage2_state.total_environment_steps
            ),

        "total_gradient_updates":
            int(
                stage2_state.total_gradient_updates
            ),

        "episode_attempts":
            int(
                stage2_state.episode_attempts
            ),

        "completed_episodes":
            int(
                stage2_state.completed_episodes
            ),

        "expo_buffer_size":
            len(
                stage2_state.expo_buffer
            ),

        "alpha_final":
            float(
                core.auto_alpha.value
            ),

        "final_checkpoint": {
            "path":
                final_checkpoint_result[
                    "path"
                ],

            "bytes":
                int(
                    final_checkpoint_result[
                        "bytes"
                    ]
                ),

            "sha256":
                final_checkpoint_result[
                    "sha256"
                ],
        },
    }

    existing = [
        event
        for event in
        read_formal_events(
            event_log_path=
                event_log_path
        )
        if (
            event[
                "event_type"
            ]
            ==
            "RUN_COMPLETE"
        )
    ]

    if len(
        existing
    ) > 1:
        raise FormalRunArtifactError(
            "Formal event log contains multiple run-complete events"
        )

    if existing:
        if (
            existing[
                0
            ][
                "payload"
            ]
            !=
            to_jsonable(
                payload
            )
        ):
            raise FormalRunArtifactError(
                "Repeated run-complete artifact differs from existing record"
            )

        return {
            "event":
                existing[
                    0
                ],

            "replayed":
                True,
        }

    event = append_formal_event(
        event_log_path=
            event_log_path,

        seed=
            core.seed,

        event_type=
            "RUN_COMPLETE",

        payload=
            payload,
    )

    return {
        "event":
            event,

        "replayed":
            False,
    }
