from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


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


from training.protocol_v1 import (
    FORMAL_TRAINING_BLOCKERS,
    PROJECT_BC_WEIGHT,
    PROJECT_BC_WEIGHT_BASIS,
    PROJECT_BC_WEIGHT_SELECTION_AUDIT_PATH,
    PROJECT_BC_WEIGHT_SELECTION_SHA256,
    PROJECT_BC_WEIGHT_SELECTION_SOURCE_GIT_COMMIT,
    PROJECT_BC_WEIGHT_SELECTION_PAIR_ARTIFACTS,
    PROJECT_BC_WEIGHT_SELECTION_EPISODE_ROWS,
    PROJECT_BC_WEIGHT_SELECTION_GRID_SHA256,
    PROJECT_FORMAL_TRAINING_SEEDS,
    PROJECT_FORMAL_TRAINING_DEVICE,
    PROJECT_FORMAL_TRAINING_TORCH_NUM_THREADS,
    PROJECT_FORMAL_TRAINING_TORCH_NUM_INTEROP_THREADS,
    PROJECT_FORMAL_TRAINING_TORCH_DETERMINISTIC_ALGORITHMS,
    PROJECT_STAGE2_EXPLORATION_EPSILON,
    PROJECT_STAGE2_SAMPLES_PER_BUFFER,
    PROJECT_STAGE2_EXPO_REPLAY_CAPACITY,
    PROJECT_STAGE2_MODEL_SPLIT,
    PROJECT_STAGE2_MODEL_COUNT,
    PROJECT_STAGE2_MODEL_SAMPLING,
    PROJECT_STAGE2_MODEL_SAMPLING_RNG,
    PROJECT_STAGE2_CURRICULUM_VERSION,
    PROJECT_STAGE2_CURRICULUM_WARMUP_ENV_STEPS,
    PROJECT_STAGE2_CURRICULUM_WARMUP_MAX_STRATUM,
    PROJECT_STAGE2_CURRICULUM_WARMUP_MODEL_COUNT,
    PROJECT_STAGE2_CURRICULUM_FULL_MODEL_COUNT,
    PROJECT_STAGE2_CURRICULUM_WARMUP_POOL,
    PROJECT_STAGE2_CURRICULUM_FULL_POOL,
    PROJECT_STAGE2_CURRICULUM_PHASE_SELECTION,
    PROJECT_STAGE2_CURRICULUM_BOUNDARY_POLICY,
    PROJECT_STAGE2_CURRICULUM_SAMPLING_WITHIN_POOL,
    PROJECT_STAGE2_EPISODE_MODEL_SEMANTICS,
    PROJECT_STAGE2_DEV_ALLOWED,
    PROJECT_STAGE2_BLIND_ALLOWED,
    PROJECT_STAGE2_UPDATE_SCHEDULING,
    PROJECT_STAGE2_UPDATES_PER_COMPLETED_EPISODE,
    PROJECT_STAGE2_BUDGET_BOUNDARY_POLICY,
    assert_formal_training_ready,
    formal_training_ready,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def main():
    assert math.isclose(
        PROJECT_BC_WEIGHT,
        3.0,
    )

    assert (
        PROJECT_BC_WEIGHT_BASIS
        ==
        "FORMAL_BC_WEIGHT_CALIBRATION_V2_SELECTION"
    )

    audit_path = (
        PROJECT_ROOT
        /
        PROJECT_BC_WEIGHT_SELECTION_AUDIT_PATH
    )

    assert audit_path.is_file()

    observed_sha = (
        sha256_file(
            audit_path
        )
    )

    assert (
        observed_sha
        ==
        PROJECT_BC_WEIGHT_SELECTION_SHA256
        ==
        (
            "50e7814b75fdf1add71dcd101f9d256b1"
            "eadd0406d305bed82da0824c6d79611"
        )
    )

    payload = json.loads(
        audit_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        payload["git_commit"]
        ==
        PROJECT_BC_WEIGHT_SELECTION_SOURCE_GIT_COMMIT
        ==
        "8c2ed2887f272d27f19d848dba4cf21e92a9b6d3"
    )

    assert (
        payload["pair_artifact_count"]
        ==
        PROJECT_BC_WEIGHT_SELECTION_PAIR_ARTIFACTS
        ==
        15
    )

    assert (
        payload["episode_result_count"]
        ==
        PROJECT_BC_WEIGHT_SELECTION_EPISODE_ROWS
        ==
        75
    )

    assert math.isclose(
        float(
            payload[
                "winner"
            ][
                "bc_weight"
            ]
        ),
        PROJECT_BC_WEIGHT,
    )

    assert (
        payload[
            "winner"
        ][
            "full_hex_count"
        ]
        ==
        8
    )

    assert (
        payload[
            "winner"
        ][
            "finalization_crash_count"
        ]
        ==
        4
    )

    assert (
        PROJECT_BC_WEIGHT_SELECTION_GRID_SHA256
        ==
        (
            "fa0a94f24ea528fe72c3543847ddf339"
            "766becb06ff06d12ab3f0a320ad2bf22"
        )
    )

    assert (
        PROJECT_FORMAL_TRAINING_SEEDS
        ==
        (
            42,
            43,
            44,
        )
    )

    assert (
        PROJECT_FORMAL_TRAINING_DEVICE
        ==
        "cpu"
    )

    assert (
        PROJECT_FORMAL_TRAINING_TORCH_NUM_THREADS
        ==
        8
    )

    assert (
        PROJECT_FORMAL_TRAINING_TORCH_NUM_INTEROP_THREADS
        ==
        8
    )

    assert (
        PROJECT_FORMAL_TRAINING_TORCH_DETERMINISTIC_ALGORITHMS
        is False
    )

    assert math.isclose(
        PROJECT_STAGE2_EXPLORATION_EPSILON,
        0.05,
    )

    assert (
        PROJECT_STAGE2_SAMPLES_PER_BUFFER
        ==
        32
    )

    assert (
        PROJECT_STAGE2_EXPO_REPLAY_CAPACITY
        ==
        25_000
    )

    assert (
        PROJECT_STAGE2_MODEL_SPLIT
        ==
        "train"
    )

    assert (
        PROJECT_STAGE2_MODEL_COUNT
        ==
        49
    )

    assert (
        PROJECT_STAGE2_MODEL_SAMPLING
        ==
        "COMPLEXITY_CURRICULUM_UNIFORM_IID_PER_EPISODE"
    )

    assert (
        PROJECT_STAGE2_MODEL_SAMPLING_RNG
        ==
        "NUMPY_GENERATOR_SEEDED_BY_FORMAL_RUN_SEED"
    )

    assert (
        PROJECT_STAGE2_CURRICULUM_VERSION
        ==
        "complexity_curriculum_v1"
    )

    assert (
        PROJECT_STAGE2_CURRICULUM_WARMUP_ENV_STEPS
        ==
        5_000
    )

    assert (
        PROJECT_STAGE2_CURRICULUM_WARMUP_MAX_STRATUM
        ==
        7
    )

    assert (
        PROJECT_STAGE2_CURRICULUM_WARMUP_MODEL_COUNT
        ==
        39
    )

    assert (
        PROJECT_STAGE2_CURRICULUM_FULL_MODEL_COUNT
        ==
        49
    )

    assert (
        PROJECT_STAGE2_CURRICULUM_WARMUP_POOL
        ==
        "TRAIN_MODELS_WITH_COMPLEXITY_STRATUM_LE_7"
    )

    assert (
        PROJECT_STAGE2_CURRICULUM_FULL_POOL
        ==
        "ALL_TRAIN49"
    )

    assert (
        PROJECT_STAGE2_CURRICULUM_PHASE_SELECTION
        ==
        "AT_EPISODE_START_FROM_TOTAL_ENVIRONMENT_STEPS"
    )

    assert (
        PROJECT_STAGE2_CURRICULUM_BOUNDARY_POLICY
        ==
        "NO_MID_EPISODE_PHASE_SWITCH"
    )

    assert (
        PROJECT_STAGE2_CURRICULUM_SAMPLING_WITHIN_POOL
        ==
        "UNIFORM_IID_PER_EPISODE"
    )

    assert (
        PROJECT_STAGE2_EPISODE_MODEL_SEMANTICS
        ==
        "ONE_SAMPLED_ELIGIBLE_TRAIN_MODEL_PER_ENVIRONMENT_EPISODE"
    )

    assert (
        PROJECT_STAGE2_DEV_ALLOWED
        is False
    )

    assert (
        PROJECT_STAGE2_BLIND_ALLOWED
        is False
    )

    assert (
        PROJECT_STAGE2_UPDATE_SCHEDULING
        ==
        "AFTER_EACH_COMPLETED_EPISODE"
    )

    assert (
        PROJECT_STAGE2_UPDATES_PER_COMPLETED_EPISODE
        ==
        "EQUAL_TO_NEWLY_COLLECTED_TRANSITIONS_IN_THAT_EPISODE"
    )

    assert (
        PROJECT_STAGE2_BUDGET_BOUNDARY_POLICY
        ==
        "STOP_AT_EXACT_TRANSITION_BUDGET_WITHOUT_SYNTHETIC_TERMINAL_OR_TRUNCATION"
    )

    assert (
        FORMAL_TRAINING_BLOCKERS
        ==
        ()
    )

    assert (
        formal_training_ready()
        is True
    )

    assert_formal_training_ready()

    print(
        "selection SHA256 :",
        observed_sha,
    )

    print(
        "selected lambda  :",
        PROJECT_BC_WEIGHT,
    )

    print(
        "formal seeds     :",
        PROJECT_FORMAL_TRAINING_SEEDS,
    )

    print(
        "formal runtime   :",
        (
            PROJECT_FORMAL_TRAINING_DEVICE,
            PROJECT_FORMAL_TRAINING_TORCH_NUM_THREADS,
            PROJECT_FORMAL_TRAINING_TORCH_NUM_INTEROP_THREADS,
            PROJECT_FORMAL_TRAINING_TORCH_DETERMINISTIC_ALGORITHMS,
        ),
    )

    print(
        "PASS: BC-weight calibration evidence is frozen in-repository"
    )

    print(
        "PASS: current formal-training protocol semantics are frozen"
    )


if __name__ == "__main__":
    main()
