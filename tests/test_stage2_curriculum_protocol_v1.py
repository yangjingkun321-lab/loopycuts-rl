from __future__ import annotations

import csv
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
    PROTOCOL_VERSION,
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
    PROJECT_STAGE2_DEV_ALLOWED,
    PROJECT_STAGE2_BLIND_ALLOWED,
)


DATASET_MANIFEST = (
    PROJECT_ROOT
    /
    "data/manifests/dataset_split_v2.csv"
)


EXPECTED_DEFERRED_MODELS = {
    "bearing_plate",
    "camille_hand",
    "des6",
    "gear",
    "hinge",
    "impeller",
    "mechanical02",
    "mechanical05",
    "mechanical06",
    "motor_tail",
}


def main():
    assert (
        PROTOCOL_VERSION
        ==
        "loopycuts_training_protocol_v2_curriculum"
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
        PROJECT_STAGE2_DEV_ALLOWED
        is False
    )

    assert (
        PROJECT_STAGE2_BLIND_ALLOWED
        is False
    )

    with DATASET_MANIFEST.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    train_rows = [
        row
        for row in rows
        if row["split"] == "train"
    ]

    dev_rows = [
        row
        for row in rows
        if row["split"] == "dev"
    ]

    # Dataset Split V2 stores the sealed Blind10 corpus under
    # split="test".  "Blind10" is the project-level semantic name;
    # "test" is the authoritative manifest value.
    test_rows = [
        row
        for row in rows
        if row["split"] == "test"
    ]

    assert len(train_rows) == 49
    assert len(dev_rows) == 10
    assert len(test_rows) == 10

    warmup_rows = [
        row
        for row in train_rows
        if (
            int(row["complexity_stratum"])
            <=
            PROJECT_STAGE2_CURRICULUM_WARMUP_MAX_STRATUM
        )
    ]

    deferred_rows = [
        row
        for row in train_rows
        if (
            int(row["complexity_stratum"])
            >
            PROJECT_STAGE2_CURRICULUM_WARMUP_MAX_STRATUM
        )
    ]

    assert (
        len(warmup_rows)
        ==
        PROJECT_STAGE2_CURRICULUM_WARMUP_MODEL_COUNT
        ==
        39
    )

    assert (
        len(train_rows)
        ==
        PROJECT_STAGE2_CURRICULUM_FULL_MODEL_COUNT
        ==
        49
    )

    assert len(deferred_rows) == 10

    deferred_models = {
        row["model"]
        for row in deferred_rows
    }

    assert (
        deferred_models
        ==
        EXPECTED_DEFERRED_MODELS
    )

    motor_tail = next(
        row
        for row in train_rows
        if row["model"] == "motor_tail"
    )

    assert (
        int(
            motor_tail[
                "complexity_stratum"
            ]
        )
        ==
        9
    )

    assert (
        int(
            motor_tail[
                "header_loops"
            ]
        )
        ==
        331
    )

    assert (
        int(
            motor_tail[
                "actionable_nonconvex"
            ]
        )
        ==
        159
    )

    assert (
        "motor_tail"
        not in
        {
            row["model"]
            for row in warmup_rows
        }
    )

    assert (
        "motor_tail"
        in
        {
            row["model"]
            for row in train_rows
        }
    )

    warmup_models = {
        row["model"]
        for row in warmup_rows
    }

    assert not (
        warmup_models
        &
        {
            row["model"]
            for row in dev_rows
        }
    )

    assert not (
        warmup_models
        &
        {
            row["model"]
            for row in test_rows
        }
    )

    print(
        "protocol version       :",
        PROTOCOL_VERSION,
    )

    print(
        "warmup env boundary    :",
        PROJECT_STAGE2_CURRICULUM_WARMUP_ENV_STEPS,
    )

    print(
        "warmup max stratum     :",
        PROJECT_STAGE2_CURRICULUM_WARMUP_MAX_STRATUM,
    )

    print(
        "warmup model count     :",
        len(warmup_rows),
    )

    print(
        "deferred model count   :",
        len(deferred_rows),
    )

    print(
        "deferred models        :",
        sorted(
            deferred_models
        ),
    )

    print(
        "PASS: curriculum warmup pool is exactly Train39 strata 0-7"
    )

    print(
        "PASS: full phase remains all Train49"
    )

    print(
        "PASS: motor_tail is deferred from warmup but remains in full Train49"
    )

    print(
        "PASS: Dev10 and Blind10 remain excluded"
    )


if __name__ == "__main__":
    main()
