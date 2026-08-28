from __future__ import annotations

import shutil
import tempfile
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


from training.formal_training_input_provenance_v2 import (
    FORMAL_TRAINING_INPUT_PROVENANCE_V2_VERSION,
    FormalTrainingInputProvenanceV2Error,
    assert_formal_training_input_provenance_v2,
)

from training.formal_training_v1 import (
    DEFAULT_DATASET_MANIFEST,
    DEFAULT_DEMO_QUALITY,
    DEFAULT_EXECUTABLE,
    DEFAULT_FORMAL_INPUT_PROVENANCE,
    DEFAULT_QUALITY_REF_ROOT,
)

from training.protocol_v1 import (
    PROJECT_HISTORICAL_BC_INPUT_PROVENANCE_SHA256,
    PROJECT_HISTORICAL_FORMAL_INPUT_PROVENANCE_V1_SHA256,
    PROJECT_QUALITY_REF_SHA256SUMS_IDENTITY,
    PROJECT_STAGE2_TRAIN49_INPUT_AGGREGATE_SHA256,
    PROJECT_STAGE2_V5_EXECUTABLE_SHA256,
)


summary = (
    assert_formal_training_input_provenance_v2(
        historical_provenance_path=
            DEFAULT_FORMAL_INPUT_PROVENANCE,

        executable=
            DEFAULT_EXECUTABLE,

        dataset_manifest=
            DEFAULT_DATASET_MANIFEST,

        demo_quality_manifest=
            DEFAULT_DEMO_QUALITY,

        quality_ref_root=
            DEFAULT_QUALITY_REF_ROOT,
    )
)


assert (
    summary[
        "schema_version"
    ]
    ==
    FORMAL_TRAINING_INPUT_PROVENANCE_V2_VERSION
)

assert (
    summary[
        "historical_formal_v1_manifest_sha256"
    ]
    ==
    PROJECT_HISTORICAL_FORMAL_INPUT_PROVENANCE_V1_SHA256
)

assert (
    summary[
        "historical_bc_input_provenance_sha256"
    ]
    ==
    PROJECT_HISTORICAL_BC_INPUT_PROVENANCE_SHA256
)

assert (
    summary[
        "train49_models"
    ]
    ==
    49
)

assert (
    summary[
        "train49_aggregate_sha256"
    ]
    ==
    PROJECT_STAGE2_TRAIN49_INPUT_AGGREGATE_SHA256
)

assert (
    summary[
        "stage2_executable_sha256"
    ]
    ==
    PROJECT_STAGE2_V5_EXECUTABLE_SHA256
)

assert (
    summary[
        "quality_ref_models"
    ]
    ==
    49
)

assert (
    summary[
        "quality_ref_sha256sums_identity"
    ]
    ==
    PROJECT_QUALITY_REF_SHA256SUMS_IDENTITY
)

assert (
    summary[
        "reward_version"
    ]
    ==
    "reward_v5"
)

assert (
    summary[
        "runtime_reward_version"
    ]
    ==
    "final_v5_quality_aware_v1"
)


print(
    "PASS: V2 provenance inherits exact historical V1/BC artifacts"
)

print(
    "PASS: V2 provenance validates exact Train49 mesh/loop inputs"
)

print(
    "PASS: V2 provenance binds the exact V5 volumetric_cutter"
)

print(
    "PASS: V2 provenance validates the complete Train49 quality-ref set"
)


# ================================================================
# Historical manifest mutation must fail closed.
# ================================================================

with tempfile.TemporaryDirectory() as tmp:
    bad = (
        Path(tmp)
        /
        "formal_training_input_provenance_v1.json"
    )

    shutil.copyfile(
        DEFAULT_FORMAL_INPUT_PROVENANCE,
        bad,
    )

    with bad.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n"
        )

    try:
        assert_formal_training_input_provenance_v2(
            historical_provenance_path=
                bad,

            executable=
                DEFAULT_EXECUTABLE,

            dataset_manifest=
                DEFAULT_DATASET_MANIFEST,

            demo_quality_manifest=
                DEFAULT_DEMO_QUALITY,

            quality_ref_root=
                DEFAULT_QUALITY_REF_ROOT,
        )

    except FormalTrainingInputProvenanceV2Error:
        pass

    else:
        raise AssertionError(
            "mutated historical Formal V1 provenance was accepted"
        )


# ================================================================
# Wrong Stage-II executable must fail closed.
# ================================================================

with tempfile.TemporaryDirectory() as tmp:
    bad_exe = (
        Path(tmp)
        /
        "volumetric_cutter"
    )

    bad_exe.write_bytes(
        b"not the frozen V5 executable"
    )

    try:
        assert_formal_training_input_provenance_v2(
            historical_provenance_path=
                DEFAULT_FORMAL_INPUT_PROVENANCE,

            executable=
                bad_exe,

            dataset_manifest=
                DEFAULT_DATASET_MANIFEST,

            demo_quality_manifest=
                DEFAULT_DEMO_QUALITY,

            quality_ref_root=
                DEFAULT_QUALITY_REF_ROOT,
        )

    except FormalTrainingInputProvenanceV2Error:
        pass

    else:
        raise AssertionError(
            "wrong V5 executable was accepted"
        )


# ================================================================
# Wrong quality-ref set must fail closed.
# ================================================================

with tempfile.TemporaryDirectory() as tmp:
    root = Path(
        tmp
    )

    refs = (
        root
        /
        "refs"
    )

    refs.mkdir()

    (
        root
        /
        "SHA256SUMS.txt"
    ).write_text(
        "",
        encoding="utf-8",
    )

    try:
        assert_formal_training_input_provenance_v2(
            historical_provenance_path=
                DEFAULT_FORMAL_INPUT_PROVENANCE,

            executable=
                DEFAULT_EXECUTABLE,

            dataset_manifest=
                DEFAULT_DATASET_MANIFEST,

            demo_quality_manifest=
                DEFAULT_DEMO_QUALITY,

            quality_ref_root=
                refs,
        )

    except FormalTrainingInputProvenanceV2Error:
        pass

    else:
        raise AssertionError(
            "wrong Train49 quality-ref set was accepted"
        )


print(
    "PASS: historical provenance mutation fails closed"
)

print(
    "PASS: V5 executable mismatch fails closed"
)

print(
    "PASS: quality-ref set mismatch fails closed"
)
