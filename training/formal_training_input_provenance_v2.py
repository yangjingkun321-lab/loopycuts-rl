from __future__ import annotations

import json

from pathlib import Path


from training.formal_training_input_provenance_v1 import (
    compute_train49_inputs,
    sha256_file,
)

from training.protocol_v1 import (
    PROJECT_BC_WEIGHT,
    PROJECT_BC_WEIGHT_SELECTION_SHA256,

    PROJECT_HISTORICAL_BC_INPUT_PROVENANCE_SHA256,
    PROJECT_HISTORICAL_FORMAL_INPUT_PROVENANCE_V1_SHA256,

    PROJECT_QUALITY_REF_MODEL_COUNT,
    PROJECT_QUALITY_REF_SET_VERSION,
    PROJECT_QUALITY_REF_SHA256SUMS_IDENTITY,

    PROJECT_REWARD_VERSION,
    PROJECT_RUNTIME_REWARD_VERSION,

    PROJECT_STAGE2_MODEL_COUNT,
    PROJECT_STAGE2_TRAIN49_INPUT_AGGREGATE_SHA256,
    PROJECT_STAGE2_V5_EXECUTABLE_SHA256,

    PROJECT_TERMINAL_QUALITY_VERSION,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


FORMAL_TRAINING_INPUT_PROVENANCE_V2_VERSION = (
    "formal_training_input_provenance_v2"
)


class FormalTrainingInputProvenanceV2Error(
    RuntimeError
):
    pass


def _resolve_project_path(
    value,
):
    path = Path(
        value
    )

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT
        /
        path
    ).resolve()


def _assert_sha256sums(
    *,
    root: Path,
    sums_file: Path,
):
    root = Path(
        root
    ).resolve()

    sums_file = Path(
        sums_file
    ).resolve()

    if not root.is_dir():
        raise NotADirectoryError(
            root
        )

    if not sums_file.is_file():
        raise FileNotFoundError(
            sums_file
        )

    checked = 0

    for raw_line in sums_file.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if not line:
            continue

        parts = line.split(
            None,
            1,
        )

        if len(parts) != 2:
            raise FormalTrainingInputProvenanceV2Error(
                "Invalid SHA256SUMS line"
            )

        expected_sha = parts[0]

        relative = (
            parts[1]
            .strip()
            .lstrip("*")
        )

        target = (
            root
            /
            relative
        ).resolve()

        if not target.is_relative_to(
            root
        ):
            raise FormalTrainingInputProvenanceV2Error(
                "SHA256SUMS path escapes root"
            )

        if not target.is_file():
            raise FileNotFoundError(
                target
            )

        actual_sha = sha256_file(
            target
        )

        if actual_sha != expected_sha:
            raise FormalTrainingInputProvenanceV2Error(
                "Frozen quality-ref file SHA256 mismatch: "
                f"{relative}"
            )

        checked += 1

    if checked <= 0:
        raise FormalTrainingInputProvenanceV2Error(
            "SHA256SUMS contains no records"
        )

    return checked


def assert_formal_training_input_provenance_v2(
    *,
    historical_provenance_path: Path,
    executable: Path,
    dataset_manifest: Path,
    demo_quality_manifest: Path,
    quality_ref_root: Path,
):
    """
    Validate the V5 formal-training input contract.

    Historical Stage-I / BC / Train49 evidence is inherited by exact
    artifact identity.

    It deliberately does NOT require the current live V4 LoopyCuts
    source tree HEAD to equal its historical calibration commit.

    Current Stage-II runtime is independently pinned to:
        V5 executable SHA256
        Train49 mesh/loop inputs
        Train49 quality-reference set.
    """

    historical_provenance_path = Path(
        historical_provenance_path
    ).resolve()

    executable = Path(
        executable
    ).resolve()

    dataset_manifest = Path(
        dataset_manifest
    ).resolve()

    demo_quality_manifest = Path(
        demo_quality_manifest
    ).resolve()

    quality_ref_root = Path(
        quality_ref_root
    ).resolve()


    # ============================================================
    # Historical Formal V1 artifact identity.
    # ============================================================

    if not historical_provenance_path.is_file():
        raise FileNotFoundError(
            historical_provenance_path
        )

    historical_formal_sha = sha256_file(
        historical_provenance_path
    )

    if (
        historical_formal_sha
        !=
        PROJECT_HISTORICAL_FORMAL_INPUT_PROVENANCE_V1_SHA256
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "Historical Formal V1 provenance SHA256 mismatch"
        )


    payload = json.loads(
        historical_provenance_path.read_text(
            encoding="utf-8"
        )
    )


    if (
        payload.get(
            "schema_version"
        )
        !=
        "formal_training_input_provenance_v1"
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "Historical Formal V1 schema mismatch"
        )


    # ============================================================
    # Historical BC provenance artifact.
    #
    # This proves which historical calibration evidence is inherited,
    # without consulting the current live V4 C++ HEAD.
    # ============================================================

    base_entry = payload[
        "base_input_provenance"
    ]

    if (
        base_entry[
            "sha256"
        ]
        !=
        PROJECT_HISTORICAL_BC_INPUT_PROVENANCE_SHA256
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "Historical BC provenance identity mismatch "
            "inside Formal V1 manifest"
        )


    base_path = _resolve_project_path(
        base_entry[
            "path"
        ]
    )


    if (
        sha256_file(
            base_path
        )
        !=
        PROJECT_HISTORICAL_BC_INPUT_PROVENANCE_SHA256
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "Historical BC provenance file SHA256 mismatch"
        )


    # ============================================================
    # Dataset + demo-quality manifests consumed by formal training.
    # ============================================================

    dataset_entry = payload[
        "dataset_manifest"
    ]

    expected_dataset = _resolve_project_path(
        dataset_entry[
            "path"
        ]
    )


    if (
        dataset_manifest
        !=
        expected_dataset
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "Dataset manifest path mismatch"
        )


    if (
        sha256_file(
            dataset_manifest
        )
        !=
        dataset_entry[
            "sha256"
        ]
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "Dataset manifest SHA256 mismatch"
        )


    demo_entry = payload[
        "demo_quality_manifest"
    ]

    expected_demo_quality = _resolve_project_path(
        demo_entry[
            "path"
        ]
    )


    if (
        demo_quality_manifest
        !=
        expected_demo_quality
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "Demo-quality manifest path mismatch"
        )


    if (
        sha256_file(
            demo_quality_manifest
        )
        !=
        demo_entry[
            "sha256"
        ]
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "Demo-quality manifest SHA256 mismatch"
        )


    # ============================================================
    # Frozen BC selection.
    # ============================================================

    selection_entry = payload[
        "bc_weight_selection"
    ]

    if (
        selection_entry[
            "sha256"
        ]
        !=
        PROJECT_BC_WEIGHT_SELECTION_SHA256
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "Historical BC selection identity mismatch"
        )


    selection_path = _resolve_project_path(
        selection_entry[
            "path"
        ]
    )


    selection_sha = sha256_file(
        selection_path
    )

    if (
        selection_sha
        !=
        PROJECT_BC_WEIGHT_SELECTION_SHA256
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "BC selection artifact SHA256 mismatch"
        )


    selection = json.loads(
        selection_path.read_text(
            encoding="utf-8"
        )
    )


    selected_weight = float(
        selection[
            "winner"
        ][
            "bc_weight"
        ]
    )


    if (
        selected_weight
        !=
        float(
            PROJECT_BC_WEIGHT
        )
        or
        selected_weight
        !=
        float(
            selection_entry[
                "selected_bc_weight"
            ]
        )
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "Selected BC weight mismatch"
        )


    # ============================================================
    # Train49 Stage-II mesh / loop inputs.
    # ============================================================

    (
        observed_train49,
        observed_train49_aggregate,
    ) = compute_train49_inputs(
        dataset_manifest=
            dataset_manifest
    )


    historical_train49 = payload[
        "train49"
    ]


    if (
        len(
            observed_train49
        )
        !=
        PROJECT_STAGE2_MODEL_COUNT
        or
        PROJECT_STAGE2_MODEL_COUNT
        !=
        49
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "Train49 model count mismatch"
        )


    if (
        observed_train49
        !=
        historical_train49[
            "inputs"
        ]
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "Train49 per-file provenance mismatch"
        )


    if (
        observed_train49_aggregate
        !=
        historical_train49[
            "aggregate_sha256"
        ]
        or
        observed_train49_aggregate
        !=
        PROJECT_STAGE2_TRAIN49_INPUT_AGGREGATE_SHA256
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "Train49 aggregate SHA256 mismatch"
        )


    # ============================================================
    # Current V5 Stage-II executable.
    #
    # Executable SHA256 is the authoritative geometry runtime
    # identity.
    # ============================================================

    if not executable.is_file():
        raise FileNotFoundError(
            executable
        )


    executable_sha = sha256_file(
        executable
    )


    if (
        executable_sha
        !=
        PROJECT_STAGE2_V5_EXECUTABLE_SHA256
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "V5 volumetric_cutter SHA256 mismatch"
        )


    # ============================================================
    # Frozen Train49 quality references.
    #
    # quality_ref_root is the existing:
    #
    #     .../quality_refs_train49_v1/refs
    #
    # SHA256SUMS.txt lives one directory above it.
    # ============================================================

    if not quality_ref_root.is_dir():
        raise NotADirectoryError(
            quality_ref_root
        )


    quality_set_root = (
        quality_ref_root
        .parent
        .resolve()
    )


    quality_sums = (
        quality_set_root
        /
        "SHA256SUMS.txt"
    )


    quality_sums_sha = sha256_file(
        quality_sums
    )


    if (
        quality_sums_sha
        !=
        PROJECT_QUALITY_REF_SHA256SUMS_IDENTITY
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "Train49 quality-reference SHA256SUMS identity mismatch"
        )


    quality_checksum_records = (
        _assert_sha256sums(
            root=
                quality_set_root,

            sums_file=
                quality_sums,
        )
    )


    ref_files = sorted(
        quality_ref_root.glob(
            "*.quality_ref_v1"
        )
    )


    if (
        len(
            ref_files
        )
        !=
        PROJECT_QUALITY_REF_MODEL_COUNT
        or
        PROJECT_QUALITY_REF_MODEL_COUNT
        !=
        49
    ):
        raise FormalTrainingInputProvenanceV2Error(
            "Train49 quality-reference model count mismatch"
        )


    expected_ref_names = {
        (
            record[
                "model"
            ]
            +
            ".quality_ref_v1"
        )
        for record in observed_train49
    }


    actual_ref_names = {
        path.name
        for path in ref_files
    }


    if (
        actual_ref_names
        !=
        expected_ref_names
    ):
        missing = sorted(
            expected_ref_names
            -
            actual_ref_names
        )

        extra = sorted(
            actual_ref_names
            -
            expected_ref_names
        )

        raise FormalTrainingInputProvenanceV2Error(
            "Train49 quality-reference/model binding mismatch: "
            f"missing={missing}, extra={extra}"
        )


    # ============================================================
    # V5 summary for checkpoint/run-artifact provenance.
    # ============================================================

    return {
        "schema_version":
            FORMAL_TRAINING_INPUT_PROVENANCE_V2_VERSION,

        "historical_formal_v1_manifest_path":
            str(
                historical_provenance_path
            ),

        "historical_formal_v1_manifest_sha256":
            historical_formal_sha,

        "historical_bc_input_provenance_sha256":
            PROJECT_HISTORICAL_BC_INPUT_PROVENANCE_SHA256,

        "historical_protocol_freeze_commit":
            payload[
                "protocol_freeze_commit"
            ],

        "selected_bc_weight":
            selected_weight,

        "bc_weight_selection_sha256":
            selection_sha,

        "train49_models":
            len(
                observed_train49
            ),

        "train49_aggregate_sha256":
            observed_train49_aggregate,

        "stage2_executable_path":
            str(
                executable
            ),

        "stage2_executable_sha256":
            executable_sha,

        "reward_version":
            PROJECT_REWARD_VERSION,

        "runtime_reward_version":
            PROJECT_RUNTIME_REWARD_VERSION,

        "terminal_quality_version":
            PROJECT_TERMINAL_QUALITY_VERSION,

        "quality_ref_set_version":
            PROJECT_QUALITY_REF_SET_VERSION,

        "quality_ref_models":
            len(
                ref_files
            ),

        "quality_ref_sha256sums_path":
            str(
                quality_sums
            ),

        "quality_ref_sha256sums_identity":
            quality_sums_sha,

        "quality_ref_checksum_records":
            quality_checksum_records,
    }
