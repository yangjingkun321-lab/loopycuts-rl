from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


from training.bc_weight_input_provenance_v1 import (
    assert_frozen_input_provenance,
)

from training.protocol_v1 import (
    PROJECT_BC_WEIGHT,
    PROJECT_BC_WEIGHT_SELECTION_SHA256,
    PROJECT_FORMAL_TRAINING_DEVICE,
    PROJECT_FORMAL_TRAINING_TORCH_NUM_THREADS,
    PROJECT_FORMAL_TRAINING_TORCH_NUM_INTEROP_THREADS,
    PROJECT_FORMAL_TRAINING_TORCH_DETERMINISTIC_ALGORITHMS,
    PROJECT_STAGE2_MODEL_COUNT,
    PROJECT_STAGE2_MODEL_SPLIT,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


FORMAL_TRAINING_INPUT_PROVENANCE_VERSION = (
    "formal_training_input_provenance_v1"
)


class FormalTrainingInputProvenanceError(
    RuntimeError
):
    pass


def sha256_file(
    path: Path,
) -> str:
    path = Path(
        path
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
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


def _resolve_project_path(
    value: str,
) -> Path:
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


def _git_is_ancestor(
    ancestor: str,
) -> bool:
    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            ancestor,
            "HEAD",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )

    return (
        result.returncode
        ==
        0
    )


def compute_train49_inputs(
    *,
    dataset_manifest: Path,
):
    dataset_manifest = Path(
        dataset_manifest
    )

    with dataset_manifest.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    train_rows = [
        row
        for row in rows
        if (
            row[
                "split"
            ]
            ==
            PROJECT_STAGE2_MODEL_SPLIT
        )
    ]

    if (
        len(
            train_rows
        )
        !=
        PROJECT_STAGE2_MODEL_COUNT
        or
        PROJECT_STAGE2_MODEL_COUNT
        !=
        49
    ):
        raise FormalTrainingInputProvenanceError(
            "Formal Stage-II Train model count mismatch"
        )

    records = []

    for row in train_rows:
        model = row[
            "model"
        ]

        mesh = Path(
            row[
                "mesh_file"
            ]
        ).resolve()

        loop = Path(
            row[
                "loop_file"
            ]
        ).resolve()

        if not mesh.is_file():
            raise FileNotFoundError(
                mesh
            )

        if not loop.is_file():
            raise FileNotFoundError(
                loop
            )

        records.append(
            {
                "model":
                    model,

                "mesh_path":
                    str(mesh),

                "mesh_bytes":
                    int(
                        mesh.stat().st_size
                    ),

                "mesh_sha256":
                    sha256_file(mesh),

                "loop_path":
                    str(loop),

                "loop_bytes":
                    int(
                        loop.stat().st_size
                    ),

                "loop_sha256":
                    sha256_file(loop),

                "header_loops":
                    int(
                        row[
                            "header_loops"
                        ]
                    ),

                "actionable_nonconvex":
                    int(
                        row[
                            "actionable_nonconvex"
                        ]
                    ),
            }
        )

    records.sort(
        key=lambda x:
            x["model"]
    )

    canonical = "\n".join(
        "\t".join(
            [
                x["model"],
                x["mesh_path"],
                str(
                    x[
                        "mesh_bytes"
                    ]
                ),
                x["mesh_sha256"],
                x["loop_path"],
                str(
                    x[
                        "loop_bytes"
                    ]
                ),
                x["loop_sha256"],
                str(
                    x[
                        "header_loops"
                    ]
                ),
                str(
                    x[
                        "actionable_nonconvex"
                    ]
                ),
            ]
        )
        for x in records
    )

    aggregate = hashlib.sha256(
        canonical.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        records,
        aggregate,
    )


def assert_formal_training_input_provenance(
    *,
    provenance_path: Path,
    executable: Path,
    dataset_manifest: Path,
    demo_quality_manifest: Path,
    raw_demo_root: Path,
):
    provenance_path = Path(
        provenance_path
    )

    payload = json.loads(
        provenance_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        payload.get(
            "schema_version"
        )
        !=
        FORMAL_TRAINING_INPUT_PROVENANCE_VERSION
    ):
        raise FormalTrainingInputProvenanceError(
            "Formal-training provenance schema mismatch"
        )


    # ------------------------------------------------------------
    # The frozen protocol commit must remain in current history.
    # ------------------------------------------------------------

    protocol_commit = payload[
        "protocol_freeze_commit"
    ]

    if not _git_is_ancestor(
        protocol_commit
    ):
        raise FormalTrainingInputProvenanceError(
            "Frozen protocol commit is not an ancestor of HEAD"
        )


    # ------------------------------------------------------------
    # Paths and file hashes.
    # ------------------------------------------------------------

    expected_dataset = (
        _resolve_project_path(
            payload[
                "dataset_manifest"
            ][
                "path"
            ]
        )
    )

    dataset_manifest = Path(
        dataset_manifest
    ).resolve()

    if (
        dataset_manifest
        !=
        expected_dataset
    ):
        raise FormalTrainingInputProvenanceError(
            "Dataset manifest path mismatch"
        )

    if (
        sha256_file(
            dataset_manifest
        )
        !=
        payload[
            "dataset_manifest"
        ][
            "sha256"
        ]
    ):
        raise FormalTrainingInputProvenanceError(
            "Dataset manifest SHA256 mismatch"
        )


    expected_quality = (
        _resolve_project_path(
            payload[
                "demo_quality_manifest"
            ][
                "path"
            ]
        )
    )

    demo_quality_manifest = Path(
        demo_quality_manifest
    ).resolve()

    if (
        demo_quality_manifest
        !=
        expected_quality
    ):
        raise FormalTrainingInputProvenanceError(
            "Demo quality manifest path mismatch"
        )

    if (
        sha256_file(
            demo_quality_manifest
        )
        !=
        payload[
            "demo_quality_manifest"
        ][
            "sha256"
        ]
    ):
        raise FormalTrainingInputProvenanceError(
            "Demo quality manifest SHA256 mismatch"
        )


    # ------------------------------------------------------------
    # Reuse the already-audited executable / D_demo / runtime
    # provenance layer.
    # ------------------------------------------------------------

    base_path = (
        _resolve_project_path(
            payload[
                "base_input_provenance"
            ][
                "path"
            ]
        )
    )

    if (
        sha256_file(
            base_path
        )
        !=
        payload[
            "base_input_provenance"
        ][
            "sha256"
        ]
    ):
        raise FormalTrainingInputProvenanceError(
            "Base provenance SHA256 mismatch"
        )

    base_summary = (
        assert_frozen_input_provenance(
            provenance_path=
                base_path,

            executable=
                executable,

            dataset_manifest=
                dataset_manifest,

            demo_quality_manifest=
                demo_quality_manifest,

            raw_demo_root=
                raw_demo_root,
        )
    )


    # ------------------------------------------------------------
    # Selected BC weight evidence.
    # ------------------------------------------------------------

    selection_path = (
        _resolve_project_path(
            payload[
                "bc_weight_selection"
            ][
                "path"
            ]
        )
    )

    selection_sha = (
        sha256_file(
            selection_path
        )
    )

    if (
        selection_sha
        !=
        payload[
            "bc_weight_selection"
        ][
            "sha256"
        ]
    ):
        raise FormalTrainingInputProvenanceError(
            "BC-weight selection artifact SHA256 mismatch"
        )

    if (
        selection_sha
        !=
        PROJECT_BC_WEIGHT_SELECTION_SHA256
    ):
        raise FormalTrainingInputProvenanceError(
            "Protocol/selection SHA256 mismatch"
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
            payload[
                "bc_weight_selection"
            ][
                "selected_bc_weight"
            ]
        )
    ):
        raise FormalTrainingInputProvenanceError(
            "Selected BC weight mismatch"
        )


    # ------------------------------------------------------------
    # Train49 exact inputs.
    # ------------------------------------------------------------

    (
        observed_records,
        observed_aggregate,
    ) = compute_train49_inputs(
        dataset_manifest=
            dataset_manifest
    )

    expected_train = payload[
        "train49"
    ]

    if (
        int(
            expected_train[
                "models"
            ]
        )
        !=
        PROJECT_STAGE2_MODEL_COUNT
        or
        PROJECT_STAGE2_MODEL_COUNT
        !=
        49
    ):
        raise FormalTrainingInputProvenanceError(
            "Frozen Train49 model count mismatch"
        )

    if (
        observed_records
        !=
        expected_train[
            "inputs"
        ]
    ):
        raise FormalTrainingInputProvenanceError(
            "Train49 per-file provenance mismatch"
        )

    if (
        observed_aggregate
        !=
        expected_train[
            "aggregate_sha256"
        ]
    ):
        raise FormalTrainingInputProvenanceError(
            "Train49 aggregate SHA256 mismatch"
        )


    return {
        "schema_version":
            FORMAL_TRAINING_INPUT_PROVENANCE_VERSION,

        "manifest_path":
            str(
                provenance_path.resolve()
            ),

        "manifest_sha256":
            sha256_file(
                provenance_path
            ),

        "protocol_freeze_commit":
            protocol_commit,

        "train49_models":
            len(
                observed_records
            ),

        "train49_aggregate_sha256":
            observed_aggregate,

        "selected_bc_weight":
            selected_weight,

        "bc_weight_selection_sha256":
            selection_sha,

        "device":
            PROJECT_FORMAL_TRAINING_DEVICE,

        "torch_num_threads":
            PROJECT_FORMAL_TRAINING_TORCH_NUM_THREADS,

        "torch_num_interop_threads":
            PROJECT_FORMAL_TRAINING_TORCH_NUM_INTEROP_THREADS,

        "torch_deterministic_algorithms":
            PROJECT_FORMAL_TRAINING_TORCH_DETERMINISTIC_ALGORITHMS,

        "base_input_provenance":
            base_summary,
    }
