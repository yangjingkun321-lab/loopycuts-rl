from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
import tianshou


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


INPUT_PROVENANCE_SCHEMA_VERSION = (
    "bc_weight_calibration_input_provenance_v1"
)


class InputProvenanceError(
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
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def _git_head(
    repository: Path,
) -> str:
    repository = Path(
        repository
    )

    return (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(
                    repository
                ),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        )
        .strip()
    )


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


def load_input_provenance(
    path: Path,
):
    path = Path(
        path
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if (
        payload.get(
            "schema_version"
        )
        !=
        INPUT_PROVENANCE_SCHEMA_VERSION
    ):
        raise InputProvenanceError(
            "Input provenance schema mismatch"
        )

    return payload


def compute_formal_demo_aggregate(
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
        rows = list(
            csv.DictReader(
                f
            )
        )

    selected = [
        row
        for row in rows
        if (
            row[
                "main_demo_replay_eligible"
            ]
            ==
            "1"
        )
    ]

    records = []
    transition_total = 0

    for row in selected:
        model = row[
            "model"
        ]

        steps = int(
            row[
                "demo_num_steps"
            ]
        )

        npz = Path(
            row[
                "demo_npz_file"
            ]
        )

        metadata = Path(
            row[
                "demo_metadata_file"
            ]
        )

        if not npz.is_file():
            raise FileNotFoundError(
                npz
            )

        if not metadata.is_file():
            raise FileNotFoundError(
                metadata
            )

        transition_total += (
            steps
        )

        records.append(
            (
                model,
                steps,
                str(
                    npz.resolve()
                ),
                sha256_file(
                    npz
                ),
                str(
                    metadata.resolve()
                ),
                sha256_file(
                    metadata
                ),
            )
        )

    records.sort(
        key=
            lambda item:
                item[
                    0
                ]
    )

    canonical = "\n".join(
        "\t".join(
            map(
                str,
                record,
            )
        )
        for record in records
    ).encode(
        "utf-8"
    )

    aggregate = hashlib.sha256(
        canonical
    ).hexdigest()

    return {
        "episodes":
            len(
                records
            ),

        "transitions":
            transition_total,

        "aggregate_sha256":
            aggregate,
    }


def assert_frozen_input_provenance(
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

    payload = (
        load_input_provenance(
            provenance_path
        )
    )

    manifest_sha256 = (
        sha256_file(
            provenance_path
        )
    )


    # ============================================================
    # Runtime versions and frozen CPU numerical policy.
    # ============================================================

    runtime = payload[
        "runtime"
    ]

    observed_runtime = {
        "python":
            platform.python_version(),

        "numpy":
            np.__version__,

        "torch":
            torch.__version__,

        "tianshou":
            tianshou.__version__,
    }

    for key, observed in (
        observed_runtime.items()
    ):
        expected = runtime[
            key
        ]

        if observed != expected:
            raise InputProvenanceError(
                f"Runtime version mismatch for {key}: "
                f"expected={expected!r}, "
                f"observed={observed!r}"
            )


    # ============================================================
    # Frozen CPU numerical runtime.
    # ============================================================

    thread_policy = runtime.get(
        "thread_policy"
    )

    if not isinstance(
        thread_policy,
        dict,
    ):
        raise InputProvenanceError(
            "Frozen thread policy is missing"
        )

    if (
        thread_policy.get(
            "device"
        )
        !=
        "cpu"
    ):
        raise InputProvenanceError(
            "Frozen calibration device must be cpu"
        )

    observed_num_threads = (
        torch.get_num_threads()
    )

    observed_num_interop_threads = (
        torch.get_num_interop_threads()
    )

    observed_deterministic = (
        torch.are_deterministic_algorithms_enabled()
    )

    if (
        observed_num_threads
        !=
        int(
            thread_policy[
                "torch_num_threads"
            ]
        )
    ):
        raise InputProvenanceError(
            "torch intra-op thread-count mismatch"
        )

    if (
        observed_num_interop_threads
        !=
        int(
            thread_policy[
                "torch_num_interop_threads"
            ]
        )
    ):
        raise InputProvenanceError(
            "torch inter-op thread-count mismatch"
        )

    if (
        observed_deterministic
        is not
        bool(
            thread_policy[
                "torch_deterministic_algorithms"
            ]
        )
    ):
        raise InputProvenanceError(
            "torch deterministic-algorithm policy mismatch"
        )


    # ============================================================
    # Repeatability audit artifact itself is also pinned.
    # ============================================================

    repeatability = runtime.get(
        "cpu_repeatability_audit"
    )

    if not isinstance(
        repeatability,
        dict,
    ):
        raise InputProvenanceError(
            "CPU repeatability audit metadata is missing"
        )

    repeatability_path = (
        _resolve_project_path(
            repeatability[
                "artifact_path"
            ]
        )
    )

    observed_repeatability_sha = (
        sha256_file(
            repeatability_path
        )
    )

    if (
        observed_repeatability_sha
        !=
        repeatability[
            "artifact_sha256"
        ]
    ):
        raise InputProvenanceError(
            "CPU repeatability audit SHA256 mismatch"
        )

    repeatability_payload = json.loads(
        repeatability_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        repeatability_payload.get(
            "audit_version"
        )
        !=
        repeatability[
            "audit_version"
        ]
    ):
        raise InputProvenanceError(
            "CPU repeatability audit version mismatch"
        )

    if (
        int(
            repeatability_payload[
                "seed"
            ]
        )
        !=
        int(
            repeatability[
                "seed"
            ]
        )
    ):
        raise InputProvenanceError(
            "CPU repeatability audit seed mismatch"
        )

    if (
        float(
            repeatability_payload[
                "bc_weight"
            ]
        )
        !=
        float(
            repeatability[
                "bc_weight"
            ]
        )
    ):
        raise InputProvenanceError(
            "CPU repeatability audit lambda mismatch"
        )

    if (
        int(
            repeatability_payload[
                "updates"
            ]
        )
        !=
        int(
            repeatability[
                "updates"
            ]
        )
    ):
        raise InputProvenanceError(
            "CPU repeatability audit update-count mismatch"
        )

    if (
        int(
            repeatability_payload[
                "torch_num_threads"
            ]
        )
        !=
        observed_num_threads
    ):
        raise InputProvenanceError(
            "CPU repeatability audit intra-op mismatch"
        )

    if (
        int(
            repeatability_payload[
                "torch_num_interop_threads"
            ]
        )
        !=
        observed_num_interop_threads
    ):
        raise InputProvenanceError(
            "CPU repeatability audit inter-op mismatch"
        )

    if (
        bool(
            repeatability_payload[
                "torch_deterministic_algorithms"
            ]
        )
        is not
        observed_deterministic
    ):
        raise InputProvenanceError(
            "CPU repeatability audit deterministic-policy mismatch"
        )


    # ============================================================
    # LoopyCuts source/base provenance.
    #
    # IMPORTANT:
    # The LoopyCuts worktree is intentionally allowed to be dirty.
    # The exact executable SHA256 is the authoritative geometry
    # runtime identity.
    # ============================================================

    loopycuts = payload[
        "loopycuts"
    ]

    source_root = Path(
        loopycuts[
            "source_root"
        ]
    ).resolve()

    if not source_root.is_dir():
        raise NotADirectoryError(
            source_root
        )

    observed_source_commit = (
        _git_head(
            source_root
        )
    )

    expected_source_commit = (
        loopycuts[
            "source_base_commit"
        ]
    )

    if (
        observed_source_commit
        !=
        expected_source_commit
    ):
        raise InputProvenanceError(
            "LoopyCuts source base commit mismatch: "
            f"expected={expected_source_commit}, "
            f"observed={observed_source_commit}"
        )


    for (
        relative_path,
        expected_commit,
    ) in (
        loopycuts[
            "submodules"
        ]
        .items()
    ):
        submodule_root = (
            source_root
            /
            relative_path
        )

        observed_commit = (
            _git_head(
                submodule_root
            )
        )

        if (
            observed_commit
            !=
            expected_commit
        ):
            raise InputProvenanceError(
                "LoopyCuts submodule commit mismatch for "
                f"{relative_path}: "
                f"expected={expected_commit}, "
                f"observed={observed_commit}"
            )


    expected_executable = Path(
        loopycuts[
            "executable"
        ][
            "path"
        ]
    ).resolve()

    executable = Path(
        executable
    ).resolve()

    if (
        executable
        !=
        expected_executable
    ):
        raise InputProvenanceError(
            "Formal executable path mismatch: "
            f"expected={expected_executable}, "
            f"observed={executable}"
        )

    observed_executable_sha = (
        sha256_file(
            executable
        )
    )

    expected_executable_sha = (
        loopycuts[
            "executable"
        ][
            "sha256"
        ]
    )

    if (
        observed_executable_sha
        !=
        expected_executable_sha
    ):
        raise InputProvenanceError(
            "volumetric_cutter SHA256 mismatch"
        )


    # ============================================================
    # Frozen RL-side manifests.
    # ============================================================

    expected_dataset_manifest = (
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
        expected_dataset_manifest
    ):
        raise InputProvenanceError(
            "dataset manifest path mismatch"
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
        raise InputProvenanceError(
            "dataset manifest SHA256 mismatch"
        )


    expected_demo_quality = (
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
        expected_demo_quality
    ):
        raise InputProvenanceError(
            "demo quality manifest path mismatch"
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
        raise InputProvenanceError(
            "demo quality manifest SHA256 mismatch"
        )


    # ============================================================
    # Engineering5 model files.
    # ============================================================

    with dataset_manifest.open(
        newline="",
        encoding="utf-8",
    ) as f:
        dataset_rows = list(
            csv.DictReader(
                f
            )
        )

    engineering = payload[
        "engineering_calibration"
    ]

    expected_models = tuple(
        engineering[
            "models"
        ]
    )

    for model in expected_models:
        rows = [
            row
            for row in
            dataset_rows
            if (
                row[
                    "model"
                ]
                ==
                model
            )
        ]

        if len(
            rows
        ) != 1:
            raise InputProvenanceError(
                f"{model}: expected exactly one dataset row"
            )

        row = rows[
            0
        ]

        if (
            row[
                "split"
            ]
            !=
            "engineering_calibration"
        ):
            raise InputProvenanceError(
                f"{model}: wrong calibration split"
            )

        expected_input = (
            engineering[
                "inputs"
            ][
                model
            ]
        )

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

        expected_mesh = Path(
            expected_input[
                "mesh_path"
            ]
        ).resolve()

        expected_loop = Path(
            expected_input[
                "loop_path"
            ]
        ).resolve()

        if mesh != expected_mesh:
            raise InputProvenanceError(
                f"{model}: mesh path mismatch"
            )

        if loop != expected_loop:
            raise InputProvenanceError(
                f"{model}: loop path mismatch"
            )

        if (
            mesh.stat().st_size
            !=
            int(
                expected_input[
                    "mesh_bytes"
                ]
            )
        ):
            raise InputProvenanceError(
                f"{model}: mesh size mismatch"
            )

        if (
            loop.stat().st_size
            !=
            int(
                expected_input[
                    "loop_bytes"
                ]
            )
        ):
            raise InputProvenanceError(
                f"{model}: loop size mismatch"
            )

        if (
            sha256_file(
                mesh
            )
            !=
            expected_input[
                "mesh_sha256"
            ]
        ):
            raise InputProvenanceError(
                f"{model}: mesh SHA256 mismatch"
            )

        if (
            sha256_file(
                loop
            )
            !=
            expected_input[
                "loop_sha256"
            ]
        ):
            raise InputProvenanceError(
                f"{model}: loop SHA256 mismatch"
            )


    # ============================================================
    # Formal D_demo aggregate fingerprint.
    # ============================================================

    formal_demo = payload[
        "formal_d_demo"
    ]

    expected_raw_root = Path(
        formal_demo[
            "raw_root"
        ]
    ).resolve()

    raw_demo_root = Path(
        raw_demo_root
    ).resolve()

    if (
        raw_demo_root
        !=
        expected_raw_root
    ):
        raise InputProvenanceError(
            "formal D_demo raw-root mismatch"
        )

    demo_observed = (
        compute_formal_demo_aggregate(
            quality_manifest=
                demo_quality_manifest,
        )
    )

    if (
        demo_observed[
            "episodes"
        ]
        !=
        int(
            formal_demo[
                "episodes"
            ]
        )
    ):
        raise InputProvenanceError(
            "formal D_demo episode-count mismatch"
        )

    if (
        demo_observed[
            "transitions"
        ]
        !=
        int(
            formal_demo[
                "transitions"
            ]
        )
    ):
        raise InputProvenanceError(
            "formal D_demo transition-count mismatch"
        )

    if (
        demo_observed[
            "aggregate_sha256"
        ]
        !=
        formal_demo[
            "aggregate_sha256"
        ]
    ):
        raise InputProvenanceError(
            "formal D_demo aggregate SHA256 mismatch"
        )


    return {
        "schema_version":
            INPUT_PROVENANCE_SCHEMA_VERSION,

        "manifest_path":
            str(
                provenance_path.resolve()
            ),

        "manifest_sha256":
            manifest_sha256,

        "loopycuts_source_base_commit":
            observed_source_commit,

        "loopycuts_executable_sha256":
            observed_executable_sha,

        "dataset_manifest_sha256":
            payload[
                "dataset_manifest"
            ][
                "sha256"
            ],

        "demo_quality_manifest_sha256":
            payload[
                "demo_quality_manifest"
            ][
                "sha256"
            ],

        "formal_d_demo_aggregate_sha256":
            demo_observed[
                "aggregate_sha256"
            ],

        "torch_num_threads":
            observed_num_threads,

        "torch_num_interop_threads":
            observed_num_interop_threads,

        "torch_deterministic_algorithms":
            observed_deterministic,

        "cpu_repeatability_audit_sha256":
            observed_repeatability_sha,
    }
