from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys

from pathlib import Path

import numpy as np


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


from quality.quality_ref_v1 import (
    METRIC_CONTRACT_V3_SHA256,
    read_quality_ref_v1,
    sha256_file,
)

from training.formal_training_input_provenance_v1 import (
    compute_train49_inputs,
)


SCHEMA_VERSION = (
    "loopycuts_quality_refs_train49_v1"
)

DEFAULT_DATASET_MANIFEST = (
    PROJECT_ROOT
    /
    "data/manifests/"
    "dataset_split_v2.csv"
)

DEFAULT_FORMAL_PROVENANCE = (
    PROJECT_ROOT
    /
    "data/manifests/"
    "formal_training_input_provenance_v1.json"
)

DEFAULT_POINT_TRIANGLE_BACKEND = Path(
    "/home/yjk/loopycuts_test/"
    "analysis_tools/bin/"
    "point_triangle_octree_v1"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "quality_refs_train49_v1"
)

STAGE2_SUFFIX = (
    "_splitted.obj"
)

QUALITY_REF_CORE_FILES = (
    "quality/quality_ref_v1.py",
    "quality/build_quality_ref_v1.py",
    "quality/validate_quality_ref_v1.py",
)


def git_output(
    *args: str,
) -> str:
    completed = subprocess.run(
        [
            "git",
            *args,
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "git command failed:\n"
            +
            completed.stdout
        )

    return completed.stdout.strip()


def require_clean_git() -> str:
    status = git_output(
        "status",
        "--porcelain",
    )

    if status:
        raise RuntimeError(
            "Refusing Train49 artifact generation "
            "from dirty Git worktree:\n"
            +
            status
        )

    return git_output(
        "rev-parse",
        "HEAD",
    )


def read_sharp_declared_count(
    path: Path,
) -> int:
    lines = [
        raw.strip()
        for raw in path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if (
            raw.strip()
            and
            not raw.lstrip().startswith(
                "#"
            )
        )
    ]

    if not lines:
        raise RuntimeError(
            f"{path}: empty SHARP"
        )

    declared = int(
        lines[0]
    )

    if declared < 0:
        raise RuntimeError(
            f"{path}: negative SHARP count"
        )

    records = lines[1:]

    if declared != len(records):
        raise RuntimeError(
            f"{path}: declared={declared}, "
            f"records={len(records)}"
        )

    return declared


def derive_sharp_sidecars(
    mesh: Path,
):
    mesh = mesh.resolve()

    if not mesh.name.endswith(
        STAGE2_SUFFIX
    ):
        raise RuntimeError(
            "Formal Stage2 mesh does not end "
            f"with {STAGE2_SUFFIX}: {mesh}"
        )

    base = mesh.name[
        :-
        len(STAGE2_SUFFIX)
    ]

    source_obj = (
        mesh.parent
        /
        f"{base}.obj"
    ).resolve()

    sharp_file = (
        mesh.parent
        /
        f"{base}.sharp"
    ).resolve()

    source_exists = (
        source_obj.is_file()
    )

    sharp_exists = (
        sharp_file.is_file()
    )

    if source_exists != sharp_exists:
        raise RuntimeError(
            "Partial SHARP sidecar state:\n"
            f"mesh={mesh}\n"
            f"source_obj={source_obj} "
            f"exists={source_exists}\n"
            f"sharp_file={sharp_file} "
            f"exists={sharp_exists}"
        )

    if not sharp_exists:
        other_sharps = sorted(
            mesh.parent.glob(
                "*.sharp"
            )
        )

        if other_sharps:
            raise RuntimeError(
                "Derived SHARP sidecar is absent, "
                "but other SHARP files exist in "
                f"{mesh.parent}:\n"
                +
                "\n".join(
                    str(x)
                    for x in other_sharps
                )
            )

        return {
            "state":
                "NONE",

            "declared_count":
                None,

            "sharp_file":
                None,

            "sharp_file_sha256":
                None,

            "sharp_source_obj":
                None,

            "sharp_source_obj_sha256":
                None,
        }

    declared = (
        read_sharp_declared_count(
            sharp_file
        )
    )

    state = (
        "ACTIVE"
        if declared > 0
        else
        "EXPLICIT_ZERO"
    )

    return {
        "state":
            state,

        "declared_count":
            declared,

        "sharp_file":
            sharp_file,

        "sharp_file_sha256":
            sha256_file(
                sharp_file
            ),

        "sharp_source_obj":
            source_obj,

        "sharp_source_obj_sha256":
            sha256_file(
                source_obj
            ),
    }


def quality_ref_aggregate(
    rows,
):
    canonical = "\n".join(
        "\t".join(
            (
                row["model"],
                row["quality_ref_path"],
                str(
                    row[
                        "quality_ref_bytes"
                    ]
                ),
                row[
                    "quality_ref_sha256"
                ],
            )
        )
        for row in sorted(
            rows,
            key=lambda x:
                x["model"],
        )
    )

    return hashlib.sha256(
        canonical.encode(
            "utf-8"
        )
    ).hexdigest()


def write_csv(
    path: Path,
    rows,
):
    fieldnames = [
        "model",
        "complexity_stratum",

        "mesh_path",
        "mesh_bytes",
        "mesh_sha256",

        "loop_path",
        "loop_bytes",
        "loop_sha256",

        "sharp_state",
        "sharp_declared_count",
        "sharp_file",
        "sharp_file_sha256",
        "sharp_source_obj",
        "sharp_source_obj_sha256",

        "input_sample_seed_u64",
        "final_draw_seed_u64",

        "geometry_sample_count",
        "final_draw_count",
        "sharp_sample_count",

        "quality_ref_path",
        "quality_ref_bytes",
        "quality_ref_sha256",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key:
                        (
                            ""
                            if row[key] is None
                            else row[key]
                        )
                    for key in fieldnames
                }
            )


def build_one(
    *,
    model: str,
    mesh: Path,
    sidecar,
    backend: Path,
    output: Path,
):
    temporary = output.with_name(
        "."
        +
        output.name
        +
        ".batch_tmp"
    )

    if temporary.exists():
        temporary.unlink()

    command = [
        sys.executable,
        "-u",
        "-m",
        "quality.build_quality_ref_v1",

        "--model",
        model,

        "--stage2-input",
        str(mesh),

        "--output",
        str(temporary),
    ]

    if sidecar["state"] == "NONE":
        command.append(
            "--no-sharp"
        )

    else:
        command.extend(
            [
                "--sharp-file",
                str(
                    sidecar[
                        "sharp_file"
                    ]
                ),

                "--sharp-source-obj",
                str(
                    sidecar[
                        "sharp_source_obj"
                    ]
                ),
            ]
        )

        if sidecar[
            "state"
        ] == "ACTIVE":

            command.extend(
                [
                    "--point-triangle-backend",
                    str(backend),
                ]
            )

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH":
                str(PROJECT_ROOT),
        },
    )

    if completed.returncode != 0:
        if temporary.exists():
            temporary.unlink()

        raise RuntimeError(
            f"{model}: reference build failed:\n"
            +
            completed.stdout
        )

    ref = read_quality_ref_v1(
        temporary,
        require_v1_sample_counts=True,
        require_canonical_bytes=True,
    )

    if ref.model != model:
        raise RuntimeError(
            f"{model}: generated ref "
            f"contains model={ref.model}"
        )

    if (
        ref.stage2_input_sha256
        !=
        sha256_file(mesh)
    ):
        raise RuntimeError(
            f"{model}: Stage2 SHA mismatch "
            "inside generated ref"
        )

    expected_declared = (
        sidecar[
            "declared_count"
        ]
    )

    if (
        ref.sharp_declared_count
        !=
        expected_declared
    ):
        raise RuntimeError(
            f"{model}: generated SHARP declared "
            "count mismatch"
        )

    expected_present = (
        sidecar[
            "state"
        ]
        ==
        "ACTIVE"
    )

    if (
        ref.sharp_present
        !=
        expected_present
    ):
        raise RuntimeError(
            f"{model}: generated SHARP_PRESENT "
            "mismatch"
        )

    if (
        ref.sharp_file_sha256
        !=
        sidecar[
            "sharp_file_sha256"
        ]
    ):
        raise RuntimeError(
            f"{model}: generated SHARP SHA mismatch"
        )

    if (
        ref.sharp_source_obj_sha256
        !=
        sidecar[
            "sharp_source_obj_sha256"
        ]
    ):
        raise RuntimeError(
            f"{model}: generated SHARP source "
            "OBJ SHA mismatch"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.replace(
        temporary,
        output,
    )

    return ref


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-manifest",
        default=str(
            DEFAULT_DATASET_MANIFEST
        ),
    )

    parser.add_argument(
        "--formal-provenance",
        default=str(
            DEFAULT_FORMAL_PROVENANCE
        ),
    )

    parser.add_argument(
        "--point-triangle-backend",
        default=str(
            DEFAULT_POINT_TRIANGLE_BACKEND
        ),
    )

    parser.add_argument(
        "--output-root",
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    args = parser.parse_args()

    dataset_manifest = Path(
        args.dataset_manifest
    ).resolve()

    formal_provenance = Path(
        args.formal_provenance
    ).resolve()

    backend = Path(
        args.point_triangle_backend
    ).resolve()

    output_root = Path(
        args.output_root
    ).resolve()

    git_commit = (
        require_clean_git()
    )

    if not dataset_manifest.is_file():
        raise FileNotFoundError(
            dataset_manifest
        )

    if not formal_provenance.is_file():
        raise FileNotFoundError(
            formal_provenance
        )

    if not backend.is_file():
        raise FileNotFoundError(
            backend
        )

    if not os.access(
        backend,
        os.X_OK,
    ):
        raise RuntimeError(
            f"Backend is not executable: "
            f"{backend}"
        )


    # ========================================================
    # Frozen formal Train49 provenance.
    # ========================================================

    provenance = json.loads(
        formal_provenance.read_text(
            encoding="utf-8"
        )
    )

    expected_dataset_rel = Path(
        provenance[
            "dataset_manifest"
        ][
            "path"
        ]
    )

    expected_dataset = (
        PROJECT_ROOT
        /
        expected_dataset_rel
    ).resolve()

    if (
        dataset_manifest
        !=
        expected_dataset
    ):
        raise RuntimeError(
            "Dataset manifest path does not match "
            "formal frozen provenance"
        )

    dataset_sha = (
        sha256_file(
            dataset_manifest
        )
    )

    expected_dataset_sha = (
        provenance[
            "dataset_manifest"
        ][
            "sha256"
        ]
    )

    if (
        dataset_sha
        !=
        expected_dataset_sha
    ):
        raise RuntimeError(
            "Dataset manifest SHA mismatch"
        )

    (
        formal_records,
        formal_aggregate,
    ) = compute_train49_inputs(
        dataset_manifest=
            dataset_manifest,
    )

    if (
        formal_records
        !=
        provenance[
            "train49"
        ][
            "inputs"
        ]
    ):
        raise RuntimeError(
            "Current formal Train49 inputs differ "
            "from frozen formal provenance"
        )

    if (
        formal_aggregate
        !=
        provenance[
            "train49"
        ][
            "aggregate_sha256"
        ]
    ):
        raise RuntimeError(
            "Formal Train49 aggregate mismatch"
        )

    if len(
        formal_records
    ) != 49:
        raise RuntimeError(
            "Formal Train49 count is not 49"
        )


    # ========================================================
    # Dataset complexity strata.
    # ========================================================

    with dataset_manifest.open(
        newline="",
        encoding="utf-8",
    ) as f:
        dataset_rows = list(
            csv.DictReader(f)
        )

    train_by_model = {
        row[
            "model"
        ]:
            row

        for row in dataset_rows

        if row[
            "split"
        ] == "train"
    }

    if len(
        train_by_model
    ) != 49:
        raise RuntimeError(
            "Dataset Train49 count mismatch"
        )


    # ========================================================
    # Output directories.
    # ========================================================

    refs_dir = (
        output_root
        /
        "refs"
    )

    refs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Generate canonical refs.
    # ========================================================

    rows = []

    print(
        "=" * 110
    )

    print(
        "BUILD TRAIN49 QUALITY REFERENCES V1"
    )

    print(
        "=" * 110
    )

    print(
        "git_commit =",
        git_commit,
    )

    print(
        "formal_train49_aggregate =",
        formal_aggregate,
    )

    print(
        "output_root =",
        output_root,
    )

    print()


    for index, record in enumerate(
        formal_records,
        start=1,
    ):
        model = record[
            "model"
        ]

        mesh = Path(
            record[
                "mesh_path"
            ]
        ).resolve()

        sidecar = (
            derive_sharp_sidecars(
                mesh
            )
        )

        relative_ref = Path(
            "refs"
        ) / (
            model
            +
            ".quality_ref_v1"
        )

        output = (
            output_root
            /
            relative_ref
        )

        print(
            f"[{index:02d}/49] "
            f"{model:24s} "
            f"sharp="
            f"{sidecar['state']:13s} "
            f"declared="
            f"{str(sidecar['declared_count']):>5s}"
        )

        ref = build_one(
            model=model,
            mesh=mesh,
            sidecar=sidecar,
            backend=backend,
            output=output,
        )

        ref_sha = (
            sha256_file(
                output
            )
        )

        ref_bytes = int(
            output.stat().st_size
        )

        dataset_row = (
            train_by_model[
                model
            ]
        )

        rows.append(
            {
                "model":
                    model,

                "complexity_stratum":
                    int(
                        dataset_row[
                            "complexity_stratum"
                        ]
                    ),

                "mesh_path":
                    str(mesh),

                "mesh_bytes":
                    int(
                        record[
                            "mesh_bytes"
                        ]
                    ),

                "mesh_sha256":
                    record[
                        "mesh_sha256"
                    ],

                "loop_path":
                    record[
                        "loop_path"
                    ],

                "loop_bytes":
                    int(
                        record[
                            "loop_bytes"
                        ]
                    ),

                "loop_sha256":
                    record[
                        "loop_sha256"
                    ],

                "sharp_state":
                    sidecar[
                        "state"
                    ],

                "sharp_declared_count":
                    sidecar[
                        "declared_count"
                    ],

                "sharp_file":
                    (
                        None
                        if sidecar[
                            "sharp_file"
                        ] is None
                        else str(
                            sidecar[
                                "sharp_file"
                            ]
                        )
                    ),

                "sharp_file_sha256":
                    sidecar[
                        "sharp_file_sha256"
                    ],

                "sharp_source_obj":
                    (
                        None
                        if sidecar[
                            "sharp_source_obj"
                        ] is None
                        else str(
                            sidecar[
                                "sharp_source_obj"
                            ]
                        )
                    ),

                "sharp_source_obj_sha256":
                    sidecar[
                        "sharp_source_obj_sha256"
                    ],

                "input_sample_seed_u64":
                    int(
                        ref.input_sample_seed_u64
                    ),

                "final_draw_seed_u64":
                    int(
                        ref.final_draw_seed_u64
                    ),

                "geometry_sample_count":
                    len(
                        ref.input_geometry
                    ),

                "final_draw_count":
                    len(
                        ref.final_draws
                    ),

                "sharp_sample_count":
                    len(
                        ref.sharp_samples
                    ),

                "quality_ref_path":
                    relative_ref.as_posix(),

                "quality_ref_bytes":
                    ref_bytes,

                "quality_ref_sha256":
                    ref_sha,
            }
        )

        print(
            f"          samples="
            f"{len(ref.input_geometry)}/"
            f"{len(ref.final_draws)}/"
            f"{len(ref.sharp_samples)} "
            f"sha256={ref_sha[:16]}..."
        )


    rows.sort(
        key=lambda x:
            x["model"]
    )


    # ========================================================
    # Final aggregate + counts.
    # ========================================================

    ref_aggregate = (
        quality_ref_aggregate(
            rows
        )
    )

    active_count = sum(
        row[
            "sharp_state"
        ] == "ACTIVE"
        for row in rows
    )

    zero_count = sum(
        row[
            "sharp_state"
        ] == "EXPLICIT_ZERO"
        for row in rows
    )

    none_count = sum(
        row[
            "sharp_state"
        ] == "NONE"
        for row in rows
    )

    if (
        active_count,
        zero_count,
        none_count,
    ) != (
        29,
        20,
        0,
    ):
        raise RuntimeError(
            "Unexpected Train49 SHARP-state counts: "
            f"ACTIVE={active_count}, "
            f"EXPLICIT_ZERO={zero_count}, "
            f"NONE={none_count}"
        )


    # ========================================================
    # CSV manifest.
    # ========================================================

    manifest_csv = (
        output_root
        /
        "train49_quality_refs_v1.csv"
    )

    write_csv(
        manifest_csv,
        rows,
    )

    manifest_csv_sha = (
        sha256_file(
            manifest_csv
        )
    )


    # ========================================================
    # JSON manifest.
    # ========================================================

    core_hashes = {
        rel:
            sha256_file(
                PROJECT_ROOT
                /
                rel
            )

        for rel
        in QUALITY_REF_CORE_FILES
    }

    builder_rel = (
        "quality/"
        "build_quality_refs_train49_v1.py"
    )

    payload = {
        "schema_version":
            SCHEMA_VERSION,

        "metric_contract_sha256":
            METRIC_CONTRACT_V3_SHA256,

        "generator_environment": {
            "git_commit":
                git_commit,

            "python_version":
                platform.python_version(),

            "numpy_version":
                np.__version__,

            "batch_builder": {
                "path":
                    builder_rel,

                "sha256":
                    sha256_file(
                        PROJECT_ROOT
                        /
                        builder_rel
                    ),
            },

            "quality_ref_core_files":
                core_hashes,
        },

        "dataset_manifest": {
            "path":
                str(
                    dataset_manifest
                    .relative_to(
                        PROJECT_ROOT
                    )
                ),

            "sha256":
                dataset_sha,
        },

        "formal_training_input_provenance": {
            "path":
                str(
                    formal_provenance
                    .relative_to(
                        PROJECT_ROOT
                    )
                ),

            "sha256":
                sha256_file(
                    formal_provenance
                ),

            "train49_aggregate_sha256":
                formal_aggregate,
        },

        "point_triangle_backend": {
            "path":
                str(backend),

            "sha256":
                sha256_file(
                    backend
                ),
        },

        "counts": {
            "models":
                len(rows),

            "active_sharp":
                active_count,

            "explicit_zero_sharp":
                zero_count,

            "no_sharp_source":
                none_count,
        },

        "quality_ref_aggregate": {
            "algorithm":
                (
                    "SORT_BY_MODEL_THEN_SHA256_OF_"
                    "TAB_SEPARATED_"
                    "MODEL_REF_PATH_REF_BYTES_REF_SHA256"
                ),

            "sha256":
                ref_aggregate,
        },

        "csv_manifest": {
            "path":
                manifest_csv.name,

            "sha256":
                manifest_csv_sha,
        },

        "records":
            rows,
    }

    manifest_json = (
        output_root
        /
        "train49_quality_refs_v1.json"
    )

    manifest_json.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
        newline="\n",
    )

    manifest_json_sha = (
        sha256_file(
            manifest_json
        )
    )


    # ========================================================
    # SHA256SUMS.
    # ========================================================

    sums = []

    for row in rows:
        sums.append(
            (
                row[
                    "quality_ref_sha256"
                ],
                row[
                    "quality_ref_path"
                ],
            )
        )

    sums.extend(
        [
            (
                manifest_csv_sha,
                manifest_csv.name,
            ),

            (
                manifest_json_sha,
                manifest_json.name,
            ),
        ]
    )

    sums_path = (
        output_root
        /
        "SHA256SUMS.txt"
    )

    sums_path.write_text(
        "".join(
            f"{sha}  {rel}\n"
            for sha, rel in sums
        ),
        encoding="utf-8",
        newline="\n",
    )


    print()

    print(
        "=" * 110
    )

    print(
        "PASS: BUILT TRAIN49 QUALITY REFERENCES V1"
    )

    print(
        "=" * 110
    )

    print(
        "models =",
        len(rows),
    )

    print(
        "ACTIVE =",
        active_count,
    )

    print(
        "EXPLICIT_ZERO =",
        zero_count,
    )

    print(
        "NONE =",
        none_count,
    )

    print(
        "quality_ref_aggregate_sha256 =",
        ref_aggregate,
    )

    print(
        "manifest_csv_sha256 =",
        manifest_csv_sha,
    )

    print(
        "manifest_json_sha256 =",
        manifest_json_sha,
    )

    print(
        "SHA256SUMS_sha256 =",
        sha256_file(
            sums_path
        ),
    )

    print(
        "output_root =",
        output_root,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
