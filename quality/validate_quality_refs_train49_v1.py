from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
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


from quality.quality_ref_v1 import (
    FINAL_DRAW_COUNT_V1,
    GEOMETRY_SAMPLE_COUNT_V1,
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

DEFAULT_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "quality_refs_train49_v1"
)

EXPECTED_CORE_FILES = {
    "quality/quality_ref_v1.py",
    "quality/build_quality_ref_v1.py",
    "quality/validate_quality_ref_v1.py",
}

EXPECTED_BUILDER = (
    "quality/build_quality_refs_train49_v1.py"
)

STAGE2_SUFFIX = "_splitted.obj"


def sha256_bytes(
    data: bytes,
) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def stable_seed(
    model: str,
    role: str,
) -> int:
    digest = hashlib.sha256(
        (
            "loopycuts_seed42_local_geometry42_v1"
            "|"
            + model
            + "|"
            + role
        ).encode(
            "utf-8"
        )
    ).digest()

    return int.from_bytes(
        digest[:8],
        "little",
    )


def git_blob(
    commit: str,
    path: str,
) -> bytes:
    completed = subprocess.run(
        [
            "git",
            "show",
            f"{commit}:{path}",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Cannot read frozen Git blob:\n"
            f"commit={commit}\n"
            f"path={path}\n"
            +
            completed.stderr.decode(
                "utf-8",
                errors="replace",
            )
        )

    return completed.stdout


def read_declared_count(
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
            not raw.lstrip().startswith("#")
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

    if declared != len(
        lines[1:]
    ):
        raise RuntimeError(
            f"{path}: declared/records mismatch"
        )

    return declared


def derive_sidecar(
    mesh: Path,
):
    if not mesh.name.endswith(
        STAGE2_SUFFIX
    ):
        raise RuntimeError(
            f"{mesh}: bad Stage2 suffix"
        )

    base = mesh.name[
        :-
        len(STAGE2_SUFFIX)
    ]

    source = (
        mesh.parent
        /
        f"{base}.obj"
    ).resolve()

    sharp = (
        mesh.parent
        /
        f"{base}.sharp"
    ).resolve()

    if (
        source.is_file()
        !=
        sharp.is_file()
    ):
        raise RuntimeError(
            f"{mesh}: partial SHARP sidecar"
        )

    if not sharp.is_file():
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
        read_declared_count(
            sharp
        )
    )

    return {
        "state":
            (
                "ACTIVE"
                if declared > 0
                else
                "EXPLICIT_ZERO"
            ),

        "declared_count":
            declared,

        "sharp_file":
            str(sharp),

        "sharp_file_sha256":
            sha256_file(
                sharp
            ),

        "sharp_source_obj":
            str(source),

        "sharp_source_obj_sha256":
            sha256_file(
                source
            ),
    }


def ref_aggregate(
    rows,
) -> str:
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


def fail(
    message: str,
):
    raise RuntimeError(
        message
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=str(
            DEFAULT_ROOT
        ),
    )

    args = parser.parse_args()

    root = Path(
        args.root
    ).resolve()

    manifest_json = (
        root
        /
        "train49_quality_refs_v1.json"
    )

    manifest_csv = (
        root
        /
        "train49_quality_refs_v1.csv"
    )

    sums_path = (
        root
        /
        "SHA256SUMS.txt"
    )

    for path in (
        manifest_json,
        manifest_csv,
        sums_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )


    # ========================================================
    # JSON contract.
    # ========================================================

    payload = json.loads(
        manifest_json.read_text(
            encoding="utf-8"
        )
    )

    if (
        payload.get(
            "schema_version"
        )
        !=
        SCHEMA_VERSION
    ):
        fail(
            "Manifest schema mismatch"
        )

    if (
        payload.get(
            "metric_contract_sha256"
        )
        !=
        METRIC_CONTRACT_V3_SHA256
    ):
        fail(
            "Metric-contract SHA mismatch"
        )


    # ========================================================
    # Frozen generator source provenance.
    # ========================================================

    env = payload[
        "generator_environment"
    ]

    generator_commit = env[
        "git_commit"
    ]

    builder = env[
        "batch_builder"
    ]

    if (
        builder[
            "path"
        ]
        !=
        EXPECTED_BUILDER
    ):
        fail(
            "Unexpected batch-builder path"
        )

    builder_blob = git_blob(
        generator_commit,
        EXPECTED_BUILDER,
    )

    if (
        sha256_bytes(
            builder_blob
        )
        !=
        builder[
            "sha256"
        ]
    ):
        fail(
            "Frozen batch-builder SHA mismatch"
        )

    core = env[
        "quality_ref_core_files"
    ]

    if set(
        core
    ) != EXPECTED_CORE_FILES:
        fail(
            "Unexpected quality-ref core-file set"
        )

    for rel, expected_sha in core.items():
        actual_sha = sha256_bytes(
            git_blob(
                generator_commit,
                rel,
            )
        )

        if actual_sha != expected_sha:
            fail(
                f"Frozen source SHA mismatch: "
                f"{rel}"
            )


    # ========================================================
    # Dataset + formal Train49 provenance.
    # ========================================================

    dataset_info = payload[
        "dataset_manifest"
    ]

    dataset_path = (
        PROJECT_ROOT
        /
        dataset_info[
            "path"
        ]
    ).resolve()

    if not dataset_path.is_file():
        raise FileNotFoundError(
            dataset_path
        )

    if (
        sha256_file(
            dataset_path
        )
        !=
        dataset_info[
            "sha256"
        ]
    ):
        fail(
            "Dataset manifest SHA mismatch"
        )


    formal_info = payload[
        "formal_training_input_provenance"
    ]

    formal_path = (
        PROJECT_ROOT
        /
        formal_info[
            "path"
        ]
    ).resolve()

    if not formal_path.is_file():
        raise FileNotFoundError(
            formal_path
        )

    if (
        sha256_file(
            formal_path
        )
        !=
        formal_info[
            "sha256"
        ]
    ):
        fail(
            "Formal provenance SHA mismatch"
        )


    frozen_formal = json.loads(
        formal_path.read_text(
            encoding="utf-8"
        )
    )

    (
        formal_records,
        formal_aggregate,
    ) = compute_train49_inputs(
        dataset_manifest=
            dataset_path,
    )

    if (
        formal_records
        !=
        frozen_formal[
            "train49"
        ][
            "inputs"
        ]
    ):
        fail(
            "Formal Train49 per-file "
            "provenance mismatch"
        )

    if (
        formal_aggregate
        !=
        frozen_formal[
            "train49"
        ][
            "aggregate_sha256"
        ]
    ):
        fail(
            "Formal Train49 aggregate mismatch"
        )

    if (
        formal_aggregate
        !=
        formal_info[
            "train49_aggregate_sha256"
        ]
    ):
        fail(
            "Quality manifest references wrong "
            "formal Train49 aggregate"
        )

    if len(
        formal_records
    ) != 49:
        fail(
            "Formal Train49 count != 49"
        )


    formal_by_model = {
        row[
            "model"
        ]:
            row
        for row in formal_records
    }


    # ========================================================
    # Backend provenance.
    # ========================================================

    backend_info = payload[
        "point_triangle_backend"
    ]

    backend = Path(
        backend_info[
            "path"
        ]
    )

    if not backend.is_file():
        raise FileNotFoundError(
            backend
        )

    if (
        sha256_file(
            backend
        )
        !=
        backend_info[
            "sha256"
        ]
    ):
        fail(
            "Point-triangle backend SHA mismatch"
        )


    # ========================================================
    # JSON records.
    # ========================================================

    rows = payload[
        "records"
    ]

    if len(rows) != 49:
        fail(
            "Quality manifest record count != 49"
        )

    models = [
        row[
            "model"
        ]
        for row in rows
    ]

    if (
        len(
            set(models)
        )
        !=
        49
    ):
        fail(
            "Duplicate model in quality manifest"
        )

    if models != sorted(
        models
    ):
        fail(
            "JSON quality records are not "
            "sorted by model"
        )

    if set(
        models
    ) != set(
        formal_by_model
    ):
        fail(
            "Quality-ref model set differs "
            "from formal Train49"
        )


    # Exact ref directory membership.
    refs_dir = (
        root
        /
        "refs"
    )

    actual_refs = {
        path.name
        for path in refs_dir.glob(
            "*.quality_ref_v1"
        )
    }

    expected_refs = {
        f"{model}.quality_ref_v1"
        for model in models
    }

    if actual_refs != expected_refs:
        fail(
            "refs/ exact file set mismatch"
        )

    stale_tmp = list(
        root.rglob(
            "*.batch_tmp"
        )
    )

    if stale_tmp:
        fail(
            "Stale batch temporary files exist:\n"
            +
            "\n".join(
                str(x)
                for x in stale_tmp
            )
        )


    # ========================================================
    # Validate each model independently.
    # ========================================================

    active = 0
    zero = 0
    none = 0


    for index, row in enumerate(
        rows,
        start=1,
    ):
        model = row[
            "model"
        ]

        formal = formal_by_model[
            model
        ]

        mesh = Path(
            formal[
                "mesh_path"
            ]
        ).resolve()

        loop = Path(
            formal[
                "loop_path"
            ]
        ).resolve()


        # Formal mesh/loop record must be copied exactly.
        for key in (
            "mesh_path",
            "mesh_bytes",
            "mesh_sha256",
            "loop_path",
            "loop_bytes",
            "loop_sha256",
        ):
            if row[
                key
            ] != formal[
                key
            ]:
                fail(
                    f"{model}: formal field "
                    f"mismatch: {key}"
                )


        if (
            mesh.stat().st_size
            !=
            row[
                "mesh_bytes"
            ]
        ):
            fail(
                f"{model}: mesh byte-size mismatch"
            )

        if (
            sha256_file(
                mesh
            )
            !=
            row[
                "mesh_sha256"
            ]
        ):
            fail(
                f"{model}: mesh SHA mismatch"
            )

        if (
            loop.stat().st_size
            !=
            row[
                "loop_bytes"
            ]
        ):
            fail(
                f"{model}: loop byte-size mismatch"
            )

        if (
            sha256_file(
                loop
            )
            !=
            row[
                "loop_sha256"
            ]
        ):
            fail(
                f"{model}: loop SHA mismatch"
            )


        sidecar = derive_sidecar(
            mesh
        )

        for key in (
            "state",
            "declared_count",
            "sharp_file",
            "sharp_file_sha256",
            "sharp_source_obj",
            "sharp_source_obj_sha256",
        ):
            manifest_key = {
                "state":
                    "sharp_state",

                "declared_count":
                    "sharp_declared_count",

            }.get(
                key,
                key,
            )

            if (
                row[
                    manifest_key
                ]
                !=
                sidecar[
                    key
                ]
            ):
                fail(
                    f"{model}: SHARP provenance "
                    f"mismatch: {manifest_key}"
                )


        if sidecar[
            "state"
        ] == "ACTIVE":
            active += 1

        elif sidecar[
            "state"
        ] == "EXPLICIT_ZERO":
            zero += 1

        elif sidecar[
            "state"
        ] == "NONE":
            none += 1

        else:
            fail(
                f"{model}: unknown SHARP state"
            )


        expected_rel = (
            f"refs/"
            f"{model}.quality_ref_v1"
        )

        if (
            row[
                "quality_ref_path"
            ]
            !=
            expected_rel
        ):
            fail(
                f"{model}: unexpected ref path"
            )

        ref_path = (
            root
            /
            row[
                "quality_ref_path"
            ]
        ).resolve()

        if (
            root
            not in
            ref_path.parents
        ):
            fail(
                f"{model}: ref escapes output root"
            )

        if not ref_path.is_file():
            raise FileNotFoundError(
                ref_path
            )

        actual_bytes = int(
            ref_path.stat().st_size
        )

        actual_sha = (
            sha256_file(
                ref_path
            )
        )

        if (
            actual_bytes
            !=
            row[
                "quality_ref_bytes"
            ]
        ):
            fail(
                f"{model}: ref byte-size mismatch"
            )

        if (
            actual_sha
            !=
            row[
                "quality_ref_sha256"
            ]
        ):
            fail(
                f"{model}: ref SHA mismatch"
            )


        ref = read_quality_ref_v1(
            ref_path,
            require_v1_sample_counts=True,
            require_canonical_bytes=True,
        )

        if ref.model != model:
            fail(
                f"{model}: embedded model mismatch"
            )

        if (
            ref.metric_contract_sha256
            !=
            METRIC_CONTRACT_V3_SHA256
        ):
            fail(
                f"{model}: embedded metric "
                "contract mismatch"
            )

        if (
            ref.stage2_input_sha256
            !=
            row[
                "mesh_sha256"
            ]
        ):
            fail(
                f"{model}: embedded Stage2 "
                "SHA mismatch"
            )

        if (
            ref.sharp_declared_count
            !=
            row[
                "sharp_declared_count"
            ]
        ):
            fail(
                f"{model}: embedded SHARP "
                "declared count mismatch"
            )

        expected_present = (
            row[
                "sharp_state"
            ]
            ==
            "ACTIVE"
        )

        if (
            ref.sharp_present
            !=
            expected_present
        ):
            fail(
                f"{model}: embedded "
                "SHARP_PRESENT mismatch"
            )

        if (
            ref.sharp_file_sha256
            !=
            row[
                "sharp_file_sha256"
            ]
        ):
            fail(
                f"{model}: embedded SHARP "
                "file SHA mismatch"
            )

        if (
            ref.sharp_source_obj_sha256
            !=
            row[
                "sharp_source_obj_sha256"
            ]
        ):
            fail(
                f"{model}: embedded SHARP "
                "source SHA mismatch"
            )


        expected_input_seed = (
            stable_seed(
                model,
                "input",
            )
        )

        expected_final_seed = (
            stable_seed(
                model,
                "rl_final",
            )
        )

        if (
            ref.input_sample_seed_u64
            !=
            expected_input_seed
            or
            row[
                "input_sample_seed_u64"
            ]
            !=
            expected_input_seed
        ):
            fail(
                f"{model}: input seed mismatch"
            )

        if (
            ref.final_draw_seed_u64
            !=
            expected_final_seed
            or
            row[
                "final_draw_seed_u64"
            ]
            !=
            expected_final_seed
        ):
            fail(
                f"{model}: final-draw seed mismatch"
            )


        if (
            len(
                ref.input_geometry
            )
            !=
            GEOMETRY_SAMPLE_COUNT_V1
            or
            row[
                "geometry_sample_count"
            ]
            !=
            GEOMETRY_SAMPLE_COUNT_V1
        ):
            fail(
                f"{model}: geometry sample "
                "count mismatch"
            )

        if (
            len(
                ref.final_draws
            )
            !=
            FINAL_DRAW_COUNT_V1
            or
            row[
                "final_draw_count"
            ]
            !=
            FINAL_DRAW_COUNT_V1
        ):
            fail(
                f"{model}: final draw "
                "count mismatch"
            )

        if (
            len(
                ref.sharp_samples
            )
            !=
            row[
                "sharp_sample_count"
            ]
        ):
            fail(
                f"{model}: SHARP sample "
                "count mismatch"
            )

        if (
            row[
                "sharp_state"
            ]
            ==
            "ACTIVE"
            and
            len(
                ref.sharp_samples
            )
            <=
            0
        ):
            fail(
                f"{model}: active SHARP "
                "has zero samples"
            )

        if (
            row[
                "sharp_state"
            ]
            !=
            "ACTIVE"
            and
            len(
                ref.sharp_samples
            )
            !=
            0
        ):
            fail(
                f"{model}: inactive SHARP "
                "has samples"
            )


        print(
            f"[{index:02d}/49] "
            f"PASS {model:24s} "
            f"sharp="
            f"{row['sharp_state']:13s} "
            f"samples="
            f"{len(ref.input_geometry)}/"
            f"{len(ref.final_draws)}/"
            f"{len(ref.sharp_samples)} "
            f"sha="
            f"{actual_sha[:16]}..."
        )


    # ========================================================
    # Counts.
    # ========================================================

    if (
        active,
        zero,
        none,
    ) != (
        29,
        20,
        0,
    ):
        fail(
            "Train49 SHARP-state count mismatch"
        )

    counts = payload[
        "counts"
    ]

    if counts != {
        "models":
            49,

        "active_sharp":
            29,

        "explicit_zero_sharp":
            20,

        "no_sharp_source":
            0,
    }:
        fail(
            "JSON count summary mismatch"
        )


    # ========================================================
    # Quality-ref aggregate.
    # ========================================================

    actual_ref_aggregate = (
        ref_aggregate(
            rows
        )
    )

    if (
        actual_ref_aggregate
        !=
        payload[
            "quality_ref_aggregate"
        ][
            "sha256"
        ]
    ):
        fail(
            "Quality-ref aggregate mismatch"
        )


    # ========================================================
    # CSV manifest exact row semantics.
    # ========================================================

    if (
        sha256_file(
            manifest_csv
        )
        !=
        payload[
            "csv_manifest"
        ][
            "sha256"
        ]
    ):
        fail(
            "CSV manifest SHA mismatch"
        )

    with manifest_csv.open(
        newline="",
        encoding="utf-8",
    ) as f:
        csv_rows = list(
            csv.DictReader(f)
        )

    if len(
        csv_rows
    ) != 49:
        fail(
            "CSV manifest row count != 49"
        )

    if [
        row[
            "model"
        ]
        for row in csv_rows
    ] != models:
        fail(
            "CSV model ordering mismatch"
        )

    for json_row, csv_row in zip(
        rows,
        csv_rows,
        strict=True,
    ):
        for key, csv_value in csv_row.items():

            json_value = (
                json_row[
                    key
                ]
            )

            expected = (
                ""
                if json_value is None
                else str(
                    json_value
                )
            )

            if csv_value != expected:
                fail(
                    f"{json_row['model']}: "
                    f"CSV/JSON mismatch: {key}"
                )


    # ========================================================
    # SHA256SUMS exact contents.
    # ========================================================

    manifest_json_sha = (
        sha256_file(
            manifest_json
        )
    )

    expected_sums = "".join(
        (
            f"{row['quality_ref_sha256']}  "
            f"{row['quality_ref_path']}\n"
        )
        for row in rows
    )

    expected_sums += (
        f"{sha256_file(manifest_csv)}  "
        f"{manifest_csv.name}\n"
    )

    expected_sums += (
        f"{manifest_json_sha}  "
        f"{manifest_json.name}\n"
    )

    actual_sums = (
        sums_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        actual_sums
        !=
        expected_sums
    ):
        fail(
            "SHA256SUMS.txt exact-content mismatch"
        )


    print()
    print(
        "=" * 110
    )
    print(
        "PASS: VALIDATED TRAIN49 "
        "QUALITY REFERENCES V1"
    )
    print(
        "=" * 110
    )

    print(
        "generator_commit =",
        generator_commit,
    )

    print(
        "formal_train49_aggregate_sha256 =",
        formal_aggregate,
    )

    print(
        "models =",
        len(rows),
    )

    print(
        "ACTIVE =",
        active,
    )

    print(
        "EXPLICIT_ZERO =",
        zero,
    )

    print(
        "NONE =",
        none,
    )

    print(
        "quality_ref_aggregate_sha256 =",
        actual_ref_aggregate,
    )

    print(
        "manifest_csv_sha256 =",
        sha256_file(
            manifest_csv
        ),
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

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
