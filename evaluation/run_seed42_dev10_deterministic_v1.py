from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback

from dataclasses import dataclass
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


import evaluation.run_seed42_train48_deterministic_v1 as base


SEED = 42

EVALUATOR_VERSION = (
    "loopycuts_seed42_dev10_deterministic_evaluator_v1"
)

RESULT_SCHEMA_VERSION = (
    "loopycuts_seed42_dev10_model_result_v1"
)

RUN_MANIFEST_SCHEMA_VERSION = (
    "loopycuts_seed42_dev10_run_manifest_v1"
)

SUMMARY_SCHEMA_VERSION = (
    "loopycuts_seed42_dev10_summary_v1"
)

RUN_COMPLETE_SCHEMA_VERSION = (
    "loopycuts_seed42_dev10_run_complete_v1"
)


DATASET_MANIFEST = (
    PROJECT_ROOT
    /
    "data/manifests/dataset_split_v2.csv"
)

EXPECTED_DATASET_MANIFEST_SHA256 = (
    "e7bc6ba976417d427d9a105f5b90a54"
    "c304c08fdd0baff542e083c8f42ff826b"
)


QUALITY_REF_SET_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "quality_refs_dev10_v1"
)

QUALITY_REF_MANIFEST = (
    QUALITY_REF_SET_ROOT
    /
    "manifest_v1.csv"
)

QUALITY_REF_PROVENANCE = (
    QUALITY_REF_SET_ROOT
    /
    "provenance_v1.json"
)

QUALITY_REF_SHA256SUMS = (
    QUALITY_REF_SET_ROOT
    /
    "SHA256SUMS.txt"
)


EXPECTED_QUALITY_REF_AGGREGATE_SHA256 = (
    "7baccb7369b28e1205bbcac5bb8a3093"
    "3f74b34fe1facc9991bd65a7d539580b"
)

EXPECTED_QUALITY_REF_MANIFEST_SHA256 = (
    "138ab478c06281c7b41713882cb2fd78"
    "2ceee53483331d9cc861cf2375142f14"
)

EXPECTED_QUALITY_REF_PROVENANCE_SHA256 = (
    "c8c2d30e29c2f2123e6691b15302ade"
    "483a72c8b7a06c31cf8ccde48af4872f9"
)

EXPECTED_QUALITY_REF_SHA256SUMS_IDENTITY = (
    "bfd11d88ba6a498bc70fffd399195beb"
    "57b2deb25d5b4830ac0877e60674f1a4"
)


DEFAULT_RUN_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "evaluation_v5_seed42/"
    "dev10_deterministic_v1"
)


DEV10_NAMES = (
    "kiss",
    "sphinx",
    "bone1",
    "ellipse",
    "vessel",
    "busto_bimba",
    "sculpt",
    "beveled_shoulder_2",
    "pinion",
    "rod",
)


@dataclass(frozen=True)
class Dev10Model:
    model: str
    complexity_stratum: int
    mesh_file: Path
    loop_file: Path
    quality_ref_file: Path


def configure_base_globals():
    """
    Reuse the already validated Train48 evaluator implementation
    for actor inference, C++ STEP execution, ResourceGuard,
    FINALIZE_QUALITY_EXPORT, failure classification, and geometry
    evidence.

    Only dataset/result schema and quality-reference identity differ.
    """

    base.EVALUATOR_VERSION = (
        EVALUATOR_VERSION
    )

    base.RESULT_SCHEMA_VERSION = (
        RESULT_SCHEMA_VERSION
    )

    base.RUN_MANIFEST_SCHEMA_VERSION = (
        RUN_MANIFEST_SCHEMA_VERSION
    )

    base.SUMMARY_SCHEMA_VERSION = (
        SUMMARY_SCHEMA_VERSION
    )

    base.SEED = SEED

    base.TRAINING_DATASET_NAME = (
        "Train49"
    )

    base.OPERATIONAL_DATASET_NAME = (
        "Dev10"
    )

    # No Train48-style operational exclusion exists in Dev10.
    base.EXCLUDED_MODEL = None

    base.QUALITY_REF_SET_ROOT = (
        QUALITY_REF_SET_ROOT
    )

    base.QUALITY_REF_SHA256SUMS = (
        QUALITY_REF_SHA256SUMS
    )

    base.EXPECTED_QUALITY_REF_SHA256SUMS_IDENTITY = (
        EXPECTED_QUALITY_REF_SHA256SUMS_IDENTITY
    )


def require_frozen_dev_inputs():
    if not DATASET_MANIFEST.is_file():
        raise FileNotFoundError(
            DATASET_MANIFEST
        )

    actual_dataset_sha = (
        base.sha256_file(
            DATASET_MANIFEST
        )
    )

    if (
        actual_dataset_sha
        !=
        EXPECTED_DATASET_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "Frozen Dataset Split V2 SHA256 mismatch"
        )

    for path, expected in (
        (
            QUALITY_REF_MANIFEST,
            EXPECTED_QUALITY_REF_MANIFEST_SHA256,
        ),
        (
            QUALITY_REF_PROVENANCE,
            EXPECTED_QUALITY_REF_PROVENANCE_SHA256,
        ),
        (
            QUALITY_REF_SHA256SUMS,
            EXPECTED_QUALITY_REF_SHA256SUMS_IDENTITY,
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        actual = (
            base.sha256_file(
                path
            )
        )

        if actual != expected:
            raise RuntimeError(
                "Frozen Dev10 quality-ref "
                f"artifact mismatch: {path}"
            )

    provenance = json.loads(
        QUALITY_REF_PROVENANCE.read_text(
            encoding="utf-8"
        )
    )

    if (
        provenance.get(
            "schema_version"
        )
        !=
        "loopycuts_quality_refs_dev10_v1"
    ):
        raise RuntimeError(
            "Dev10 quality-ref provenance "
            "schema mismatch"
        )

    if int(
        provenance.get(
            "model_count",
            -1,
        )
    ) != 10:
        raise RuntimeError(
            "Dev10 quality-ref provenance "
            "model count mismatch"
        )

    if (
        provenance.get(
            "quality_ref_aggregate_sha256"
        )
        !=
        EXPECTED_QUALITY_REF_AGGREGATE_SHA256
    ):
        raise RuntimeError(
            "Dev10 quality-ref aggregate mismatch"
        )


def load_dev10_models():
    require_frozen_dev_inputs()

    with DATASET_MANIFEST.open(
        newline="",
        encoding="utf-8",
    ) as f:
        dataset_rows = list(
            csv.DictReader(f)
        )

    dev_rows = [
        row
        for row in dataset_rows
        if row["split"] == "dev"
    ]

    train_names = {
        row["model"]
        for row in dataset_rows
        if row["split"] == "train"
    }

    if len(dev_rows) != 10:
        raise RuntimeError(
            "Dataset Split V2 Dev count != 10"
        )

    dev_rows.sort(
        key=lambda row: (
            int(
                row[
                    "complexity_stratum"
                ]
            ),
            row["model"],
        )
    )

    names = tuple(
        row["model"]
        for row in dev_rows
    )

    if names != DEV10_NAMES:
        raise RuntimeError(
            "Frozen Dev10 model identity/order mismatch: "
            f"{names}"
        )

    overlap = (
        set(names)
        &
        train_names
    )

    if overlap:
        raise RuntimeError(
            "Dev10 leaked into Train49: "
            f"{sorted(overlap)}"
        )

    with QUALITY_REF_MANIFEST.open(
        newline="",
        encoding="utf-8",
    ) as f:
        quality_rows = list(
            csv.DictReader(f)
        )

    if len(quality_rows) != 10:
        raise RuntimeError(
            "Dev10 quality-ref manifest count != 10"
        )

    quality_by_model = {
        row["model"]:
            row
        for row in quality_rows
    }

    if (
        set(quality_by_model)
        !=
        set(names)
    ):
        raise RuntimeError(
            "Dev10 quality-ref model set mismatch"
        )

    models = []

    for row in dev_rows:
        model = row["model"]

        qrow = (
            quality_by_model[
                model
            ]
        )

        stratum = int(
            row[
                "complexity_stratum"
            ]
        )

        if int(
            qrow[
                "complexity_stratum"
            ]
        ) != stratum:
            raise RuntimeError(
                f"{model}: complexity stratum mismatch"
            )

        mesh = Path(
            row[
                "mesh_file"
            ]
        ).resolve()

        loops = Path(
            row[
                "loop_file"
            ]
        ).resolve()

        ref = (
            QUALITY_REF_SET_ROOT
            /
            qrow[
                "quality_ref_path"
            ]
        ).resolve()

        for path in (
            mesh,
            loops,
            ref,
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"{model}: {path}"
                )

        if (
            base.sha256_file(
                mesh
            )
            !=
            qrow[
                "mesh_sha256"
            ]
        ):
            raise RuntimeError(
                f"{model}: mesh SHA mismatch"
            )

        if (
            base.sha256_file(
                loops
            )
            !=
            qrow[
                "loop_sha256"
            ]
        ):
            raise RuntimeError(
                f"{model}: loop SHA mismatch"
            )

        if (
            base.sha256_file(
                ref
            )
            !=
            qrow[
                "quality_ref_sha256"
            ]
        ):
            raise RuntimeError(
                f"{model}: quality-ref SHA mismatch"
            )

        models.append(
            Dev10Model(
                model=model,
                complexity_stratum=
                    stratum,
                mesh_file=mesh,
                loop_file=loops,
                quality_ref_file=ref,
            )
        )

    return tuple(
        models
    )


def build_dev10_run_manifest(
    *,
    runtime,
    models,
    runner_commit,
):
    """
    Start from the already audited Train48 manifest builder so the
    actor, C++ executable, evaluation provenance, Train49 training
    provenance, resource-guard contract, and per-model hashes are
    checked identically.
    """

    require_frozen_dev_inputs()

    candidate = (
        base.build_run_manifest(
            runtime=runtime,
            models=models,
            runner_commit=runner_commit,
        )
    )

    candidate[
        "schema_version"
    ] = RUN_MANIFEST_SCHEMA_VERSION

    candidate[
        "evaluator_version"
    ] = EVALUATOR_VERSION

    candidate[
        "dataset"
    ][
        "operational_evaluation_dataset"
    ] = "Dev10"

    candidate[
        "dataset"
    ][
        "operational_model_count"
    ] = 10

    candidate[
        "dataset"
    ][
        "excluded_model"
    ] = None

    candidate[
        "dataset"
    ][
        "exclusion_scope"
    ] = None

    candidate[
        "dataset"
    ][
        "dataset_manifest_path"
    ] = str(
        DATASET_MANIFEST
    )

    candidate[
        "dataset"
    ][
        "dataset_manifest_sha256"
    ] = (
        EXPECTED_DATASET_MANIFEST_SHA256
    )

    candidate[
        "quality_refs"
    ] = {
        "set":
            "quality_refs_dev10_v1",

        "root":
            str(
                QUALITY_REF_SET_ROOT
            ),

        "manifest_path":
            str(
                QUALITY_REF_MANIFEST
            ),

        "manifest_sha256":
            EXPECTED_QUALITY_REF_MANIFEST_SHA256,

        "provenance_path":
            str(
                QUALITY_REF_PROVENANCE
            ),

        "provenance_sha256":
            EXPECTED_QUALITY_REF_PROVENANCE_SHA256,

        "quality_ref_aggregate_sha256":
            EXPECTED_QUALITY_REF_AGGREGATE_SHA256,

        "sha256sums_path":
            str(
                QUALITY_REF_SHA256SUMS
            ),

        "sha256sums_identity":
            EXPECTED_QUALITY_REF_SHA256SUMS_IDENTITY,
    }

    return candidate


def summarize_dev10(
    records,
):
    records = list(
        records
    )

    summary = (
        base.summarize_records(
            records
        )
    )

    summary[
        "schema_version"
    ] = SUMMARY_SCHEMA_VERSION

    summary[
        "expected_models"
    ] = 10

    summary[
        "completed_model_records"
    ] = len(
        records
    )

    summary[
        "complete"
    ] = (
        len(
            records
        )
        == 10
    )

    summary.pop(
        "full_hex_rate_over_operational_train48",
        None,
    )

    full_hex_count = int(
        summary[
            "full_hex_count"
        ]
    )

    summary[
        "full_hex_rate_over_dev10"
    ] = float(
        full_hex_count
        /
        10.0
    )

    return summary


def collect_existing_results(
    *,
    run_root,
    models,
    runner_commit,
):
    records = []

    result_root = (
        Path(
            run_root
        )
        /
        "results"
    )

    for model in models:
        path = (
            result_root
            /
            f"{model.model}.json"
        )

        if not path.is_file():
            continue

        result = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        base.validate_existing_result(
            result=result,
            model=model,
            runner_commit=runner_commit,
        )

        records.append(
            result
        )

    return records


def write_summary(
    *,
    run_root,
    models,
    runner_commit,
):
    records = (
        collect_existing_results(
            run_root=run_root,
            models=models,
            runner_commit=runner_commit,
        )
    )

    summary = (
        summarize_dev10(
            records
        )
    )

    base.atomic_write_json(
        Path(
            run_root
        )
        /
        "summary.json",
        summary,
    )

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    args = parser.parse_args()

    run_root = (
        args.run_root
        .resolve()
    )

    if any(
        ch.isspace()
        for ch
        in str(
            run_root
        )
    ):
        raise RuntimeError(
            "Dev10 run_root must not contain whitespace"
        )

    configure_base_globals()


    print(
        "===== DEV10 EVALUATION SOFTWARE ====="
    )

    runner_commit = (
        base.git_head(
            PROJECT_ROOT
        )
    )

    runner_clean = (
        base.git_is_clean(
            PROJECT_ROOT
        )
    )

    print(
        "runner_git_commit =",
        runner_commit,
    )

    print(
        "runner_git_clean =",
        runner_clean,
    )

    if not runner_clean:
        raise RuntimeError(
            "Formal Dev10 evaluation requires "
            "a clean RL repository"
        )


    print()
    print(
        "===== NUMERICAL RUNTIME ====="
    )

    runtime = (
        base.configure_formal_training_runtime()
    )

    base.set_formal_training_seed(
        SEED
    )

    print(
        runtime
    )


    print()
    print(
        "===== FROZEN ACTOR ====="
    )

    (
        actor,
        _actor_artifact,
    ) = (
        base.load_frozen_actor()
    )

    print(
        "actor_sha256 =",
        base.EXPECTED_ACTOR_SHA256,
    )

    print(
        "actor_parameter_count =",
        base.count_trainable_parameters(
            actor
        ),
    )

    print(
        "critic_loaded = False"
    )

    print(
        "deterministic = True"
    )

    print(
        "epsilon = 0.0"
    )


    print()
    print(
        "===== DEV10 DATASET ====="
    )

    models = (
        load_dev10_models()
    )

    print(
        "training_protocol_dataset = Train49"
    )

    print(
        "operational_evaluation_dataset = Dev10"
    )

    print(
        "operational_models =",
        len(
            models
        ),
    )

    print(
        "train_overlap = []"
    )

    print(
        "models ="
    )

    for index, model in enumerate(
        models,
        start=1,
    ):
        print(
            f"{index:02d} "
            f"{model.model} "
            f"stratum="
            f"{model.complexity_stratum}"
        )


    print()
    print(
        "===== BUILD RUN MANIFEST ====="
    )

    candidate_manifest = (
        build_dev10_run_manifest(
            runtime=runtime,
            models=models,
            runner_commit=runner_commit,
        )
    )

    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        base.load_or_create_run_manifest(
            run_root=run_root,
            candidate=candidate_manifest,
        )
    )

    print(
        "run_manifest =",
        manifest_path,
    )

    print(
        "run_manifest_sha256 =",
        base.sha256_file(
            manifest_path
        ),
    )


    if args.preflight_only:
        print()
        print(
            "===== DEV10 PREFLIGHT PASS ====="
        )

        print(
            "No C++ evaluation episode was started."
        )

        return


    result_root = (
        run_root
        /
        "results"
    )

    result_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    for index, model in enumerate(
        models,
        start=1,
    ):
        result_path = (
            result_root
            /
            f"{model.model}.json"
        )

        print()
        print(
            "================================================"
        )

        print(
            f"MODEL {index:02d}/10: "
            f"{model.model}"
        )

        if result_path.is_file():
            existing = json.loads(
                result_path.read_text(
                    encoding="utf-8"
                )
            )

            base.validate_existing_result(
                result=existing,
                model=model,
                runner_commit=runner_commit,
            )

            print(
                "SKIP: durable result already exists; "
                f"outcome={existing['outcome']}"
            )

            continue


        preflight = (
            base.wait_for_formal_resource_rearm(
                emit_logs=True
            )
        )

        print(
            "resource_preflight =",
            preflight,
        )


        (
            attempt_directory,
            attempt_index,
        ) = (
            base.next_attempt_directory(
                run_root,
                model.model,
            )
        )

        print(
            "attempt =",
            attempt_index,
        )

        print(
            "attempt_directory =",
            attempt_directory,
        )


        try:
            result = (
                base.run_one_model(
                    actor=actor,
                    model=model,
                    runner_commit=runner_commit,
                    attempt_index=attempt_index,
                    attempt_directory=attempt_directory,
                )
            )

        except Exception as exc:
            fatal_record = {
                "schema_version":
                    "loopycuts_dev10_fatal_attempt_v1",

                "model":
                    model.model,

                "attempt_index":
                    attempt_index,

                "runner_git_commit":
                    runner_commit,

                "exception_type":
                    type(
                        exc
                    ).__name__,

                "exception_message":
                    str(
                        exc
                    ),

                "traceback":
                    traceback.format_exc(),

                "geometry_files":
                    base.geometry_manifest(
                        attempt_directory
                        /
                        "geometry"
                    ),
            }

            base.write_once_json(
                attempt_directory
                /
                "fatal_error.json",
                fatal_record,
            )

            raise


        result[
            "resource_preflight"
        ] = preflight

        base.write_once_json(
            result_path,
            result,
        )

        print(
            "outcome =",
            result[
                "outcome"
            ],
        )

        print(
            "steps =",
            len(
                result[
                    "action_sequence"
                ]
            ),
        )

        if (
            result[
                "quality"
            ]
            is not None
        ):
            print(
                "D_C =",
                result[
                    "quality"
                ][
                    "d_c"
                ],
            )

            print(
                "q_fidelity =",
                result[
                    "quality"
                ][
                    "q_fidelity"
                ],
            )

            print(
                "utility =",
                result[
                    "utility"
                ],
            )

        print(
            "geometry_file_count =",
            len(
                result[
                    "geometry_files"
                ]
            ),
        )

        summary = (
            write_summary(
                run_root=run_root,
                models=models,
                runner_commit=runner_commit,
            )
        )

        print(
            "completed_model_records =",
            summary[
                "completed_model_records"
            ],
        )


    summary = (
        write_summary(
            run_root=run_root,
            models=models,
            runner_commit=runner_commit,
        )
    )

    if (
        summary[
            "completed_model_records"
        ]
        != 10
    ):
        raise RuntimeError(
            "Dev10 evaluation ended without "
            "10 durable model results"
        )

    if (
        summary[
            "complete"
        ]
        is not True
    ):
        raise RuntimeError(
            "Dev10 summary completeness mismatch"
        )


    run_complete = {
        "schema_version":
            RUN_COMPLETE_SCHEMA_VERSION,

        "evaluator_version":
            EVALUATOR_VERSION,

        "seed":
            SEED,

        "runner_git_commit":
            runner_commit,

        "run_manifest_sha256":
            base.sha256_file(
                manifest_path
            ),

        "summary":
            summary,
    }

    run_complete_path = (
        run_root
        /
        "RUN_COMPLETE.json"
    )

    if run_complete_path.is_file():
        existing = json.loads(
            run_complete_path.read_text(
                encoding="utf-8"
            )
        )

        if existing != run_complete:
            raise RuntimeError(
                "Existing RUN_COMPLETE.json does "
                "not match current complete run"
            )

    else:
        base.write_once_json(
            run_complete_path,
            run_complete,
        )


    print()
    print(
        "===== DEV10 EVALUATION COMPLETE ====="
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "RUN_COMPLETE =",
        run_complete_path,
    )


if __name__ == "__main__":
    main()
