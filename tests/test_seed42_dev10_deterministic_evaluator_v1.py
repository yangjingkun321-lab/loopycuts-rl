from __future__ import annotations

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


from evaluation import (
    run_seed42_dev10_deterministic_v1
    as dev,
)

import evaluation.run_seed42_train48_deterministic_v1 as base


EXPECTED_NAMES = (
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


def test_dev10_frozen_input_identities():
    assert (
        base.sha256_file(
            dev.DATASET_MANIFEST
        )
        ==
        dev.EXPECTED_DATASET_MANIFEST_SHA256
    )

    assert (
        base.sha256_file(
            dev.QUALITY_REF_MANIFEST
        )
        ==
        dev.EXPECTED_QUALITY_REF_MANIFEST_SHA256
    )

    assert (
        base.sha256_file(
            dev.QUALITY_REF_PROVENANCE
        )
        ==
        dev.EXPECTED_QUALITY_REF_PROVENANCE_SHA256
    )

    assert (
        base.sha256_file(
            dev.QUALITY_REF_SHA256SUMS
        )
        ==
        dev.EXPECTED_QUALITY_REF_SHA256SUMS_IDENTITY
    )


def test_dev10_model_selection_is_exact_and_disjoint():
    models = (
        dev.load_dev10_models()
    )

    assert len(
        models
    ) == 10

    assert tuple(
        model.model
        for model in models
    ) == EXPECTED_NAMES

    assert tuple(
        model.complexity_stratum
        for model in models
    ) == tuple(
        range(10)
    )

    assert len(
        {
            model.model
            for model in models
        }
    ) == 10


def test_dev10_model_inputs_and_refs_exist():
    models = (
        dev.load_dev10_models()
    )

    for model in models:
        assert (
            model.mesh_file
            .is_file()
        )

        assert (
            model.loop_file
            .is_file()
        )

        assert (
            model.quality_ref_file
            .is_file()
        )

        assert (
            model.quality_ref_file
            .parent
            ==
            (
                dev.QUALITY_REF_SET_ROOT
                /
                "refs"
            )
        )


def test_dev10_base_configuration():
    dev.configure_base_globals()

    assert (
        base.EVALUATOR_VERSION
        ==
        dev.EVALUATOR_VERSION
    )

    assert (
        base.RESULT_SCHEMA_VERSION
        ==
        dev.RESULT_SCHEMA_VERSION
    )

    assert (
        base.OPERATIONAL_DATASET_NAME
        ==
        "Dev10"
    )

    assert (
        base.EXCLUDED_MODEL
        is None
    )

    assert (
        base.QUALITY_REF_SET_ROOT
        ==
        dev.QUALITY_REF_SET_ROOT
    )

    assert (
        base.EXPECTED_QUALITY_REF_SHA256SUMS_IDENTITY
        ==
        dev.EXPECTED_QUALITY_REF_SHA256SUMS_IDENTITY
    )


def test_dev10_summary_semantics():
    fake_records = [
        {
            "outcome":
                "FULL_HEX",

            "quality": {
                "d_c": 1.0,
                "q_fidelity": 0.9,
                "total_polys": 100,
                "nonhex": 0,
            },

            "utility":
                0.9,
        },

        {
            "outcome":
                "NON_FULL_HEX",

            "quality": {
                "d_c": 0.8,
                "q_fidelity": 0.95,
                "total_polys": 100,
                "nonhex": 2,
            },

            "utility":
                0.76,
        },
    ]

    dev.configure_base_globals()

    summary = (
        dev.summarize_dev10(
            fake_records
        )
    )

    assert (
        summary[
            "schema_version"
        ]
        ==
        dev.SUMMARY_SCHEMA_VERSION
    )

    assert (
        summary[
            "expected_models"
        ]
        ==
        10
    )

    assert (
        summary[
            "completed_model_records"
        ]
        ==
        2
    )

    assert (
        summary[
            "complete"
        ]
        is False
    )

    assert (
        summary[
            "full_hex_count"
        ]
        ==
        1
    )

    assert (
        summary[
            "full_hex_rate_over_dev10"
        ]
        ==
        0.1
    )

    assert (
        "full_hex_rate_over_operational_train48"
        not in summary
    )


def test_dev10_direct_cli_help():
    script = (
        PROJECT_ROOT
        /
        "evaluation"
        /
        "run_seed42_dev10_deterministic_v1.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    assert (
        completed.returncode
        ==
        0
    ), completed.stdout

    assert (
        "--preflight-only"
        in completed.stdout
    )


def main():
    test_dev10_direct_cli_help()

    print(
        "PASS: direct Dev10 CLI import/--help"
    )

    test_dev10_frozen_input_identities()

    print(
        "PASS: frozen Dev10 input identities"
    )

    test_dev10_model_selection_is_exact_and_disjoint()

    print(
        "PASS: exact Dev10 model selection"
    )

    test_dev10_model_inputs_and_refs_exist()

    print(
        "PASS: Dev10 mesh/loop/quality refs"
    )

    test_dev10_base_configuration()

    print(
        "PASS: Dev10 base evaluator configuration"
    )

    test_dev10_summary_semantics()

    print(
        "PASS: Dev10 summary semantics"
    )

    print()
    print(
        "===== DEV10 EVALUATOR TEST PASS ====="
    )


if __name__ == "__main__":
    main()
