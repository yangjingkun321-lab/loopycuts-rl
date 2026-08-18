from __future__ import annotations

import copy
import json
import sys
import tempfile
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


from training.bc_weight_input_provenance_v1 import (
    INPUT_PROVENANCE_SCHEMA_VERSION,
    InputProvenanceError,
    assert_frozen_input_provenance,
    load_input_provenance,
)


PROVENANCE_PATH = (
    PROJECT_ROOT
    /
    "data/manifests/"
    "bc_weight_calibration_input_provenance_v1.json"
)

EXECUTABLE = Path(
    "/home/yjk/codes/LoopyCuts/"
    "volumetric_cutter/volumetric_cutter"
)

DATASET_MANIFEST = (
    PROJECT_ROOT
    /
    "data/manifests/"
    "dataset_split_v2.csv"
)

DEMO_QUALITY = (
    PROJECT_ROOT
    /
    "data/manifests/"
    "demo_quality_v1.csv"
)

RAW_DEMO_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "demonstrations_v1/raw/train"
)


def verify(
    provenance_path: Path,
):
    return assert_frozen_input_provenance(
        provenance_path=
            provenance_path,

        executable=
            EXECUTABLE,

        dataset_manifest=
            DATASET_MANIFEST,

        demo_quality_manifest=
            DEMO_QUALITY,

        raw_demo_root=
            RAW_DEMO_ROOT,
    )


def expect_rejection(
    *,
    payload,
    label,
):
    with tempfile.TemporaryDirectory() as tmp:
        path = (
            Path(tmp)
            /
            "mutated_provenance.json"
        )

        path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            +
            "\n",
            encoding="utf-8",
        )

        try:
            verify(
                path
            )

        except InputProvenanceError:
            return

        raise AssertionError(
            f"{label} was unexpectedly accepted"
        )


def main():
    payload = (
        load_input_provenance(
            PROVENANCE_PATH
        )
    )

    assert (
        payload[
            "schema_version"
        ]
        ==
        INPUT_PROVENANCE_SCHEMA_VERSION
    )


    # ============================================================
    # Current real formal environment must match exactly.
    # ============================================================

    observed = verify(
        PROVENANCE_PATH
    )

    assert (
        observed[
            "manifest_sha256"
        ]
        ==
        (
            "b97d70f52ebbc0bf861042e6adfe29b4"
            "305952f459a59cfffa524d154f82d01d"
        )
    )

    assert (
        observed[
            "loopycuts_executable_sha256"
        ]
        ==
        (
            "920ff358d7840dfa13dc53cc0b603721"
            "347cc77e8951afb14d322de14f84a565"
        )
    )

    assert (
        observed[
            "formal_d_demo_aggregate_sha256"
        ]
        ==
        (
            "e4de643107462712e63b55ea5839fa92"
            "293d8256b255d7f5fb03ec9619e3c383"
        )
    )


    # ============================================================
    # Frozen CPU numerical runtime.
    # ============================================================

    assert (
        observed[
            "torch_num_threads"
        ]
        ==
        8
    )

    assert (
        observed[
            "torch_num_interop_threads"
        ]
        ==
        8
    )

    assert (
        observed[
            "torch_deterministic_algorithms"
        ]
        is False
    )

    assert (
        observed[
            "cpu_repeatability_audit_sha256"
        ]
        ==
        (
            "a3c367167cb8f1f710c4eaea5edf23f"
            "dee50d401e5ca083df5fc82f4cf7f83fe"
        )
    )


    # ============================================================
    # Wrong frozen thread policy must fail.
    # ============================================================

    mutated = copy.deepcopy(
        payload
    )

    mutated[
        "runtime"
    ][
        "thread_policy"
    ][
        "torch_num_threads"
    ] = 1

    expect_rejection(
        payload=
            mutated,

        label=
            "torch thread-policy mismatch",
    )


    # ============================================================
    # Wrong repeatability-audit fingerprint must fail.
    # ============================================================

    mutated = copy.deepcopy(
        payload
    )

    mutated[
        "runtime"
    ][
        "cpu_repeatability_audit"
    ][
        "artifact_sha256"
    ] = (
        "3" * 64
    )

    expect_rejection(
        payload=
            mutated,

        label=
            "CPU repeatability audit SHA256 mismatch",
    )


    # ============================================================
    # Executable mismatch must fail.
    # ============================================================

    mutated = copy.deepcopy(
        payload
    )

    mutated[
        "loopycuts"
    ][
        "executable"
    ][
        "sha256"
    ] = (
        "0" * 64
    )

    expect_rejection(
        payload=
            mutated,

        label=
            "volumetric_cutter SHA256 mismatch",
    )


    # ============================================================
    # Engineering5 content mismatch must fail.
    # ============================================================

    mutated = copy.deepcopy(
        payload
    )

    mutated[
        "engineering_calibration"
    ][
        "inputs"
    ][
        "bimba"
    ][
        "mesh_sha256"
    ] = (
        "1" * 64
    )

    expect_rejection(
        payload=
            mutated,

        label=
            "Engineering5 mesh SHA256 mismatch",
    )


    # ============================================================
    # Formal D_demo aggregate mismatch must fail.
    # ============================================================

    mutated = copy.deepcopy(
        payload
    )

    mutated[
        "formal_d_demo"
    ][
        "aggregate_sha256"
    ] = (
        "2" * 64
    )

    expect_rejection(
        payload=
            mutated,

        label=
            "formal D_demo aggregate mismatch",
    )


    print(
        "PASS: frozen input provenance matches "
        "the current formal environment"
    )

    print(
        "PASS: executable provenance mismatch is rejected"
    )

    print(
        "PASS: Engineering5 provenance mismatch is rejected"
    )

    print(
        "PASS: formal D_demo provenance mismatch is rejected"
    )


if __name__ == "__main__":
    main()
