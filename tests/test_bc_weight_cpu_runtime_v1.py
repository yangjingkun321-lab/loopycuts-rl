from __future__ import annotations

import sys
from pathlib import Path

import torch


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
    sha256_file,
)

from training.protocol_v1 import (
    PROJECT_BC_WEIGHT_CALIBRATION_DEVICE,
    PROJECT_BC_WEIGHT_CALIBRATION_TORCH_NUM_THREADS,
    PROJECT_BC_WEIGHT_CALIBRATION_TORCH_NUM_INTEROP_THREADS,
    PROJECT_BC_WEIGHT_CALIBRATION_TORCH_DETERMINISTIC_ALGORITHMS,
    PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_VERSION,
    PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_SEED,
    PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_BC_WEIGHT,
    PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_UPDATES,
    PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_BASE_COMMIT,
    PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_SHA256,
    PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_RESULT,
)

from training.run_bc_weight_calibration_v1 import (
    assert_protocol,
    configure_calibration_cpu_runtime,
)


AUDIT_PATH = (
    PROJECT_ROOT
    /
    "data/audits/"
    "bc_weight_stage1_cpu_repeatability_v1.json"
)


def main():
    assert_protocol()

    runtime = (
        configure_calibration_cpu_runtime()
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_DEVICE
        ==
        "cpu"
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_TORCH_NUM_THREADS
        ==
        8
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_TORCH_NUM_INTEROP_THREADS
        ==
        8
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_TORCH_DETERMINISTIC_ALGORITHMS
        is False
    )

    assert runtime == {
        "device":
            "cpu",

        "torch_num_threads":
            8,

        "torch_num_interop_threads":
            8,

        "torch_deterministic_algorithms":
            False,
    }

    assert torch.get_num_threads() == 8

    assert (
        torch.get_num_interop_threads()
        ==
        8
    )

    assert (
        torch.are_deterministic_algorithms_enabled()
        is False
    )


    # Repeatability evidence.
    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_VERSION
        ==
        "stage1_cpu_repeatability_probe_v1"
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_SEED
        ==
        42
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_BC_WEIGHT
        ==
        2.0
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_UPDATES
        ==
        20
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_BASE_COMMIT
        ==
        "77f393af7ae2a0a07b86b86c05c2e03f51b3210b"
    )

    assert AUDIT_PATH.is_file()

    observed_sha = (
        sha256_file(
            AUDIT_PATH
        )
    )

    assert (
        observed_sha
        ==
        PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_AUDIT_SHA256
        ==
        (
            "a3c367167cb8f1f710c4eaea5edf23f"
            "dee50d401e5ca083df5fc82f4cf7f83fe"
        )
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_CPU_REPEATABILITY_RESULT
        ==
        (
            "BITWISE_IDENTICAL_ACROSS_TWO_"
            "INDEPENDENT_PROCESSES"
        )
    )


    print(
        "PASS: BC calibration CPU runtime is frozen "
        "to 8/8 threads with deterministic flag False"
    )

    print(
        "PASS: frozen runtime is tied to the "
        "bitwise-repeatability audit artifact"
    )


if __name__ == "__main__":
    main()
