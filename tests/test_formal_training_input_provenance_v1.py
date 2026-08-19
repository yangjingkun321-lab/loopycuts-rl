from __future__ import annotations

import json
import sys
import tempfile
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


from training.formal_training_input_provenance_v1 import (
    FormalTrainingInputProvenanceError,
    assert_formal_training_input_provenance,
)

from training.protocol_v1 import (
    PROJECT_FORMAL_TRAINING_TORCH_NUM_THREADS,
    PROJECT_FORMAL_TRAINING_TORCH_NUM_INTEROP_THREADS,
    PROJECT_FORMAL_TRAINING_TORCH_DETERMINISTIC_ALGORITHMS,
)


PROVENANCE = Path(
    "data/manifests/"
    "formal_training_input_provenance_v1.json"
)

EXECUTABLE = Path(
    "/home/yjk/codes/LoopyCuts/"
    "volumetric_cutter/"
    "volumetric_cutter"
)

DATASET = Path(
    "data/manifests/"
    "dataset_split_v2.csv"
)

DEMO_QUALITY = Path(
    "data/manifests/"
    "demo_quality_v1.csv"
)

RAW_DEMO_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "demonstrations_v1/raw/train"
)


EXPECTED_TRAIN49_AGGREGATE = (
    "a1e68312f05457e2f3ecb92e7b59fa93"
    "facbc57850833b4f8931a7143f55d42d"
)


def configure_runtime():
    if (
        torch.get_num_threads()
        !=
        PROJECT_FORMAL_TRAINING_TORCH_NUM_THREADS
    ):
        torch.set_num_threads(
            PROJECT_FORMAL_TRAINING_TORCH_NUM_THREADS
        )

    if (
        torch.get_num_interop_threads()
        !=
        PROJECT_FORMAL_TRAINING_TORCH_NUM_INTEROP_THREADS
    ):
        torch.set_num_interop_threads(
            PROJECT_FORMAL_TRAINING_TORCH_NUM_INTEROP_THREADS
        )

    torch.use_deterministic_algorithms(
        PROJECT_FORMAL_TRAINING_TORCH_DETERMINISTIC_ALGORITHMS
    )


def main():
    configure_runtime()

    result = (
        assert_formal_training_input_provenance(
            provenance_path=
                PROVENANCE,

            executable=
                EXECUTABLE,

            dataset_manifest=
                DATASET,

            demo_quality_manifest=
                DEMO_QUALITY,

            raw_demo_root=
                RAW_DEMO_ROOT,
        )
    )

    assert (
        result[
            "train49_models"
        ]
        ==
        49
    )

    assert (
        result[
            "train49_aggregate_sha256"
        ]
        ==
        EXPECTED_TRAIN49_AGGREGATE
    )

    assert (
        result[
            "selected_bc_weight"
        ]
        ==
        3.0
    )


    # ------------------------------------------------------------
    # Negative test:
    # changing the frozen Train49 aggregate must be rejected.
    # ------------------------------------------------------------

    payload = json.loads(
        PROVENANCE.read_text(
            encoding="utf-8"
        )
    )

    payload[
        "train49"
    ][
        "aggregate_sha256"
    ] = (
        "0"
        *
        64
    )

    with tempfile.TemporaryDirectory() as tmp:
        bad = (
            Path(tmp)
            /
            "bad_provenance.json"
        )

        bad.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            +
            "\n",
            encoding="utf-8",
        )

        try:
            assert_formal_training_input_provenance(
                provenance_path=
                    bad,

                executable=
                    EXECUTABLE,

                dataset_manifest=
                    DATASET,

                demo_quality_manifest=
                    DEMO_QUALITY,

                raw_demo_root=
                    RAW_DEMO_ROOT,
            )

        except FormalTrainingInputProvenanceError:
            pass

        else:
            raise AssertionError(
                "Train49 provenance mismatch was not rejected"
            )


    print(
        "formal provenance SHA256 :",
        result[
            "manifest_sha256"
        ],
    )

    print(
        "Train49 models           :",
        result[
            "train49_models"
        ],
    )

    print(
        "Train49 aggregate SHA256 :",
        result[
            "train49_aggregate_sha256"
        ],
    )

    print(
        "selected lambda_BC       :",
        result[
            "selected_bc_weight"
        ],
    )

    print(
        "PASS: formal-training provenance matches all frozen inputs"
    )

    print(
        "PASS: Train49 provenance mismatch is rejected"
    )


if __name__ == "__main__":
    main()
