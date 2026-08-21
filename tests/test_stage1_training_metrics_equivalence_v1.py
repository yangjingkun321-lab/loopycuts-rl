from __future__ import annotations

import sys

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
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


from training.formal_checkpoint_v1 import (
    build_formal_checkpoint_payload,
)

from training.formal_training_v1 import (
    prepare_formal_training_core,
    run_formal_stage1,
)

from training.protocol_v1 import (
    PAPER_BATCH_SIZE,
    PROJECT_STAGE1_GRADIENT_STEPS,
    PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS,
)

from training.training_metrics_v1 import (
    TRAINING_METRICS_FILENAME,
    TrainingMetricsWriterV1,
)


def assert_deep_equal(
    left,
    right,
    *,
    path="root",
):
    if torch.is_tensor(left):
        assert torch.is_tensor(
            right
        ), f"{path}: right is not Tensor"

        assert (
            left.dtype
            ==
            right.dtype
        ), f"{path}: Tensor dtype differs"

        assert (
            tuple(left.shape)
            ==
            tuple(right.shape)
        ), f"{path}: Tensor shape differs"

        assert torch.equal(
            left,
            right,
        ), f"{path}: Tensor contents differ"

        return


    if isinstance(
        left,
        np.ndarray,
    ):
        assert isinstance(
            right,
            np.ndarray,
        ), f"{path}: right is not ndarray"

        assert (
            left.dtype
            ==
            right.dtype
        ), f"{path}: ndarray dtype differs"

        assert (
            left.shape
            ==
            right.shape
        ), f"{path}: ndarray shape differs"

        assert np.array_equal(
            left,
            right,
        ), f"{path}: ndarray contents differ"

        return


    if isinstance(
        left,
        dict,
    ):
        assert isinstance(
            right,
            dict,
        ), f"{path}: right is not dict"

        assert (
            set(left.keys())
            ==
            set(right.keys())
        ), f"{path}: dict keys differ"

        for key in left:
            assert_deep_equal(
                left[key],
                right[key],
                path=f"{path}.{key}",
            )

        return


    if isinstance(
        left,
        tuple,
    ):
        assert isinstance(
            right,
            tuple,
        ), f"{path}: tuple type differs"

        assert (
            len(left)
            ==
            len(right)
        ), f"{path}: tuple length differs"

        for index, (
            lvalue,
            rvalue,
        ) in enumerate(
            zip(
                left,
                right,
            )
        ):
            assert_deep_equal(
                lvalue,
                rvalue,
                path=f"{path}[{index}]",
            )

        return


    if isinstance(
        left,
        list,
    ):
        assert isinstance(
            right,
            list,
        ), f"{path}: list type differs"

        assert (
            len(left)
            ==
            len(right)
        ), f"{path}: list length differs"

        for index, (
            lvalue,
            rvalue,
        ) in enumerate(
            zip(
                left,
                right,
            )
        ):
            assert_deep_equal(
                lvalue,
                rvalue,
                path=f"{path}[{index}]",
            )

        return


    if isinstance(
        left,
        np.generic,
    ):
        left = left.item()

    if isinstance(
        right,
        np.generic,
    ):
        right = right.item()


    assert (
        left
        ==
        right
    ), (
        f"{path}: "
        f"{left!r} != {right!r}"
    )


def execute_stage1(
    *,
    metrics_writer=None,
):
    core = prepare_formal_training_core(
        seed=42
    )

    result = run_formal_stage1(
        core,
        metrics_writer=
            metrics_writer,
    )

    assert (
        result["gradient_updates"]
        ==
        PROJECT_STAGE1_GRADIENT_STEPS
    )

    assert (
        result[
            "sampled_demo_transitions"
        ]
        ==
        PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS
    )

    checkpoint_payload = (
        build_formal_checkpoint_payload(
            core,
            None,
        )
    )

    return (
        result,
        checkpoint_payload,
    )


def main():
    print(
        "Running Stage-I baseline "
        "with LOGGER OFF..."
    )

    (
        off_result,
        off_payload,
    ) = execute_stage1(
        metrics_writer=None
    )


    with TemporaryDirectory() as tmp:
        metrics_path = (
            Path(tmp)
            /
            TRAINING_METRICS_FILENAME
        )

        writer = TrainingMetricsWriterV1(
            path=
                metrics_path
        )

        print(
            "Running Stage-I with "
            "LOGGER ON..."
        )

        (
            on_result,
            on_payload,
        ) = execute_stage1(
            metrics_writer=
                writer
        )


        # --------------------------------------------------------
        # Exactly one metric row per real Stage-I update.
        # --------------------------------------------------------

        assert (
            writer.record_count()
            ==
            PROJECT_STAGE1_GRADIENT_STEPS
        )

        lines = (
            metrics_path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )

        assert (
            len(lines)
            ==
            PROJECT_STAGE1_GRADIENT_STEPS
        )


        # --------------------------------------------------------
        # Visible Stage-I result must be identical.
        # elapsed_seconds is intentionally excluded because wall
        # clock time is not part of training state.
        # --------------------------------------------------------

        assert (
            on_result[
                "gradient_updates"
            ]
            ==
            off_result[
                "gradient_updates"
            ]
        )

        assert (
            on_result[
                "sampled_demo_transitions"
            ]
            ==
            off_result[
                "sampled_demo_transitions"
            ]
        )

        assert (
            on_result[
                "alpha_after_stage1"
            ]
            ==
            off_result[
                "alpha_after_stage1"
            ]
        )

        assert_deep_equal(
            on_result[
                "final_training_stats"
            ],
            off_result[
                "final_training_stats"
            ],
            path=
                "final_training_stats",
        )


        # --------------------------------------------------------
        # Strong bitwise-equivalence test.
        #
        # The formal checkpoint payload covers:
        #
        #   actor
        #   critic1 / critic2
        #   target critics
        #   optimizer states
        #   auto-alpha
        #   alpha optimizer
        #   D_demo snapshot
        #   policy exploration RNG
        #   Python / NumPy / Torch RNG
        #   formal counters
        #   protocol/software/provenance
        #
        # Logger ON must produce the exact same payload.
        # --------------------------------------------------------

        assert_deep_equal(
            on_payload,
            off_payload,
            path=
                "formal_checkpoint_payload",
        )


        # Formal exposure sanity check.
        assert (
            PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS
            ==
            PROJECT_STAGE1_GRADIENT_STEPS
            *
            PAPER_BATCH_SIZE
        )


    print(
        "PASS: Stage-I Training Metrics V1 "
        "LOGGER OFF/ON produces identical "
        "formal checkpoint state"
    )

    print(
        "PASS: Stage-I metrics rows =",
        PROJECT_STAGE1_GRADIENT_STEPS,
    )


if __name__ == "__main__":
    main()
