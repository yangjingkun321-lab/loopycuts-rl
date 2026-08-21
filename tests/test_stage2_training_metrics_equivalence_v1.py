from __future__ import annotations

import sys

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from tianshou.data import (
    Batch,
)


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
    collect_formal_stage2_model_episode,
    enter_formal_stage2,
    prepare_formal_stage2_state,
    prepare_formal_training_core,
)

from training.protocol_v1 import (
    PROJECT_STAGE1_GRADIENT_STEPS,
    PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS,
)

from training.training_metrics_v1 import (
    TRAINING_METRICS_FILENAME,
    TrainingMetricsWriterV1,
)


RUNTIME_ONLY_PATHS = {
    (
        "formal_checkpoint_payload."
        "stage2.expo_replay.data."
        "info.step_result.step_time"
    ),
    (
        "formal_checkpoint_payload."
        "stage2.expo_replay.data."
        "info.transition_metrics.step_time"
    ),
}


def assert_deep_equal(
    left,
    right,
    *,
    path="root",
):
    if path in RUNTIME_ONLY_PATHS:
        return


    if isinstance(
        left,
        Batch,
    ):
        assert isinstance(
            right,
            Batch,
        ), f"{path}: right is not Batch"

        left_keys = set(
            left.keys()
        )

        right_keys = set(
            right.keys()
        )

        assert (
            left_keys
            ==
            right_keys
        ), f"{path}: Batch keys differ"

        for key in left_keys:
            assert_deep_equal(
                left[
                    key
                ],
                right[
                    key
                ],
                path=
                    f"{path}.{key}",
            )

        return


    if torch.is_tensor(left):
        assert torch.is_tensor(
            right
        ), f"{path}: right is not Tensor"

        assert left.dtype == right.dtype
        assert left.shape == right.shape

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
        )

        assert left.dtype == right.dtype
        assert left.shape == right.shape

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
        )

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
        )

        assert len(left) == len(right)

        for index, (
            lvalue,
            rvalue,
        ) in enumerate(
            zip(left, right)
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
        )

        assert len(left) == len(right)

        for index, (
            lvalue,
            rvalue,
        ) in enumerate(
            zip(left, right)
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


    assert left == right, (
        f"{path}: {left!r} != {right!r}"
    )


def prepare_stage2():
    core = prepare_formal_training_core(
        seed=42
    )

    # Infrastructure equivalence test:
    # Stage-I itself was already validated separately.
    core.stage1_updates_completed = (
        PROJECT_STAGE1_GRADIENT_STEPS
    )

    core.stage1_sampled_demo_transitions = (
        PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS
    )

    enter_formal_stage2(
        core
    )

    state = prepare_formal_stage2_state(
        core
    )

    plate3 = next(
        model
        for model in state.models
        if model.model == "Plate3"
    )

    return (
        core,
        state,
        plate3,
    )


def execute_plate3(
    *,
    metrics_writer=None,
):
    (
        core,
        state,
        plate3,
    ) = prepare_stage2()

    record = (
        collect_formal_stage2_model_episode(
            core,
            state,
            model=
                plate3,
            metrics_writer=
                metrics_writer,
        )
    )

    assert record["model"] == "Plate3"
    assert record["steps"] == 2
    assert record["actions"] == [1, 0]

    assert (
        record["finalization_outcome"]
        ==
        "FULL_HEX"
    )

    assert (
        record["gradient_updates"]
        ==
        2
    )

    assert (
        state.total_environment_steps
        ==
        2
    )

    assert (
        state.total_gradient_updates
        ==
        2
    )

    payload = (
        build_formal_checkpoint_payload(
            core,
            state,
        )
    )

    return (
        record,
        payload,
    )


def main():
    print(
        "Running Plate3 Stage-II "
        "with LOGGER OFF..."
    )

    (
        off_record,
        off_payload,
    ) = execute_plate3(
        metrics_writer=None
    )


    with TemporaryDirectory() as tmp:
        metrics_path = (
            Path(tmp)
            /
            TRAINING_METRICS_FILENAME
        )

        writer = (
            TrainingMetricsWriterV1(
                path=
                    metrics_path
            )
        )

        print(
            "Running Plate3 Stage-II "
            "with LOGGER ON..."
        )

        (
            on_record,
            on_payload,
        ) = execute_plate3(
            metrics_writer=
                writer
        )


        # Plate3 produces exactly two real
        # Stage-II gradient updates.
        assert writer.record_count() == 2

        lines = (
            metrics_path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )

        assert len(lines) == 2


        # Episode result must remain identical.
        assert_deep_equal(
            on_record,
            off_record,
            path=
                "episode_record",
        )


        # Strong equivalence:
        # model/critics/targets/optimizers/alpha/
        # D_demo/D_expo/RNG/history/counters.
        assert_deep_equal(
            on_payload,
            off_payload,
            path=
                "formal_checkpoint_payload",
        )


    print(
        "PASS: Stage-II Training Metrics V1 "
        "LOGGER OFF/ON produces identical "
        "formal checkpoint state"
    )

    print(
        "PASS: Plate3 metrics rows = 2"
    )


if __name__ == "__main__":
    main()
