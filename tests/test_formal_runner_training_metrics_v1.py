from __future__ import annotations

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


import training.run_formal_training_v1 as runner_module


from training.formal_run_artifacts_v1 import (
    EVENT_LOG_FILENAME,
    create_formal_run_artifacts,
    read_formal_events,
    record_stage1_complete,
    record_stage2_episode,
)

from training.formal_training_v1 import (
    collect_formal_stage2_model_episode,
    enter_formal_stage2,
    prepare_formal_stage2_state,
    prepare_formal_training_core,
)

from training.protocol_v1 import (
    PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS,
    PROJECT_STAGE1_GRADIENT_STEPS,
    PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS,
)

from training.run_formal_training_v1 import (
    _resume_existing_checkpoint,
    _run_stage2_loop,

    formal_training_metrics_path,
    latest_checkpoint_path,

    save_and_record_formal_checkpoint,
    validate_complete_formal_training_metrics,
)

from training.training_metrics_v1 import (
    TrainingMetricsError,
    TrainingMetricsWriterV1,
)


def prepare_zero_stage2():
    core = prepare_formal_training_core(
        seed=42
    )

    # Same lightweight convention used by the existing formal runner
    # regression tests: Stage-I mathematical execution has already
    # been independently tested, so this orchestration test freezes
    # only its formal counters.
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

    return (
        core,
        state,
    )


def append_stage1_metrics(
    writer,
):
    for update_index in range(
        1,
        PROJECT_STAGE1_GRADIENT_STEPS + 1,
    ):
        writer.append(
            seed=42,
            stage="STAGE_I",

            gradient_update=
                update_index,

            sampled_demo_transitions=
                update_index * 64,

            stats={
                "actor_loss":
                    float(
                        -update_index
                    ),

                "bc_loss":
                    0.0,

                "alpha":
                    1.0,
            },
        )


def synthetic_stage1_result(
    core,
):
    return {
        "alpha_after_stage1":
            float(
                core.auto_alpha.value
            ),

        "elapsed_seconds":
            0.0,

        "final_training_stats": {
            "actor_loss":
                0.0,

            "bc_loss":
                0.0,

            "alpha":
                float(
                    core.auto_alpha.value
                ),
        },
    }


def plate3(
    state,
):
    return next(
        model
        for model in state.models
        if model.model == "Plate3"
    )


def make_plate3_runner(
    writer,
):
    def run(
        core,
        state,
    ):
        return (
            collect_formal_stage2_model_episode(
                core,
                state,

                model=
                    plate3(
                        state
                    ),

                metrics_writer=
                    writer,
            )
        )

    return run


def bootstrap_zero_checkpoint(
    run_directory: Path,
):
    (
        core,
        state,
    ) = prepare_zero_stage2()

    create_formal_run_artifacts(
        run_directory=
            run_directory,

        core=
            core,

        require_clean_git=
            False,
    )

    writer = TrainingMetricsWriterV1(
        path=
            formal_training_metrics_path(
                run_directory
            )
    )

    append_stage1_metrics(
        writer
    )

    record_stage1_complete(
        run_directory=
            run_directory,

        core=
            core,

        stage1_result=
            synthetic_stage1_result(
                core
            ),
    )

    checkpoint = (
        latest_checkpoint_path(
            run_directory
        )
    )

    save_and_record_formal_checkpoint(
        run_directory=
            run_directory,

        checkpoint_path=
            checkpoint,

        core=
            core,

        stage2_state=
            state,

        require_clean_git=
            False,

        metrics_writer=
            writer,
    )

    assert checkpoint.is_file()

    assert (
        state.total_environment_steps
        ==
        0
    )

    assert (
        state.total_gradient_updates
        ==
        0
    )

    return (
        core,
        state,
        writer,
        checkpoint,
    )


def test_checkpoint_order():
    """
    Prove the orchestration order itself:

        metrics prefix
        -> metrics fsync
        -> checkpoint save
        -> checkpoint event
    """

    (
        core,
        state,
    ) = prepare_zero_stage2()

    order = []


    class DurabilityProbe:
        def assert_complete_prefix(
            self,
            *,
            seed,
            stage,
            gradient_updates,
        ):
            order.append(
                (
                    "prefix",
                    int(seed),
                    str(stage),
                    int(gradient_updates),
                )
            )

            return {
                "seed":
                    int(seed),

                "stage":
                    str(stage),

                "gradient_updates":
                    int(gradient_updates),
            }


        def sync(
            self,
        ):
            order.append(
                "sync"
            )


    probe = DurabilityProbe()

    original_save = (
        runner_module.save_formal_checkpoint
    )

    original_record = (
        runner_module.record_checkpoint
    )


    def fake_save(
        *,
        checkpoint_path,
        core,
        stage2_state,
        require_clean_git,
    ):
        assert (
            "sync"
            in
            order
        )

        order.append(
            "save"
        )

        return {
            "path":
                str(
                    Path(
                        checkpoint_path
                    )
                    .resolve()
                ),

            "bytes":
                0,

            "sha256":
                "0" * 64,

            "stage":
                core.stage,

            "seed":
                int(
                    core.seed
                ),
        }


    def fake_record(
        **kwargs,
    ):
        assert (
            "save"
            in
            order
        )

        order.append(
            "event"
        )

        return {
            "event":
                {},
        }


    try:
        runner_module.save_formal_checkpoint = (
            fake_save
        )

        runner_module.record_checkpoint = (
            fake_record
        )

        runner_module.save_and_record_formal_checkpoint(
            run_directory=
                Path(
                    "/tmp/"
                    "loopycuts_metrics_order_probe"
                ),

            checkpoint_path=
                Path(
                    "/tmp/"
                    "loopycuts_metrics_order_probe/"
                    "latest.pt"
                ),

            core=
                core,

            stage2_state=
                state,

            require_clean_git=
                False,

            metrics_writer=
                probe,
        )

    finally:
        runner_module.save_formal_checkpoint = (
            original_save
        )

        runner_module.record_checkpoint = (
            original_record
        )


    assert order == [
        (
            "prefix",
            42,
            "STAGE_I",
            PROJECT_STAGE1_GRADIENT_STEPS,
        ),
        (
            "prefix",
            42,
            "STAGE_II",
            0,
        ),
        "sync",
        "save",
        "event",
    ], order


def test_incomplete_prefix_blocks_checkpoint(
    root: Path,
):
    run_directory = (
        root
        /
        "incomplete_prefix"
    )

    (
        core,
        state,
    ) = prepare_zero_stage2()

    create_formal_run_artifacts(
        run_directory=
            run_directory,

        core=
            core,

        require_clean_git=
            False,
    )

    writer = TrainingMetricsWriterV1(
        path=
            formal_training_metrics_path(
                run_directory
            )
    )

    # Deliberately omit Stage-I update 782.
    for update_index in range(
        1,
        PROJECT_STAGE1_GRADIENT_STEPS,
    ):
        writer.append(
            seed=42,
            stage="STAGE_I",

            gradient_update=
                update_index,

            stats={
                "actor_loss":
                    0.0,
            },
        )

    checkpoint = (
        latest_checkpoint_path(
            run_directory
        )
    )

    try:
        save_and_record_formal_checkpoint(
            run_directory=
                run_directory,

            checkpoint_path=
                checkpoint,

            core=
                core,

            stage2_state=
                state,

            require_clean_git=
                False,

            metrics_writer=
                writer,
        )

    except TrainingMetricsError:
        pass

    else:
        raise RuntimeError(
            "Checkpoint accepted an incomplete "
            "Training Metrics prefix"
        )

    assert (
        not
        checkpoint.exists()
    )


def test_valid_plate3_checkpoint(
    root: Path,
):
    run_directory = (
        root
        /
        "valid_plate3"
    )

    (
        core,
        state,
        writer,
        checkpoint,
    ) = bootstrap_zero_checkpoint(
        run_directory
    )

    record = (
        collect_formal_stage2_model_episode(
            core,
            state,

            model=
                plate3(
                    state
                ),

            metrics_writer=
                writer,
        )
    )

    assert (
        record[
            "steps"
        ]
        ==
        2
    )

    assert (
        record[
            "actions"
        ]
        ==
        [
            1,
            0,
        ]
    )

    assert (
        record[
            "finalization_outcome"
        ]
        ==
        "FULL_HEX"
    )

    record_stage2_episode(
        run_directory=
            run_directory,

        core=
            core,

        record=
            record,
    )

    result = (
        save_and_record_formal_checkpoint(
            run_directory=
                run_directory,

            checkpoint_path=
                checkpoint,

            core=
                core,

            stage2_state=
                state,

            require_clean_git=
                False,

            metrics_writer=
                writer,
        )
    )

    assert checkpoint.is_file()

    assert (
        result[
            "path"
        ]
        ==
        str(
            checkpoint.resolve()
        )
    )

    assert (
        writer.stage_record_count(
            seed=42,
            stage="STAGE_I",
        )
        ==
        PROJECT_STAGE1_GRADIENT_STEPS
    )

    assert (
        writer.stage_record_count(
            seed=42,
            stage="STAGE_II",
        )
        ==
        2
    )


def test_logged_ahead_metrics_replay(
    root: Path,
):
    """
    checkpoint@0
      -> episode executes
      -> event + metrics 1..2 durable
      -> no new checkpoint
      -> process restart
      -> checkpoint@0 reload
      -> deterministic replay
      -> same event and same metrics rows are reused
    """

    run_directory = (
        root
        /
        "ahead_replay"
    )

    (
        core,
        state,
        writer,
        checkpoint,
    ) = bootstrap_zero_checkpoint(
        run_directory
    )

    first = _run_stage2_loop(
        core=
            core,

        stage2_state=
            state,

        run_directory=
            run_directory,

        checkpoint_path=
            checkpoint,

        last_checkpoint_environment_steps=
            0,

        require_clean_git=
            False,

        metrics_writer=
            writer,

        # Plate3 has two transitions. Keep the
        # zero-transition checkpoint as latest.pt.
        checkpoint_interval_environment_steps=
            4,

        episode_runner=
            make_plate3_runner(
                writer
            ),

        max_new_episode_executions=
            1,

        resource_rearm_emit_logs=
            False,
    )

    assert (
        first[
            "latest_checkpoint_result"
        ]
        is None
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

    metrics_path = (
        formal_training_metrics_path(
            run_directory
        )
    )

    lines_before = (
        metrics_path
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    assert (
        len(
            lines_before
        )
        ==
        PROJECT_STAGE1_GRADIENT_STEPS
        +
        2
    )

    events_before = (
        read_formal_events(
            event_log_path=
                run_directory
                /
                EVENT_LOG_FILENAME
        )
    )

    assert [
        event[
            "event_type"
        ]
        for event in events_before
    ] == [
        "STAGE1_COMPLETE",
        "CHECKPOINT",
        "STAGE2_EPISODE",
    ]


    # Simulated restart. Opening the writer must preserve the complete
    # logged-ahead suffix.
    resumed_writer = (
        TrainingMetricsWriterV1(
            path=
                metrics_path
        )
    )

    (
        resumed_core,
        resumed_state,
        _,
        _,
    ) = _resume_existing_checkpoint(
        seed=42,

        run_directory=
            run_directory,

        checkpoint_path=
            checkpoint,

        require_clean_git=
            False,

        metrics_writer=
            resumed_writer,
    )

    assert resumed_state is not None

    assert (
        resumed_state.total_environment_steps
        ==
        0
    )

    replay = _run_stage2_loop(
        core=
            resumed_core,

        stage2_state=
            resumed_state,

        run_directory=
            run_directory,

        checkpoint_path=
            checkpoint,

        last_checkpoint_environment_steps=
            0,

        require_clean_git=
            False,

        metrics_writer=
            resumed_writer,

        checkpoint_interval_environment_steps=
            4,

        episode_runner=
            make_plate3_runner(
                resumed_writer
            ),

        max_new_episode_executions=
            1,

        resource_rearm_emit_logs=
            False,
    )

    assert (
        replay[
            "appended_episode_events"
        ]
        ==
        0
    )

    assert (
        replay[
            "replayed_episode_events"
        ]
        ==
        1
    )

    assert (
        resumed_state.total_environment_steps
        ==
        2
    )

    assert (
        resumed_state.total_gradient_updates
        ==
        2
    )

    lines_after = (
        metrics_path
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    assert (
        lines_after
        ==
        lines_before
    )

    events_after = (
        read_formal_events(
            event_log_path=
                run_directory
                /
                EVENT_LOG_FILENAME
        )
    )

    assert (
        events_after
        ==
        events_before
    )


def test_conflicting_logged_ahead_metrics_fatal(
    root: Path,
):
    """
    Logged-ahead telemetry is allowed only when deterministic replay
    produces EXACTLY the same row.
    """

    run_directory = (
        root
        /
        "ahead_conflict"
    )

    (
        core,
        state,
        writer,
        checkpoint,
    ) = bootstrap_zero_checkpoint(
        run_directory
    )

    first = _run_stage2_loop(
        core=
            core,

        stage2_state=
            state,

        run_directory=
            run_directory,

        checkpoint_path=
            checkpoint,

        last_checkpoint_environment_steps=
            0,

        require_clean_git=
            False,

        metrics_writer=
            writer,

        checkpoint_interval_environment_steps=
            4,

        episode_runner=
            make_plate3_runner(
                writer
            ),

        max_new_episode_executions=
            1,

        resource_rearm_emit_logs=
            False,
    )

    assert (
        first[
            "latest_checkpoint_result"
        ]
        is None
    )

    metrics_path = (
        formal_training_metrics_path(
            run_directory
        )
    )

    rows = [
        json.loads(
            line
        )
        for line in
        metrics_path
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    ]

    target_count = 0

    for row in rows:
        if (
            row[
                "stage"
            ]
            ==
            "STAGE_II"
            and
            int(
                row[
                    "gradient_update"
                ]
            )
            ==
            1
        ):
            row[
                "stats"
            ][
                "actor_loss"
            ] = (
                float(
                    row[
                        "stats"
                    ][
                        "actor_loss"
                    ]
                )
                +
                1.0
            )

            target_count += 1

    assert (
        target_count
        ==
        1
    )

    metrics_path.write_text(
        (
            "\n".join(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(
                        ",",
                        ":",
                    ),
                    allow_nan=False,
                )
                for row in rows
            )
            +
            "\n"
        ),
        encoding="utf-8",
    )


    tampered_writer = (
        TrainingMetricsWriterV1(
            path=
                metrics_path
        )
    )

    (
        resumed_core,
        resumed_state,
        _,
        _,
    ) = _resume_existing_checkpoint(
        seed=42,

        run_directory=
            run_directory,

        checkpoint_path=
            checkpoint,

        require_clean_git=
            False,

        metrics_writer=
            tampered_writer,
    )

    assert resumed_state is not None

    try:
        _run_stage2_loop(
            core=
                resumed_core,

            stage2_state=
                resumed_state,

            run_directory=
                run_directory,

            checkpoint_path=
                checkpoint,

            last_checkpoint_environment_steps=
                0,

            require_clean_git=
                False,

            metrics_writer=
                tampered_writer,

            checkpoint_interval_environment_steps=
                4,

            episode_runner=
                make_plate3_runner(
                    tampered_writer
                ),

            max_new_episode_executions=
                1,

            resource_rearm_emit_logs=
                False,
        )

    except TrainingMetricsError:
        pass

    else:
        raise RuntimeError(
            "Conflicting logged-ahead Training Metrics "
            "row was accepted"
        )


def test_final_exact_budget_contract():
    """
    Cheap contract probe: the runner must ask the concrete writer for
    exactly 782 Stage-I rows and exactly 25,000 Stage-II rows.
    """

    core = prepare_formal_training_core(
        seed=42
    )

    calls = []


    class ExactContractProbe:
        def assert_exact_stage(
            self,
            *,
            seed,
            stage,
            gradient_updates,
        ):
            calls.append(
                (
                    int(
                        seed
                    ),
                    str(
                        stage
                    ),
                    int(
                        gradient_updates
                    ),
                )
            )

            return {
                "seed":
                    int(
                        seed
                    ),

                "stage":
                    str(
                        stage
                    ),

                "gradient_updates":
                    int(
                        gradient_updates
                    ),
            }


    probe = ExactContractProbe()

    result = (
        validate_complete_formal_training_metrics(
            metrics_writer=
                probe,

            core=
                core,
        )
    )

    assert calls == [
        (
            42,
            "STAGE_I",
            PROJECT_STAGE1_GRADIENT_STEPS,
        ),
        (
            42,
            "STAGE_II",
            PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS,
        ),
    ]

    assert (
        result[
            "total_records"
        ]
        ==
        PROJECT_STAGE1_GRADIENT_STEPS
        +
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
        ==
        25_782
    )


def main():
    test_checkpoint_order()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(
            tmp
        )

        test_incomplete_prefix_blocks_checkpoint(
            root
        )

        test_valid_plate3_checkpoint(
            root
        )

        test_logged_ahead_metrics_replay(
            root
        )

        test_conflicting_logged_ahead_metrics_fatal(
            root
        )

    test_final_exact_budget_contract()

    print(
        "PASS: metrics prefix validation and fsync "
        "precede formal checkpoint save"
    )

    print(
        "PASS: incomplete metrics prefix cannot "
        "create latest.pt"
    )

    print(
        "PASS: Plate3 checkpoint accepts exactly "
        "782 Stage-I + 2 Stage-II metric rows"
    )

    print(
        "PASS: checkpoint-ahead telemetry suffix "
        "replays without duplicate metrics/events"
    )

    print(
        "PASS: conflicting logged-ahead telemetry "
        "fails closed"
    )

    print(
        "PASS: final formal metrics contract is "
        "exactly 782 + 25,000 = 25,782 rows"
    )


if __name__ == "__main__":
    main()
