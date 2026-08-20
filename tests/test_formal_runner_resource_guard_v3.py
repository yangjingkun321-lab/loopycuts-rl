from __future__ import annotations

import os
import sys
import tempfile

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from bridge.resource_guard_v1 import (
    GIB,

    ProcessMemorySnapshot,
    ResourceGuardPolicyV1,
    ResourceSnapshot,
)

from training.formal_checkpoint_v1 import (
    load_formal_checkpoint,
    save_formal_checkpoint,
)

from training.formal_run_artifacts_v1 import (
    EVENT_LOG_FILENAME,

    create_formal_run_artifacts,
    read_formal_events,

    record_checkpoint,
    record_stage1_complete,
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
)

from training.run_formal_training_v1 import (
    FORMAL_RUNNER_VERSION,

    FormalRunnerError,

    _run_stage2_loop,
    latest_checkpoint_path,
    wait_for_formal_resource_rearm,
)


def snapshot(
    *,
    swap_used_gib,
):
    total = (
        34
        *
        GIB
    )

    used = int(
        float(
            swap_used_gib
        )
        *
        GIB
    )

    return ResourceSnapshot(
        mem_available_bytes=
            2 * GIB,

        swap_total_bytes=
            total,

        swap_free_bytes=
            total - used,

        swap_used_bytes=
            used,

        python_memory=
            ProcessMemorySnapshot(
                pid=
                    os.getpid(),

                rss_bytes=
                    768 * 1024 * 1024,

                swap_bytes=
                    256 * 1024 * 1024,
            ),

        cpp_memory=
            None,
    )


def emergency_episode_snapshot(
    *,
    cpp_pid,
):
    total = (
        34
        *
        GIB
    )

    used = (
        12
        *
        GIB
    )

    return ResourceSnapshot(
        mem_available_bytes=
            512 * 1024 * 1024,

        swap_total_bytes=
            total,

        swap_free_bytes=
            total - used,

        swap_used_bytes=
            used,

        python_memory=
            ProcessMemorySnapshot(
                pid=
                    os.getpid(),

                rss_bytes=
                    768 * 1024 * 1024,

                swap_bytes=
                    256 * 1024 * 1024,
            ),

        cpp_memory=
            ProcessMemorySnapshot(
                pid=
                    int(
                        cpp_pid
                    ),

                rss_bytes=
                    4 * GIB,

                swap_bytes=
                    8 * GIB,
            ),
    )


class FakeClock:
    def __init__(
        self,
    ):
        self.now = 0.0

    def monotonic(
        self,
    ):
        return float(
            self.now
        )

    def sleep(
        self,
        seconds,
    ):
        self.now += float(
            seconds
        )


def prepare_zero_stage2():
    core = (
        prepare_formal_training_core(
            seed=42
        )
    )

    core.stage1_updates_completed = (
        PROJECT_STAGE1_GRADIENT_STEPS
    )

    core.stage1_sampled_demo_transitions = (
        PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS
    )

    enter_formal_stage2(
        core
    )

    state = (
        prepare_formal_stage2_state(
            core
        )
    )

    return (
        core,
        state,
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


def main():
    assert (
        FORMAL_RUNNER_VERSION
        ==
        "loopycuts_formal_runner_v3_resource_guard"
    )


    # ============================================================
    # Pure re-arm helper:
    # 8 GiB -> wait -> 6 GiB -> READY.
    # ============================================================

    values = [
        8.0,
        6.0,
    ]

    def recovery_reader():
        value = (
            values.pop(
                0
            )
            if values
            else
            6.0
        )

        return snapshot(
            swap_used_gib=
                value
        )

    clock = FakeClock()

    recovery = (
        wait_for_formal_resource_rearm(
            resource_guard_policy=
                ResourceGuardPolicyV1(),

            resource_snapshot_reader=
                recovery_reader,

            sample_interval_seconds=
                1.0,

            timeout_seconds=
                60.0,

            sleep_fn=
                clock.sleep,

            monotonic_fn=
                clock.monotonic,

            emit_logs=
                False,
        )
    )

    assert (
        recovery[
            "rearmed"
        ]
        is True
    )

    assert (
        recovery[
            "swap_used_bytes"
        ]
        ==
        6 * GIB
    )

    assert (
        recovery[
            "sample_count"
        ]
        ==
        2
    )

    assert (
        recovery[
            "wait_seconds"
        ]
        ==
        1.0
    )


    # ============================================================
    # Timeout safety regression.
    # ============================================================

    timeout_clock = FakeClock()

    try:
        wait_for_formal_resource_rearm(
            resource_guard_policy=
                ResourceGuardPolicyV1(),

            resource_snapshot_reader=
                lambda:
                    snapshot(
                        swap_used_gib=
                            7.0
                    ),

            sample_interval_seconds=
                1.0,

            timeout_seconds=
                2.0,

            sleep_fn=
                timeout_clock.sleep,

            monotonic_fn=
                timeout_clock.monotonic,

            emit_logs=
                False,
        )

    except FormalRunnerError as exc:
        assert (
            "did not re-arm"
            in
            str(
                exc
            )
        )

    else:
        raise AssertionError(
            "Resource re-arm timeout did not fail safely"
        )


    # ============================================================
    # REAL formal RESOURCE_ABORT episode:
    #
    # event -> forced checkpoint -> re-arm
    # ============================================================

    with tempfile.TemporaryDirectory(
        prefix=
            "loopycuts_runner_guard_"
    ) as tmp:
        run_directory = Path(
            tmp
        )

        core, state = (
            prepare_zero_stage2()
        )

        create_formal_run_artifacts(
            run_directory=
                run_directory,

            core=
                core,

            require_clean_git=
                False,
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

        checkpoint_path = (
            latest_checkpoint_path(
                run_directory
            )
        )

        initial_checkpoint = (
            save_formal_checkpoint(
                checkpoint_path=
                    checkpoint_path,

                core=
                    core,

                stage2_state=
                    state,

                require_clean_git=
                    False,
            )
        )

        record_checkpoint(
            run_directory=
                run_directory,

            core=
                core,

            stage2_state=
                state,

            checkpoint_result=
                initial_checkpoint,
        )

        plate3 = next(
            model
            for model in state.models
            if model.model == "Plate3"
        )

        def abort_episode_runner(
            runner_core,
            runner_state,
        ):
            return (
                collect_formal_stage2_model_episode(
                    runner_core,
                    runner_state,

                    model=
                        plate3,

                    resource_guard_policy=
                        ResourceGuardPolicyV1(),

                    resource_guard_sample_interval_seconds=
                        0.02,

                    resource_snapshot_reader=
                        emergency_episode_snapshot,
                )
            )

        # First pair:
        #     pre-episode gate
        #
        # Second pair:
        #     post-RESOURCE_ABORT checkpoint recovery
        rearm_values = [
            8.0,
            6.0,
            8.0,
            6.0,
        ]

        def runner_rearm_reader():
            value = (
                rearm_values.pop(
                    0
                )
                if rearm_values
                else
                6.0
            )

            return snapshot(
                swap_used_gib=
                    value
            )

        runner_clock = (
            FakeClock()
        )

        result = (
            _run_stage2_loop(
                core=
                    core,

                stage2_state=
                    state,

                run_directory=
                    run_directory,

                checkpoint_path=
                    checkpoint_path,

                last_checkpoint_environment_steps=
                    0,

                require_clean_git=
                    False,

                # Far above env=1:
                # the checkpoint must therefore come ONLY from
                # RESOURCE_ABORT, not periodic cadence.
                checkpoint_interval_environment_steps=
                    2_500,

                episode_runner=
                    abort_episode_runner,

                max_new_episode_executions=
                    1,

                resource_rearm_policy=
                    ResourceGuardPolicyV1(),

                resource_snapshot_reader=
                    runner_rearm_reader,

                resource_rearm_sample_interval_seconds=
                    1.0,

                resource_rearm_timeout_seconds=
                    60.0,

                resource_rearm_sleep_fn=
                    runner_clock.sleep,

                resource_rearm_monotonic_fn=
                    runner_clock.monotonic,

                resource_rearm_emit_logs=
                    False,
            )
        )


        assert (
            state.total_environment_steps
            ==
            1
        )

        assert (
            state.total_gradient_updates
            ==
            1
        )

        assert (
            len(
                state.expo_buffer
            )
            ==
            1
        )

        assert (
            state.history[
                0
            ][
                "resource_abort"
            ]
            is True
        )

        assert (
            result[
                "resource_abort_episode_count"
            ]
            ==
            1
        )

        assert (
            result[
                "last_checkpoint_environment_steps"
            ]
            ==
            1
        )

        assert (
            result[
                "latest_checkpoint_result"
            ]
            is not None
        )

        assert (
            result[
                "latest_resource_rearm_result"
            ][
                "rearmed"
            ]
            is True
        )

        assert (
            result[
                "resource_rearm_wait_seconds_total"
            ]
            ==
            2.0
        )


        # ========================================================
        # Event order:
        #
        # Stage-I
        # initial checkpoint
        # RESOURCE_ABORT episode
        # immediate forced checkpoint
        # ========================================================

        events = (
            read_formal_events(
                event_log_path=
                    run_directory
                    /
                    EVENT_LOG_FILENAME
            )
        )

        event_types = [
            event[
                "event_type"
            ]
            for event in events
        ]

        assert event_types == [
            "STAGE1_COMPLETE",
            "CHECKPOINT",
            "STAGE2_EPISODE",
            "CHECKPOINT",
        ]


        # ========================================================
        # Forced checkpoint itself must contain the abort transition.
        # ========================================================

        (
            loaded_core,
            loaded_state,
            _,
        ) = load_formal_checkpoint(
            checkpoint_path=
                checkpoint_path,

            strict_git=
                False,
        )

        assert loaded_state is not None

        assert (
            loaded_state.total_environment_steps
            ==
            1
        )

        assert (
            loaded_state.total_gradient_updates
            ==
            1
        )

        assert (
            len(
                loaded_state.expo_buffer
            )
            ==
            1
        )

        assert (
            loaded_state.history[
                0
            ][
                "resource_abort"
            ]
            is True
        )

        assert (
            loaded_state.history[
                0
            ][
                "finalization_outcome"
            ]
            ==
            "RESOURCE_ABORT"
        )


    print(
        "PASS: every model launch requires <=6 GiB SwapUsed"
    )

    print(
        "PASS: RESOURCE_ABORT recovery also requires <=6 GiB before continuation"
    )

    print(
        "PASS: unrecovered >6 GiB state times out safely"
    )

    print(
        "PASS: RESOURCE_ABORT forces checkpoint independent of 2500-step cadence"
    )

    print(
        "PASS: forced checkpoint contains RESOURCE_ABORT replay transition"
    )

    print(
        "PASS: re-arm occurs only after RESOURCE_ABORT checkpoint"
    )

    print(
        "PASS: formal Python runner survives and can continue after re-arm"
    )


if __name__ == "__main__":
    main()
