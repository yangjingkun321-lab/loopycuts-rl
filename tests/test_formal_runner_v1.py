from __future__ import annotations

import copy
import shutil
import sys
import tempfile

from pathlib import Path


import numpy as np
import torch


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


from training.formal_checkpoint_v1 import (
    capture_global_rng_state,
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
    sample_formal_stage2_model,
)

from training.protocol_v1 import (
    PROJECT_STAGE1_GRADIENT_STEPS,
    PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS,
)

from training.run_formal_training_v1 import (
    FORMAL_CHECKPOINT_INTERVAL_ENV_STEPS,
    FORMAL_RUNNER_VERSION,

    _run_stage2_loop,
    checkpoint_due,
    latest_checkpoint_path,
)


def assert_nested_equal(
    left,
    right,
    *,
    path="root",
):
    if isinstance(
        left,
        torch.Tensor,
    ):
        assert isinstance(
            right,
            torch.Tensor,
        ), path

        assert torch.equal(
            left,
            right,
        ), path

        return

    if isinstance(
        left,
        np.ndarray,
    ):
        assert isinstance(
            right,
            np.ndarray,
        ), path

        assert np.array_equal(
            left,
            right,
        ), path

        return

    if isinstance(
        left,
        dict,
    ):
        assert isinstance(
            right,
            dict,
        ), path

        assert (
            left.keys()
            ==
            right.keys()
        ), path

        for key in left:
            assert_nested_equal(
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

    if isinstance(
        left,
        (
            list,
            tuple,
        ),
    ):
        assert isinstance(
            right,
            type(
                left
            ),
        ), path

        assert (
            len(
                left
            )
            ==
            len(
                right
            )
        ), path

        for index, (
            l_value,
            r_value,
        ) in enumerate(
            zip(
                left,
                right,
            )
        ):
            assert_nested_equal(
                l_value,
                r_value,
                path=
                    f"{path}[{index}]",
            )

        return

    assert (
        left
        ==
        right
    ), (
        path,
        left,
        right,
    )


def prepare_zero_stage2():
    core = prepare_formal_training_core(
        seed=42
    )

    # Phase 18.2 already executed the exact real 782-update Stage-I
    # integration smoke. This runner test validates orchestration and
    # interruption semantics only.
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


def fixed_plate3_runner(
    core,
    state,
):
    # Consume the exact production model-sampler RNG once per episode
    # so interruption/resume also validates scheduler RNG continuity.
    scheduled_model = (
        sample_formal_stage2_model(
            state
        )
        .model
    )

    plate3 = next(
        model
        for model in state.models
        if model.model == "Plate3"
    )

    record = (
        collect_formal_stage2_model_episode(
            core,
            state,

            model=
                plate3,
        )
    )

    # Test-only audit field. Mutation is deliberate: collect() appends
    # this same record object to state.history.
    record[
        "scheduled_model_for_rng_audit"
    ] = (
        scheduled_model
    )

    return record


def bootstrap_branch(
    *,
    branch_directory: Path,
    source_checkpoint: Path,
):
    branch_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = (
        latest_checkpoint_path(
            branch_directory
        )
    )

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source_checkpoint,
        checkpoint,
    )

    (
        core,
        state,
        _,
    ) = load_formal_checkpoint(
        checkpoint_path=
            checkpoint,

        strict_git=
            False,
    )

    assert state is not None

    create_formal_run_artifacts(
        run_directory=
            branch_directory,

        core=
            core,

        require_clean_git=
            False,
    )

    synthetic_stage1 = {
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

    record_stage1_complete(
        run_directory=
            branch_directory,

        core=
            core,

        stage1_result=
            synthetic_stage1,
    )

    checkpoint_result = {
        "path":
            str(
                checkpoint
            ),

        "bytes":
            int(
                checkpoint
                .stat()
                .st_size
            ),

        "sha256":
            __import__(
                "hashlib"
            )
            .sha256(
                checkpoint.read_bytes()
            )
            .hexdigest(),

        "stage":
            core.stage,

        "seed":
            core.seed,
    }

    record_checkpoint(
        run_directory=
            branch_directory,

        core=
            core,

        stage2_state=
            state,

        checkpoint_result=
            checkpoint_result,
    )

    return (
        core,
        state,
        checkpoint,
    )


def capture_branch_state(
    core,
    state,
):
    return {
        "algorithm":
            copy.deepcopy(
                core.algorithm.state_dict()
            ),

        "alpha_optimizer":
            copy.deepcopy(
                core.auto_alpha
                ._optim
                .state_dict()
            ),

        "history":
            copy.deepcopy(
                state.history
            ),

        "demo_rng":
            copy.deepcopy(
                core.demo_buffer
                ._random_state
                .get_state()
            ),

        "expo_rng":
            copy.deepcopy(
                state.expo_buffer
                ._random_state
                .get_state()
            ),

        "model_rng":
            copy.deepcopy(
                state.model_rng
                .bit_generator
                .state
            ),

        "exploration_rng":
            copy.deepcopy(
                core.policy
                ._exploration_rng
                .bit_generator
                .state
            ),

        "global_rng":
            capture_global_rng_state(),

        "environment_steps":
            state.total_environment_steps,

        "gradient_updates":
            state.total_gradient_updates,

        "episode_attempts":
            state.episode_attempts,
    }


def main():
    assert (
        FORMAL_RUNNER_VERSION
        ==
        "loopycuts_formal_runner_v3_resource_guard"
    )

    assert (
        FORMAL_CHECKPOINT_INTERVAL_ENV_STEPS
        ==
        2_500
    )

    assert (
        checkpoint_due(
            last_checkpoint_environment_steps=
                0,

            current_environment_steps=
                2_499,
        )
        is False
    )

    assert (
        checkpoint_due(
            last_checkpoint_environment_steps=
                0,

            current_environment_steps=
                2_500,
        )
        is True
    )


    with tempfile.TemporaryDirectory() as tmp:
        root = Path(
            tmp
        )

        source_checkpoint = (
            root
            /
            "source_zero_stage2.pt"
        )

        (
            source_core,
            source_state,
        ) = prepare_zero_stage2()

        source_save = (
            save_formal_checkpoint(
                checkpoint_path=
                    source_checkpoint,

                core=
                    source_core,

                stage2_state=
                    source_state,

                require_clean_git=
                    False,
            )
        )

        assert (
            source_state.total_environment_steps
            ==
            0
        )


        # ============================================================
        # BASELINE:
        # zero checkpoint -> one Plate3 episode.
        # ============================================================

        baseline_directory = (
            root
            /
            "baseline"
        )

        (
            baseline_core,
            baseline_state,
            baseline_checkpoint,
        ) = bootstrap_branch(
            branch_directory=
                baseline_directory,

            source_checkpoint=
                source_checkpoint,
        )

        baseline_result = (
            _run_stage2_loop(
                core=
                    baseline_core,

                stage2_state=
                    baseline_state,

                run_directory=
                    baseline_directory,

                checkpoint_path=
                    baseline_checkpoint,

                last_checkpoint_environment_steps=
                    0,

                require_clean_git=
                    False,

                # Prevent a new checkpoint after this 2-step episode.
                checkpoint_interval_environment_steps=
                    4,

                episode_runner=
                    fixed_plate3_runner,

                max_new_episode_executions=
                    1,
            )
        )

        assert (
            baseline_result[
                "appended_episode_events"
            ]
            ==
            1
        )

        assert (
            baseline_result[
                "replayed_episode_events"
            ]
            ==
            0
        )

        baseline_snapshot = (
            capture_branch_state(
                baseline_core,
                baseline_state,
            )
        )


        # ============================================================
        # INTERRUPTED:
        #
        # zero checkpoint
        # -> episode is completed/logged
        # -> no checkpoint because interval=4
        # -> simulated process death.
        # ============================================================

        interrupted_directory = (
            root
            /
            "interrupted"
        )

        (
            interrupted_core,
            interrupted_state,
            interrupted_checkpoint,
        ) = bootstrap_branch(
            branch_directory=
                interrupted_directory,

            source_checkpoint=
                source_checkpoint,
        )

        first_execution = (
            _run_stage2_loop(
                core=
                    interrupted_core,

                stage2_state=
                    interrupted_state,

                run_directory=
                    interrupted_directory,

                checkpoint_path=
                    interrupted_checkpoint,

                last_checkpoint_environment_steps=
                    0,

                require_clean_git=
                    False,

                checkpoint_interval_environment_steps=
                    4,

                episode_runner=
                    fixed_plate3_runner,

                max_new_episode_executions=
                    1,
            )
        )

        assert (
            first_execution[
                "appended_episode_events"
            ]
            ==
            1
        )

        assert (
            first_execution[
                "latest_checkpoint_result"
            ]
            is None
        )

        assert (
            interrupted_state.total_environment_steps
            ==
            2
        )


        events_before_restart = (
            read_formal_events(
                event_log_path=
                    interrupted_directory
                    /
                    EVENT_LOG_FILENAME
            )
        )

        assert [
            event[
                "event_type"
            ]
            for event in
            events_before_restart
        ] == [
            "STAGE1_COMPLETE",
            "CHECKPOINT",
            "STAGE2_EPISODE",
        ]


        # ============================================================
        # SIMULATED RESTART:
        #
        # latest.pt is still the zero-transition checkpoint.
        # The Stage-II episode exists only in the durable event log.
        # ============================================================

        (
            resumed_core,
            resumed_state,
            _,
        ) = load_formal_checkpoint(
            checkpoint_path=
                interrupted_checkpoint,

            strict_git=
                False,
        )

        assert resumed_state is not None

        assert (
            resumed_state.total_environment_steps
            ==
            0
        )

        replay_execution = (
            _run_stage2_loop(
                core=
                    resumed_core,

                stage2_state=
                    resumed_state,

                run_directory=
                    interrupted_directory,

                checkpoint_path=
                    interrupted_checkpoint,

                last_checkpoint_environment_steps=
                    0,

                require_clean_git=
                    False,

                checkpoint_interval_environment_steps=
                    4,

                episode_runner=
                    fixed_plate3_runner,

                max_new_episode_executions=
                    1,
            )
        )

        assert (
            replay_execution[
                "appended_episode_events"
            ]
            ==
            0
        )

        assert (
            replay_execution[
                "replayed_episode_events"
            ]
            ==
            1
        )


        events_after_restart = (
            read_formal_events(
                event_log_path=
                    interrupted_directory
                    /
                    EVENT_LOG_FILENAME
            )
        )

        assert (
            len(
                events_after_restart
            )
            ==
            len(
                events_before_restart
            )
        )

        assert [
            event[
                "event_type"
            ]
            for event in
            events_after_restart
        ] == [
            "STAGE1_COMPLETE",
            "CHECKPOINT",
            "STAGE2_EPISODE",
        ]


        resumed_snapshot = (
            capture_branch_state(
                resumed_core,
                resumed_state,
            )
        )


        # ============================================================
        # STRONG INTERRUPTION EQUIVALENCE.
        # ============================================================

        assert_nested_equal(
            baseline_snapshot,
            resumed_snapshot,
            path=
                "baseline_vs_resumed",
        )


        assert (
            baseline_state.history[
                0
            ][
                "actions"
            ]
            ==
            resumed_state.history[
                0
            ][
                "actions"
            ]
            ==
            [
                1,
                0,
            ]
        )

        assert (
            baseline_state.history[
                0
            ][
                "finalization_outcome"
            ]
            ==
            resumed_state.history[
                0
            ][
                "finalization_outcome"
            ]
            ==
            "FULL_HEX"
        )


        print("=" * 104)
        print("FORMAL RUNNER V3 RESOURCE GUARD INTERRUPTION / RESUME")
        print("=" * 104)

        print(
            "source checkpoint bytes :",
            source_save[
                "bytes"
            ],
        )

        print(
            "checkpoint cadence       :",
            FORMAL_CHECKPOINT_INTERVAL_ENV_STEPS,
        )

        print(
            "baseline actions         :",
            baseline_state.history[
                0
            ][
                "actions"
            ],
        )

        print(
            "resumed actions          :",
            resumed_state.history[
                0
            ][
                "actions"
            ],
        )

        print(
            "outcome                  :",
            resumed_state.history[
                0
            ][
                "finalization_outcome"
            ],
        )

        print(
            "event count before crash :",
            len(
                events_before_restart
            ),
        )

        print(
            "event count after replay :",
            len(
                events_after_restart
            ),
        )

        print()

        print(
            "PASS: formal runner checkpoint cadence is fixed at 2500 Stage-II transitions"
        )

        print(
            "PASS: an episode can be durably logged ahead of the latest checkpoint"
        )

        print(
            "PASS: resume re-executes logged-ahead work without duplicating the episode event"
        )

        print(
            "PASS: replayed Plate3 action sequence remains exactly [1, 0] with FULL_HEX"
        )

        print(
            "PASS: resumed neural/optimizer/replay/RNG state is bitwise-equivalent to uninterrupted execution"
        )


if __name__ == "__main__":
    main()
