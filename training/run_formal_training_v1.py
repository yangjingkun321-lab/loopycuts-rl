from __future__ import annotations

import argparse

from pathlib import Path
from typing import Callable, Any


from training.formal_checkpoint_v1 import (
    FormalCheckpointError,
    load_formal_checkpoint,
    repository_is_dirty,
    save_formal_checkpoint,
    sha256_file,
)

from training.formal_run_artifacts_v1 import (
    EVENT_LOG_FILENAME,
    FormalRunArtifactError,

    create_formal_run_artifacts,
    load_and_validate_run_manifest,
    read_formal_events,

    record_checkpoint,
    record_run_complete,
    record_stage1_complete,
    record_stage2_episode,
    to_jsonable,
)

from training.formal_training_v1 import (
    FormalStage2StateV1,
    FormalTrainingCoreV1,

    enter_formal_stage2,
    prepare_formal_stage2_state,
    prepare_formal_training_core,

    run_formal_stage1,
    run_next_formal_stage2_episode,
)

from training.protocol_v1 import (
    PROJECT_FORMAL_TRAINING_SEEDS,

    PROJECT_STAGE1_GRADIENT_STEPS,
    PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS,

    PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS,
)


FORMAL_RUNNER_VERSION = (
    "loopycuts_formal_runner_v1"
)


# ----------------------------------------------------------------------
# Engineering-only recovery cadence.
#
# This is deliberately NOT a Training Protocol hyperparameter.
# It changes only how much completed work could need deterministic
# replay after an interruption.
#
# 25,000 / 2,500 -> approximately ten Stage-II periodic checkpoints,
# plus Stage-I and final checkpoints.
#
# Actual saves occur only after a fully collected/flushed episode,
# except the exact final 25,000-transition prefix, which is itself a
# valid final checkpoint boundary.
# ----------------------------------------------------------------------

FORMAL_CHECKPOINT_INTERVAL_ENV_STEPS = (
    2_500
)


DEFAULT_FORMAL_RUN_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "formal_training_v1"
)

CHECKPOINT_DIRECTORY_NAME = (
    "checkpoints"
)

LATEST_CHECKPOINT_FILENAME = (
    "latest.pt"
)


class FormalRunnerError(
    RuntimeError
):
    pass


def formal_run_directory(
    *,
    seed: int,
    run_root: Path =
        DEFAULT_FORMAL_RUN_ROOT,
):
    return (
        Path(
            run_root
        )
        .resolve()
        /
        f"seed_{int(seed)}"
    )


def latest_checkpoint_path(
    run_directory: Path,
):
    return (
        Path(
            run_directory
        )
        .resolve()
        /
        CHECKPOINT_DIRECTORY_NAME
        /
        LATEST_CHECKPOINT_FILENAME
    )


def checkpoint_due(
    *,
    last_checkpoint_environment_steps: int,
    current_environment_steps: int,
    interval_environment_steps: int =
        FORMAL_CHECKPOINT_INTERVAL_ENV_STEPS,
):
    last_checkpoint_environment_steps = int(
        last_checkpoint_environment_steps
    )

    current_environment_steps = int(
        current_environment_steps
    )

    interval_environment_steps = int(
        interval_environment_steps
    )

    if interval_environment_steps <= 0:
        raise FormalRunnerError(
            "Checkpoint interval must be positive"
        )

    if (
        current_environment_steps
        <
        last_checkpoint_environment_steps
    ):
        raise FormalRunnerError(
            "Checkpoint environment-step counter moved backwards"
        )

    if (
        current_environment_steps
        ==
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
    ):
        return True

    return (
        current_environment_steps
        -
        last_checkpoint_environment_steps
        >=
        interval_environment_steps
    )


def checkpoint_result_from_existing(
    *,
    checkpoint_path: Path,
    core: FormalTrainingCoreV1,
):
    checkpoint_path = Path(
        checkpoint_path
    ).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    return {
        "path":
            str(
                checkpoint_path
            ),

        "bytes":
            int(
                checkpoint_path
                .stat()
                .st_size
            ),

        "sha256":
            sha256_file(
                checkpoint_path
            ),

        "stage":
            core.stage,

        "seed":
            int(
                core.seed
            ),
    }


def save_and_record_formal_checkpoint(
    *,
    run_directory: Path,
    checkpoint_path: Path,
    core: FormalTrainingCoreV1,
    stage2_state: FormalStage2StateV1 | None,
    require_clean_git: bool,
):
    result = save_formal_checkpoint(
        checkpoint_path=
            checkpoint_path,

        core=
            core,

        stage2_state=
            stage2_state,

        require_clean_git=
            require_clean_git,
    )

    record_checkpoint(
        run_directory=
            run_directory,

        core=
            core,

        stage2_state=
            stage2_state,

        checkpoint_result=
            result,
    )

    return result


def _stage2_events(
    events,
):
    return [
        event
        for event in events
        if (
            event[
                "event_type"
            ]
            ==
            "STAGE2_EPISODE"
        )
    ]


def _checkpoint_events(
    events,
):
    return [
        event
        for event in events
        if (
            event[
                "event_type"
            ]
            ==
            "CHECKPOINT"
        )
    ]


def validate_event_log_against_checkpoint_state(
    *,
    run_directory: Path,
    core: FormalTrainingCoreV1,
    stage2_state: FormalStage2StateV1 | None,
):
    events = read_formal_events(
        event_log_path=
            Path(
                run_directory
            )
            /
            EVENT_LOG_FILENAME
    )

    stage1_events = [
        event
        for event in events
        if (
            event[
                "event_type"
            ]
            ==
            "STAGE1_COMPLETE"
        )
    ]

    if (
        core.stage1_updates_completed
        ==
        PROJECT_STAGE1_GRADIENT_STEPS
    ):
        if (
            len(
                stage1_events
            )
            !=
            1
        ):
            raise FormalRunnerError(
                "Completed formal checkpoint must correspond to exactly "
                "one STAGE1_COMPLETE event"
            )

    stage2_events = _stage2_events(
        events
    )

    if stage2_state is not None:
        if (
            len(
                stage2_events
            )
            <
            stage2_state.episode_attempts
        ):
            raise FormalRunnerError(
                "Formal checkpoint is ahead of its Stage-II episode log"
            )

        if (
            len(
                stage2_state.history
            )
            !=
            stage2_state.episode_attempts
        ):
            raise FormalRunnerError(
                "Checkpoint Stage-II history/episode counter mismatch"
            )

        # Events up through the checkpoint boundary must exactly agree
        # with the checkpoint's own history. Events after that boundary
        # are allowed: they represent logged work that will be
        # deterministically re-executed after resume.
        for index in range(
            stage2_state.episode_attempts
        ):
            logged_record = (
                stage2_events[
                    index
                ][
                    "payload"
                ][
                    "record"
                ]
            )

            checkpoint_record = to_jsonable(
                stage2_state.history[
                    index
                ]
            )

            if (
                logged_record
                !=
                checkpoint_record
            ):
                raise FormalRunnerError(
                    "Checkpoint Stage-II history differs from formal event log "
                    f"at episode {index + 1}"
                )

    run_complete_events = [
        event
        for event in events
        if (
            event[
                "event_type"
            ]
            ==
            "RUN_COMPLETE"
        )
    ]

    if (
        len(
            run_complete_events
        )
        >
        1
    ):
        raise FormalRunnerError(
            "Multiple RUN_COMPLETE events found"
        )

    if run_complete_events:
        if (
            events[
                -1
            ][
                "event_type"
            ]
            !=
            "RUN_COMPLETE"
        ):
            raise FormalRunnerError(
                "RUN_COMPLETE must be the final formal event"
            )

        if (
            stage2_state is None
            or
            stage2_state.total_environment_steps
            !=
            PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
        ):
            raise FormalRunnerError(
                "RUN_COMPLETE exists but checkpoint is not final"
            )

    return events


def ensure_loaded_checkpoint_event(
    *,
    run_directory: Path,
    core: FormalTrainingCoreV1,
    stage2_state: FormalStage2StateV1 | None,
    checkpoint_result,
):
    events = read_formal_events(
        event_log_path=
            Path(
                run_directory
            )
            /
            EVENT_LOG_FILENAME
    )

    checkpoint_sha = checkpoint_result[
        "sha256"
    ]

    for event in _checkpoint_events(
        events
    ):
        if (
            event[
                "payload"
            ][
                "checkpoint"
            ][
                "sha256"
            ]
            ==
            checkpoint_sha
        ):
            return {
                "represented":
                    True,

                "event":
                    event,
            }

    stage2_events = _stage2_events(
        events
    )

    checkpoint_episode_attempts = (
        0
        if stage2_state is None
        else
        int(
            stage2_state.episode_attempts
        )
    )

    # If the log has moved beyond this checkpoint, then this checkpoint
    # must already have had a historical CHECKPOINT event. Appending a
    # checkpoint event now would falsify chronological ordering.
    if (
        len(
            stage2_events
        )
        >
        checkpoint_episode_attempts
    ):
        raise FormalRunnerError(
            "Current checkpoint SHA is absent from the log even though "
            "the event log has advanced beyond its boundary"
        )

    result = record_checkpoint(
        run_directory=
            run_directory,

        core=
            core,

        stage2_state=
            stage2_state,

        checkpoint_result=
            checkpoint_result,
    )

    return {
        "represented":
            False,

        "event":
            result[
                "event"
            ],
    }


def _run_stage2_loop(
    *,
    core: FormalTrainingCoreV1,
    stage2_state: FormalStage2StateV1,
    run_directory: Path,
    checkpoint_path: Path,

    last_checkpoint_environment_steps: int,

    require_clean_git: bool,

    checkpoint_interval_environment_steps: int =
        FORMAL_CHECKPOINT_INTERVAL_ENV_STEPS,

    episode_runner: Callable[
        [
            FormalTrainingCoreV1,
            FormalStage2StateV1,
        ],
        dict,
    ] | None = None,

    # Test-only interruption hook. The production public entry point
    # always leaves this as None.
    max_new_episode_executions: int | None = None,
):
    if episode_runner is None:
        def production_episode_runner(
            runner_core,
            runner_state,
        ):
            return run_next_formal_stage2_episode(
                runner_core,
                runner_state,
            )

        episode_runner = (
            production_episode_runner
        )

    if (
        max_new_episode_executions
        is not None
        and
        max_new_episode_executions < 0
    ):
        raise FormalRunnerError(
            "max_new_episode_executions cannot be negative"
        )

    last_checkpoint_environment_steps = int(
        last_checkpoint_environment_steps
    )

    executed = 0

    appended_episode_events = 0
    replayed_episode_events = 0

    latest_checkpoint_result = None

    while (
        stage2_state.total_environment_steps
        <
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
    ):
        if (
            max_new_episode_executions
            is not None
            and
            executed
            >=
            max_new_episode_executions
        ):
            break

        record = episode_runner(
            core,
            stage2_state,
        )

        artifact_result = record_stage2_episode(
            run_directory=
                run_directory,

            core=
                core,

            record=
                record,
        )

        if artifact_result[
            "replayed"
        ]:
            replayed_episode_events += 1

        else:
            appended_episode_events += 1

        executed += 1

        print(
            "formal-runner "
            f"episode={record['episode_index']} "
            f"model={record['model']} "
            f"steps={record['steps']} "
            f"outcome={record['finalization_outcome']} "
            f"env={stage2_state.total_environment_steps}/"
            f"{PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS} "
            f"updates={stage2_state.total_gradient_updates}"
        )

        if checkpoint_due(
            last_checkpoint_environment_steps=
                last_checkpoint_environment_steps,

            current_environment_steps=
                stage2_state.total_environment_steps,

            interval_environment_steps=
                checkpoint_interval_environment_steps,
        ):
            latest_checkpoint_result = (
                save_and_record_formal_checkpoint(
                    run_directory=
                        run_directory,

                    checkpoint_path=
                        checkpoint_path,

                    core=
                        core,

                    stage2_state=
                        stage2_state,

                    require_clean_git=
                        require_clean_git,
                )
            )

            last_checkpoint_environment_steps = (
                stage2_state.total_environment_steps
            )

    return {
        "executed_episode_count":
            executed,

        "appended_episode_events":
            appended_episode_events,

        "replayed_episode_events":
            replayed_episode_events,

        "last_checkpoint_environment_steps":
            last_checkpoint_environment_steps,

        "latest_checkpoint_result":
            latest_checkpoint_result,

        "budget_complete":
            (
                stage2_state.total_environment_steps
                ==
                PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
            ),
    }


def finalize_completed_formal_run(
    *,
    core: FormalTrainingCoreV1,
    stage2_state: FormalStage2StateV1,
    run_directory: Path,
    checkpoint_path: Path,
    last_checkpoint_environment_steps: int,
    latest_checkpoint_result,
    require_clean_git: bool,
):
    if (
        stage2_state.total_environment_steps
        !=
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
    ):
        raise FormalRunnerError(
            "Cannot finalize before exact 25,000-transition budget"
        )

    if (
        stage2_state.total_gradient_updates
        !=
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
    ):
        raise FormalRunnerError(
            "Cannot finalize before exact Stage-II update budget"
        )

    if (
        len(
            stage2_state.expo_buffer
        )
        !=
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
    ):
        raise FormalRunnerError(
            "Cannot finalize with incorrect D_expo size"
        )

    if (
        latest_checkpoint_result is None
        or
        int(
            last_checkpoint_environment_steps
        )
        !=
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
    ):
        latest_checkpoint_result = (
            save_and_record_formal_checkpoint(
                run_directory=
                    run_directory,

                checkpoint_path=
                    checkpoint_path,

                core=
                    core,

                stage2_state=
                    stage2_state,

                require_clean_git=
                    require_clean_git,
            )
        )

    record_run_complete(
        run_directory=
            run_directory,

        core=
            core,

        stage2_state=
            stage2_state,

        final_checkpoint_result=
            latest_checkpoint_result,
    )

    return latest_checkpoint_result


def _resume_existing_checkpoint(
    *,
    seed: int,
    run_directory: Path,
    checkpoint_path: Path,
    require_clean_git: bool,
):
    (
        core,
        stage2_state,
        load_result,
    ) = load_formal_checkpoint(
        checkpoint_path=
            checkpoint_path,

        strict_git=
            require_clean_git,
    )

    if (
        int(
            core.seed
        )
        !=
        int(
            seed
        )
    ):
        raise FormalRunnerError(
            "Checkpoint seed does not match requested formal seed"
        )

    load_and_validate_run_manifest(
        run_directory=
            run_directory,

        core=
            core,

        strict_git=
            require_clean_git,
    )

    validate_event_log_against_checkpoint_state(
        run_directory=
            run_directory,

        core=
            core,

        stage2_state=
            stage2_state,
    )

    checkpoint_result = (
        checkpoint_result_from_existing(
            checkpoint_path=
                checkpoint_path,

            core=
                core,
        )
    )

    ensure_loaded_checkpoint_event(
        run_directory=
            run_directory,

        core=
            core,

        stage2_state=
            stage2_state,

        checkpoint_result=
            checkpoint_result,
    )

    return (
        core,
        stage2_state,
        checkpoint_result,
        load_result,
    )


def run_new_formal_training(
    *,
    seed: int,
    run_directory: Path,
    require_clean_git: bool = True,
):
    seed = int(
        seed
    )

    if (
        seed
        not in
        PROJECT_FORMAL_TRAINING_SEEDS
    ):
        raise FormalRunnerError(
            f"Seed is not a frozen formal seed: {seed}"
        )

    run_directory = Path(
        run_directory
    ).resolve()

    checkpoint_path = (
        latest_checkpoint_path(
            run_directory
        )
    )

    if run_directory.exists():
        if any(
            run_directory.iterdir()
        ):
            raise FormalRunnerError(
                "Refusing to start a new formal run in a non-empty directory"
            )

    if (
        require_clean_git
        and
        repository_is_dirty()
    ):
        raise FormalRunnerError(
            "Formal training requires a clean Git worktree"
        )

    core = prepare_formal_training_core(
        seed=
            seed
    )

    create_formal_run_artifacts(
        run_directory=
            run_directory,

        core=
            core,

        require_clean_git=
            require_clean_git,
    )

    print(
        f"formal-runner seed={seed} Stage-I starting"
    )

    stage1_result = (
        run_formal_stage1(
            core
        )
    )

    record_stage1_complete(
        run_directory=
            run_directory,

        core=
            core,

        stage1_result=
            stage1_result,
    )

    stage1_checkpoint = (
        save_and_record_formal_checkpoint(
            run_directory=
                run_directory,

            checkpoint_path=
                checkpoint_path,

            core=
                core,

            stage2_state=
                None,

            require_clean_git=
                require_clean_git,
        )
    )

    enter_formal_stage2(
        core
    )

    stage2_state = (
        prepare_formal_stage2_state(
            core
        )
    )

    loop_result = _run_stage2_loop(
        core=
            core,

        stage2_state=
            stage2_state,

        run_directory=
            run_directory,

        checkpoint_path=
            checkpoint_path,

        last_checkpoint_environment_steps=
            0,

        require_clean_git=
            require_clean_git,
    )

    final_checkpoint = (
        finalize_completed_formal_run(
            core=
                core,

            stage2_state=
                stage2_state,

            run_directory=
                run_directory,

            checkpoint_path=
                checkpoint_path,

            last_checkpoint_environment_steps=
                loop_result[
                    "last_checkpoint_environment_steps"
                ],

            latest_checkpoint_result=
                loop_result[
                    "latest_checkpoint_result"
                ],

            require_clean_git=
                require_clean_git,
        )
    )

    return {
        "seed":
            seed,

        "run_directory":
            str(
                run_directory
            ),

        "stage1_checkpoint":
            stage1_checkpoint,

        "final_checkpoint":
            final_checkpoint,

        "environment_steps":
            stage2_state.total_environment_steps,

        "gradient_updates":
            stage2_state.total_gradient_updates,

        "episode_attempts":
            stage2_state.episode_attempts,
    }


def resume_formal_training(
    *,
    seed: int,
    run_directory: Path,
    require_clean_git: bool = True,
):
    seed = int(
        seed
    )

    if (
        seed
        not in
        PROJECT_FORMAL_TRAINING_SEEDS
    ):
        raise FormalRunnerError(
            f"Seed is not a frozen formal seed: {seed}"
        )

    run_directory = Path(
        run_directory
    ).resolve()

    checkpoint_path = (
        latest_checkpoint_path(
            run_directory
        )
    )

    if (
        require_clean_git
        and
        repository_is_dirty()
    ):
        raise FormalRunnerError(
            "Formal resume requires a clean Git worktree"
        )

    if not run_directory.is_dir():
        raise FileNotFoundError(
            run_directory
        )


    # ------------------------------------------------------------------
    # Normal resume path: an atomic checkpoint exists.
    # ------------------------------------------------------------------

    if checkpoint_path.is_file():
        (
            core,
            stage2_state,
            checkpoint_result,
            _,
        ) = _resume_existing_checkpoint(
            seed=
                seed,

            run_directory=
                run_directory,

            checkpoint_path=
                checkpoint_path,

            require_clean_git=
                require_clean_git,
        )

        if (
            core.stage
            ==
            "STAGE_I"
        ):
            if stage2_state is not None:
                raise FormalRunnerError(
                    "STAGE_I checkpoint unexpectedly contains Stage-II state"
                )

            enter_formal_stage2(
                core
            )

            stage2_state = (
                prepare_formal_stage2_state(
                    core
                )
            )

            last_checkpoint_environment_steps = 0

        elif (
            core.stage
            ==
            "STAGE_II"
        ):
            if stage2_state is None:
                raise FormalRunnerError(
                    "STAGE_II checkpoint lacks Stage-II state"
                )

            last_checkpoint_environment_steps = (
                stage2_state.total_environment_steps
            )

        else:
            raise FormalRunnerError(
                f"Unknown checkpoint stage: {core.stage}"
            )


    # ------------------------------------------------------------------
    # Rare recovery path:
    #
    # run_manifest/events exist, but Stage-I checkpoint was never
    # committed. This can happen only before the first successful
    # checkpoint. Re-run deterministic Stage-I from the frozen seed.
    # ------------------------------------------------------------------

    else:
        probe_core = (
            prepare_formal_training_core(
                seed=
                    seed
            )
        )

        load_and_validate_run_manifest(
            run_directory=
                run_directory,

            core=
                probe_core,

            strict_git=
                require_clean_git,
        )

        events = read_formal_events(
            event_log_path=
                run_directory
                /
                EVENT_LOG_FILENAME
        )

        forbidden_types = {
            "STAGE2_EPISODE",
            "CHECKPOINT",
            "RUN_COMPLETE",
        }

        if any(
            event[
                "event_type"
            ]
            in
            forbidden_types
            for event in events
        ):
            raise FormalRunnerError(
                "Formal run has post-Stage-I events but latest checkpoint is missing"
            )

        core = probe_core

        print(
            f"formal-runner seed={seed} restarting deterministic Stage-I"
        )

        stage1_result = (
            run_formal_stage1(
                core
            )
        )

        stage1_log = (
            record_stage1_complete(
                run_directory=
                    run_directory,

                core=
                    core,

                stage1_result=
                    stage1_result,
            )
        )

        if stage1_log[
            "replayed"
        ]:
            print(
                "formal-runner Stage-I event replay matched existing artifact"
            )

        checkpoint_result = (
            save_and_record_formal_checkpoint(
                run_directory=
                    run_directory,

                checkpoint_path=
                    checkpoint_path,

                core=
                    core,

                stage2_state=
                    None,

                require_clean_git=
                    require_clean_git,
            )
        )

        enter_formal_stage2(
            core
        )

        stage2_state = (
            prepare_formal_stage2_state(
                core
            )
        )

        last_checkpoint_environment_steps = 0


    # ------------------------------------------------------------------
    # Already finished:
    # only ensure RUN_COMPLETE exists/idempotently matches.
    # ------------------------------------------------------------------

    if (
        stage2_state.total_environment_steps
        ==
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
    ):
        final_checkpoint = (
            checkpoint_result_from_existing(
                checkpoint_path=
                    checkpoint_path,

                core=
                    core,
            )
        )

        record_run_complete(
            run_directory=
                run_directory,

            core=
                core,

            stage2_state=
                stage2_state,

            final_checkpoint_result=
                final_checkpoint,
        )

        return {
            "seed":
                seed,

            "run_directory":
                str(
                    run_directory
                ),

            "resumed":
                True,

            "already_complete":
                True,

            "final_checkpoint":
                final_checkpoint,

            "environment_steps":
                stage2_state.total_environment_steps,

            "gradient_updates":
                stage2_state.total_gradient_updates,

            "episode_attempts":
                stage2_state.episode_attempts,
        }


    print(
        "formal-runner resume "
        f"seed={seed} "
        f"checkpoint_env={last_checkpoint_environment_steps} "
        f"logged_episode_count="
        f"{len(_stage2_events(read_formal_events(event_log_path=run_directory / EVENT_LOG_FILENAME)))}"
    )


    loop_result = _run_stage2_loop(
        core=
            core,

        stage2_state=
            stage2_state,

        run_directory=
            run_directory,

        checkpoint_path=
            checkpoint_path,

        last_checkpoint_environment_steps=
            last_checkpoint_environment_steps,

        require_clean_git=
            require_clean_git,
    )

    final_checkpoint = (
        finalize_completed_formal_run(
            core=
                core,

            stage2_state=
                stage2_state,

            run_directory=
                run_directory,

            checkpoint_path=
                checkpoint_path,

            last_checkpoint_environment_steps=
                loop_result[
                    "last_checkpoint_environment_steps"
                ],

            latest_checkpoint_result=
                loop_result[
                    "latest_checkpoint_result"
                ],

            require_clean_git=
                require_clean_git,
        )
    )

    return {
        "seed":
            seed,

        "run_directory":
            str(
                run_directory
            ),

        "resumed":
            True,

        "already_complete":
            False,

        "final_checkpoint":
            final_checkpoint,

        "environment_steps":
            stage2_state.total_environment_steps,

        "gradient_updates":
            stage2_state.total_gradient_updates,

        "episode_attempts":
            stage2_state.episode_attempts,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LoopyCuts formal Stage-I + Stage-II training runner. "
            "This command always uses the frozen protocol budgets."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=
            PROJECT_FORMAL_TRAINING_SEEDS,
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        default=
            DEFAULT_FORMAL_RUN_ROOT,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    run_directory = (
        formal_run_directory(
            seed=
                args.seed,

            run_root=
                args.run_root,
        )
    )

    print("=" * 100)
    print("LOOPYCUTS FORMAL TRAINING RUNNER V1")
    print("=" * 100)

    print(
        "runner version          :",
        FORMAL_RUNNER_VERSION,
    )

    print(
        "seed                    :",
        args.seed,
    )

    print(
        "run directory           :",
        run_directory,
    )

    print(
        "resume                  :",
        args.resume,
    )

    print(
        "checkpoint interval     :",
        FORMAL_CHECKPOINT_INTERVAL_ENV_STEPS,
        "Stage-II transitions",
    )

    print(
        "Stage-II exact budget   :",
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS,
    )

    print()

    if args.resume:
        result = resume_formal_training(
            seed=
                args.seed,

            run_directory=
                run_directory,

            require_clean_git=
                True,
        )

    else:
        result = run_new_formal_training(
            seed=
                args.seed,

            run_directory=
                run_directory,

            require_clean_git=
                True,
        )

    print()
    print("=" * 100)
    print("FORMAL RUN RESULT")
    print("=" * 100)

    for key, value in result.items():
        print(
            f"{key:<24}:",
            value,
        )


if __name__ == "__main__":
    main()
