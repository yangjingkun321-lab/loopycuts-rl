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

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from training.formal_run_artifacts_v1 import (
    EVENT_LOG_FILENAME,
    FORMAL_RUN_ARTIFACTS_VERSION,
    FormalRunArtifactError,

    create_formal_run_artifacts,
    load_and_validate_run_manifest,
    read_formal_events,

    record_checkpoint,
    record_stage1_complete,
    record_stage2_episode,
)

from training.formal_training_v1 import (
    prepare_formal_training_core,
)


def make_episode(
    episode_index: int,
    *,
    action: int,
):
    return {
        "episode_index":
            episode_index,

        "model":
            "Plate3",

        "model_complexity_stratum":
            0,

        "curriculum_phase":
            "WARMUP",

        "eligible_model_count":
            39,

        "environment_steps_before":
            episode_index - 1,

        "mesh_file":
            "/tmp/mesh.obj",

        "loop_file":
            "/tmp/loop.txt",

        "completed":
            True,

        "budget_exhausted":
            False,

        "terminated":
            True,

        "truncated":
            False,

        "steps":
            1,

        "actions":
            [
                action
            ],

        "episode_return":
            1.25,

        "finalization_outcome":
            "FULL_HEX",

        "gradient_updates":
            1,

        "total_environment_steps":
            episode_index,

        "total_gradient_updates":
            episode_index,

        "expo_buffer_size":
            episode_index,

        "final_training_stats": {
            "actor_loss":
                -1.0,

            "critic1_loss":
                0.5,

            "critic2_loss":
                0.6,

            "alpha":
                0.9,

            "bc_loss":
                0.0,

            "bc_selected_count":
                0,
        },
    }


def main():
    core = prepare_formal_training_core(
        seed=42
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_directory = (
            Path(tmp)
            /
            "seed_42"
        )

        created = (
            create_formal_run_artifacts(
                run_directory=
                    run_directory,

                core=
                    core,

                # Test files are not committed yet.
                require_clean_git=
                    False,
            )
        )

        assert (
            FORMAL_RUN_ARTIFACTS_VERSION
            ==
            "loopycuts_formal_run_artifacts_v3_resource_guard"
        )

        manifest = (
            load_and_validate_run_manifest(
                run_directory=
                    run_directory,

                core=
                    core,

                strict_git=
                    False,
            )
        )

        assert (
            manifest[
                "seed"
            ]
            ==
            42
        )

        assert (
            manifest[
                "formal_training"
            ][
                "lambda_bc"
            ]
            ==
            3.0
        )

        assert (
            manifest[
                "formal_training"
            ][
                "stage2_model_count"
            ]
            ==
            49
        )

        assert (
            manifest[
                "formal_training"
            ][
                "stage2_environment_budget"
            ]
            ==
            25_000
        )

        assert (
            manifest[
                "formal_training"
            ][
                "stage2_curriculum_version"
            ]
            ==
            "complexity_curriculum_v1"
        )

        assert (
            manifest[
                "formal_training"
            ][
                "stage2_curriculum_warmup_env_steps"
            ]
            ==
            5_000
        )

        assert (
            manifest[
                "formal_training"
            ][
                "stage2_curriculum_warmup_max_stratum"
            ]
            ==
            7
        )

        assert (
            manifest[
                "formal_training"
            ][
                "stage2_curriculum_warmup_model_count"
            ]
            ==
            39
        )

        assert (
            manifest[
                "formal_training"
            ][
                "stage2_curriculum_full_model_count"
            ]
            ==
            49
        )


        # ============================================================
        # Stage-I event.
        # ============================================================

        core.stage1_updates_completed = 782

        core.stage1_sampled_demo_transitions = (
            50_048
        )

        stage1_result = {
            "alpha_after_stage1":
                0.8,

            "elapsed_seconds":
                10.0,

            "final_training_stats": {
                "actor_loss":
                    -2.0,

                "bc_loss":
                    0.0,

                "alpha":
                    0.8,
            },
        }

        first_stage1 = (
            record_stage1_complete(
                run_directory=
                    run_directory,

                core=
                    core,

                stage1_result=
                    stage1_result,
            )
        )

        assert (
            first_stage1[
                "replayed"
            ]
            is False
        )

        repeated_stage1 = (
            record_stage1_complete(
                run_directory=
                    run_directory,

                core=
                    core,

                stage1_result=
                    stage1_result,
            )
        )

        assert (
            repeated_stage1[
                "replayed"
            ]
            is True
        )


        # elapsed_seconds is telemetry only.  A deterministic Stage-I
        # replay may legitimately have a different wall-clock runtime.
        stage1_different_wallclock = copy.deepcopy(
            stage1_result
        )

        stage1_different_wallclock[
            "elapsed_seconds"
        ] = 12345.0

        wallclock_replay = (
            record_stage1_complete(
                run_directory=
                    run_directory,

                core=
                    core,

                stage1_result=
                    stage1_different_wallclock,
            )
        )

        assert (
            wallclock_replay[
                "replayed"
            ]
            is True
        )


        # ============================================================
        # Episode 1.
        # ============================================================

        episode1 = make_episode(
            1,
            action=1,
        )

        first_episode = (
            record_stage2_episode(
                run_directory=
                    run_directory,

                core=
                    core,

                record=
                    episode1,
            )
        )

        assert (
            first_episode[
                "replayed"
            ]
            is False
        )


        # Exact replay must not append a duplicate event.
        replay_episode = (
            record_stage2_episode(
                run_directory=
                    run_directory,

                core=
                    core,

                record=
                    copy.deepcopy(
                        episode1
                    ),
            )
        )

        assert (
            replay_episode[
                "replayed"
            ]
            is True
        )


        events_after_replay = (
            read_formal_events(
                event_log_path=
                    run_directory
                    /
                    EVENT_LOG_FILENAME
            )
        )

        assert (
            len(
                events_after_replay
            )
            ==
            2
        )


        # A divergent re-execution of an already logged episode must
        # be rejected.
        divergent = copy.deepcopy(
            episode1
        )

        divergent[
            "actions"
        ] = [
            7
        ]

        try:
            record_stage2_episode(
                run_directory=
                    run_directory,

                core=
                    core,

                record=
                    divergent,
            )

        except FormalRunArtifactError:
            pass

        else:
            raise AssertionError(
                "Divergent replayed Stage-II episode was not rejected"
            )


        # Episode gap must be rejected.
        try:
            record_stage2_episode(
                run_directory=
                    run_directory,

                core=
                    core,

                record=
                    make_episode(
                        3,
                        action=3,
                    ),
            )

        except FormalRunArtifactError:
            pass

        else:
            raise AssertionError(
                "Stage-II episode gap was not rejected"
            )


        # Episode 2 is then legal.
        episode2 = make_episode(
            2,
            action=0,
        )

        second_episode = (
            record_stage2_episode(
                run_directory=
                    run_directory,

                core=
                    core,

                record=
                    episode2,
            )
        )

        assert (
            second_episode[
                "replayed"
            ]
            is False
        )


        # ============================================================
        # Synthetic checkpoint metadata event.
        #
        # Checkpoint serialization itself was already verified by
        # Phase 18.4; this test checks only artifact logging.
        # ============================================================

        checkpoint_result = {
            "path":
                "/tmp/latest.pt",

            "bytes":
                123456,

            "sha256":
                "a"
                *
                64,

            "stage":
                "STAGE_I",

            "seed":
                42,
        }

        checkpoint_event = (
            record_checkpoint(
                run_directory=
                    run_directory,

                core=
                    core,

                stage2_state=
                    None,

                checkpoint_result=
                    checkpoint_result,
            )
        )

        assert (
            checkpoint_event[
                "replayed"
            ]
            is False
        )


        # ============================================================
        # Full hash-chain validation.
        # ============================================================

        events = (
            read_formal_events(
                event_log_path=
                    run_directory
                    /
                    EVENT_LOG_FILENAME
            )
        )

        assert (
            len(
                events
            )
            ==
            4
        )

        assert [
            event[
                "sequence"
            ]
            for event in
            events
        ] == [
            1,
            2,
            3,
            4,
        ]

        assert [
            event[
                "event_type"
            ]
            for event in
            events
        ] == [
            "STAGE1_COMPLETE",
            "STAGE2_EPISODE",
            "STAGE2_EPISODE",
            "CHECKPOINT",
        ]


        # ============================================================
        # Tamper detection.
        # ============================================================

        original_log = (
            run_directory
            /
            EVENT_LOG_FILENAME
        )

        bad_log = (
            run_directory
            /
            "tampered.jsonl"
        )

        lines = (
            original_log
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )

        first = json.loads(
            lines[
                0
            ]
        )

        first[
            "payload"
        ][
            "gradient_updates"
        ] = 999

        lines[
            0
        ] = json.dumps(
            first,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )

        bad_log.write_text(
            "\n".join(
                lines
            )
            +
            "\n",
            encoding="utf-8",
        )

        try:
            read_formal_events(
                event_log_path=
                    bad_log
            )

        except FormalRunArtifactError:
            pass

        else:
            raise AssertionError(
                "Tampered formal event log was not rejected"
            )


        print("=" * 100)
        print("FORMAL RUN ARTIFACTS V3 RESOURCE GUARD")
        print("=" * 100)

        print(
            "manifest SHA256       :",
            created[
                "manifest_sha256"
            ],
        )

        print(
            "event count           :",
            len(
                events
            ),
        )

        print(
            "event types           :",
            [
                event[
                    "event_type"
                ]
                for event in
                events
            ],
        )

        print(
            "last event SHA256     :",
            events[
                -1
            ][
                "event_sha256"
            ],
        )

        print()

        print(
            "PASS: formal run manifest freezes seed/protocol/provenance/runtime"
        )

        print(
            "PASS: formal JSONL event sequence forms a validated SHA256 hash chain"
        )

        print(
            "PASS: repeated identical Stage-I and Stage-II records are idempotent"
        )

        print(
            "PASS: divergent replayed Stage-II episode is rejected"
        )

        print(
            "PASS: Stage-II episode gaps are rejected"
        )

        print(
            "PASS: tampering with an existing event is detected"
        )


if __name__ == "__main__":
    main()
