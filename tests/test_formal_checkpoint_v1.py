from __future__ import annotations

import copy
import random
import sys
import tempfile

from pathlib import Path


import numpy as np
import torch

from tianshou.utils.torch_utils import (
    policy_within_training_step,
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
    FORMAL_CHECKPOINT_VERSION,
    capture_global_rng_state,
    load_formal_checkpoint,
    restore_global_rng_state,
    save_formal_checkpoint,
)

from training.formal_training_v1 import (
    collect_formal_stage2_model_episode,
    enter_formal_stage2,
    formal_stage2_curriculum_phase,
    prepare_formal_stage2_state,
    prepare_formal_training_core,
    sample_formal_stage2_model,
    training_stats_snapshot,
)

from training.protocol_v1 import (
    PROJECT_STAGE1_GRADIENT_STEPS,
    PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS,
    PROJECT_STAGE2_SAMPLES_PER_BUFFER,
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
            left_value,
            right_value,
        ) in enumerate(
            zip(
                left,
                right,
            )
        ):
            assert_nested_equal(
                left_value,
                right_value,
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


def main():
    core = prepare_formal_training_core(
        seed=42
    )

    # Phase 18.2 already executed the exact 782-update integration
    # smoke. Do not repeat those ~9 minutes here.
    core.stage1_updates_completed = (
        PROJECT_STAGE1_GRADIENT_STEPS
    )

    core.stage1_sampled_demo_transitions = (
        PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS
    )

    enter_formal_stage2(
        core
    )

    stage2 = prepare_formal_stage2_state(
        core
    )


    # Advance the independent Train49 sampler once so checkpointing
    # has a nontrivial model RNG state.
    first_sample = (
        sample_formal_stage2_model(
            stage2
        )
    )

    assert (
        first_sample.model
        ==
        "blade"
    )


    # Small real Stage-II episode to populate:
    #
    # D_expo
    # policy AdamW
    # critics AdamW
    # target critics
    # alpha + alpha Adam
    plate3 = next(
        model
        for model in
        stage2.models
        if model.model == "Plate3"
    )

    record = (
        collect_formal_stage2_model_episode(
            core,
            stage2,

            model=
                plate3,
        )
    )

    assert record[
        "steps"
    ] == 2

    assert record[
        "actions"
    ] == [
        1,
        0,
    ]

    assert record[
        "finalization_outcome"
    ] == "FULL_HEX"

    assert (
        record[
            "curriculum_phase"
        ]
        ==
        "WARMUP"
    )

    assert (
        record[
            "eligible_model_count"
        ]
        ==
        39
    )

    assert (
        record[
            "model_complexity_stratum"
        ]
        ==
        0
    )

    assert (
        formal_stage2_curriculum_phase(
            stage2
        )
        ==
        "WARMUP"
    )

    assert (
        stage2.total_environment_steps
        ==
        2
    )

    assert (
        stage2.total_gradient_updates
        ==
        2
    )

    assert (
        len(
            stage2.expo_buffer
        )
        ==
        2
    )


    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = (
            Path(tmp)
            /
            "formal_checkpoint_v1.pt"
        )

        save_result = (
            save_formal_checkpoint(
                checkpoint_path=
                    checkpoint,

                core=
                    core,

                stage2_state=
                    stage2,

                # Development test runs from a dirty worktree because
                # the checkpoint module itself is not committed yet.
                require_clean_git=
                    False,
            )
        )


        assert (
            FORMAL_CHECKPOINT_VERSION
            ==
            "loopycuts_formal_checkpoint_v4_cpp_rss_compat"
        )

        assert checkpoint.is_file()

        assert (
            save_result[
                "bytes"
            ]
            >
            0
        )


        # State at exact checkpoint boundary.
        algorithm_at_checkpoint = (
            copy.deepcopy(
                core.algorithm.state_dict()
            )
        )

        alpha_optim_at_checkpoint = (
            copy.deepcopy(
                core.auto_alpha
                ._optim
                .state_dict()
            )
        )

        alpha_last_at_checkpoint = (
            copy.deepcopy(
                core.auto_alpha.last_update
            )
        )

        history_at_checkpoint = (
            copy.deepcopy(
                stage2.history
            )
        )


        # ============================================================
        # Advance every independent non-global RNG AFTER checkpoint.
        #
        # The restored objects must reproduce these exact next draws.
        # ============================================================

        expected_next_model = (
            sample_formal_stage2_model(
                stage2
            )
            .model
        )

        expected_demo_indices = (
            core.demo_buffer
            .sample_indices(
                16
            )
            .copy()
        )

        expected_expo_indices = (
            stage2.expo_buffer
            .sample_indices(
                16
            )
            .copy()
        )

        expected_exploration_random = (
            core.policy
            ._exploration_rng
            .random(
                16
            )
            .copy()
        )


        # ============================================================
        # Fresh trainer process-equivalent reconstruction.
        # ============================================================

        (
            restored_core,
            restored_stage2,
            load_result,
        ) = load_formal_checkpoint(
            checkpoint_path=
                checkpoint,

            # Same reason as require_clean_git=False above.
            strict_git=
                False,
        )


        assert (
            restored_stage2
            is not
            None
        )

        assert (
            restored_core.stage
            ==
            "STAGE_II"
        )

        assert (
            restored_core.algorithm.bc_enabled
            is False
        )

        assert (
            formal_stage2_curriculum_phase(
                restored_stage2
            )
            ==
            "WARMUP"
        )

        assert (
            restored_stage2.total_environment_steps
            ==
            2
        )

        assert (
            restored_stage2.total_gradient_updates
            ==
            2
        )

        assert (
            len(
                restored_stage2.expo_buffer
            )
            ==
            2
        )

        assert (
            restored_stage2.history
            ==
            history_at_checkpoint
        )


        # ============================================================
        # Model / target / optimizer / alpha exact checkpoint state.
        # ============================================================

        assert_nested_equal(
            algorithm_at_checkpoint,
            restored_core
            .algorithm
            .state_dict(),
            path=
                "algorithm_state",
        )

        assert_nested_equal(
            alpha_optim_at_checkpoint,
            restored_core
            .auto_alpha
            ._optim
            .state_dict(),
            path=
                "alpha_optimizer",
        )

        assert (
            restored_core.auto_alpha.last_update
            ==
            alpha_last_at_checkpoint
        )


        # ============================================================
        # Independent RNG continuity.
        # ============================================================

        observed_next_model = (
            sample_formal_stage2_model(
                restored_stage2
            )
            .model
        )

        assert (
            observed_next_model
            ==
            expected_next_model
        )


        observed_demo_indices = (
            restored_core
            .demo_buffer
            .sample_indices(
                16
            )
        )

        assert np.array_equal(
            observed_demo_indices,
            expected_demo_indices,
        )


        observed_expo_indices = (
            restored_stage2
            .expo_buffer
            .sample_indices(
                16
            )
        )

        assert np.array_equal(
            observed_expo_indices,
            expected_expo_indices,
        )


        observed_exploration_random = (
            restored_core
            .policy
            ._exploration_rng
            .random(
                16
            )
        )

        assert np.array_equal(
            observed_exploration_random,
            expected_exploration_random,
        )


        # ============================================================
        # Strong next-update equivalence.
        #
        # Both original and restored replay RNGs are now in identical
        # post-audit states.
        #
        # Use the same global RNG state for each branch so SAC's
        # categorical sampling receives identical torch randomness.
        # ============================================================

        branch_global_rng = (
            capture_global_rng_state()
        )


        with policy_within_training_step(
            core.policy
        ):
            stats_original, mix_original = (
                core.algorithm.update_equal_replay(
                    demo_buffer=
                        core.demo_buffer,

                    expo_buffer=
                        stage2.expo_buffer,

                    samples_per_buffer=
                        PROJECT_STAGE2_SAMPLES_PER_BUFFER,
                )
            )


        original_snapshot = (
            training_stats_snapshot(
                stats_original
            )
        )

        original_algorithm_after = (
            copy.deepcopy(
                core.algorithm.state_dict()
            )
        )

        original_alpha_optim_after = (
            copy.deepcopy(
                core.auto_alpha
                ._optim
                .state_dict()
            )
        )


        # Rewind GLOBAL RNG only.
        #
        # The restored replay/model/exploration RNGs are independent
        # objects and already match the original branch's pre-update
        # positions.
        restore_global_rng_state(
            branch_global_rng
        )


        with policy_within_training_step(
            restored_core.policy
        ):
            stats_restored, mix_restored = (
                restored_core
                .algorithm
                .update_equal_replay(
                    demo_buffer=
                        restored_core.demo_buffer,

                    expo_buffer=
                        restored_stage2.expo_buffer,

                    samples_per_buffer=
                        PROJECT_STAGE2_SAMPLES_PER_BUFFER,
                )
            )


        restored_snapshot = (
            training_stats_snapshot(
                stats_restored
            )
        )


        assert (
            mix_original
            ==
            mix_restored
        )

        assert (
            original_snapshot
            ==
            restored_snapshot
        )


        assert_nested_equal(
            original_algorithm_after,
            restored_core
            .algorithm
            .state_dict(),
            path=
                "post_resume_next_update_algorithm",
        )

        assert_nested_equal(
            original_alpha_optim_after,
            restored_core
            .auto_alpha
            ._optim
            .state_dict(),
            path=
                "post_resume_next_update_alpha_optimizer",
        )


        # Replay sampling RNGs must also land in the same state after
        # the equivalent update.
        assert_nested_equal(
            core.demo_buffer
            ._random_state
            .get_state(),

            restored_core.demo_buffer
            ._random_state
            .get_state(),

            path=
                "post_update_demo_rng",
        )

        assert_nested_equal(
            stage2.expo_buffer
            ._random_state
            .get_state(),

            restored_stage2
            .expo_buffer
            ._random_state
            .get_state(),

            path=
                "post_update_expo_rng",
        )


        print("=" * 100)
        print("FORMAL CHECKPOINT V4 CPP RSS COMPAT ROUND-TRIP")
        print("=" * 100)

        print(
            "checkpoint bytes       :",
            save_result[
                "bytes"
            ],
        )

        print(
            "checkpoint SHA256      :",
            save_result[
                "sha256"
            ],
        )

        print(
            "stage                  :",
            restored_core.stage,
        )

        print(
            "D_demo                 :",
            len(
                restored_core.demo_buffer
            ),
        )

        print(
            "D_expo                 :",
            len(
                restored_stage2.expo_buffer
            ),
        )

        print(
            "environment steps      :",
            restored_stage2.total_environment_steps,
        )

        print(
            "gradient updates       :",
            restored_stage2.total_gradient_updates,
        )

        print(
            "next sampled model     :",
            observed_next_model,
        )

        print(
            "next-update actor loss :",
            restored_snapshot.get(
                "actor_loss"
            ),
        )

        print(
            "next-update alpha      :",
            restored_snapshot.get(
                "alpha"
            ),
        )

        print()

        print(
            "PASS: atomic formal checkpoint save/load round-trip succeeds"
        )

        print(
            "PASS: algorithm/targets/AdamW states restore exactly"
        )

        print(
            "PASS: masked alpha parameter and alpha Adam state restore exactly"
        )

        print(
            "PASS: frozen D_demo is reconstructed with identical replay RNG state"
        )

        print(
            "PASS: compact D_expo transitions reconstruct exact replay execution state"
        )

        print(
            "PASS: Train49 sampler and masked epsilon RNG states restore exactly"
        )

        print(
            "PASS: resumed next SAC update is bitwise-equivalent to uninterrupted training"
        )


if __name__ == "__main__":
    main()
