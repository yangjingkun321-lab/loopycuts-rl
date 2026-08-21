from __future__ import annotations

import copy
import hashlib
import os
import pickle
import random
import subprocess
import sys

from pathlib import Path
from typing import Any


import numpy as np
import torch
import tianshou

from tianshou.data import (
    ReplayBuffer,
)


from training.formal_training_v1 import (
    DEFAULT_DATASET_MANIFEST,
    DEFAULT_DEMO_QUALITY,
    DEFAULT_EXECUTABLE,
    DEFAULT_FORMAL_INPUT_PROVENANCE,
    DEFAULT_RAW_DEMO_ROOT,

    FORMAL_STAGE2_ONLINE_VERSION,
    FORMAL_TRAINER_CORE_VERSION,

    FormalStage2StateV1,
    FormalTrainingCoreError,
    FormalTrainingCoreV1,

    formal_stage2_curriculum_phase,
    prepare_formal_stage2_state,
    prepare_formal_training_core,
)

from training.protocol_v1 import (
    PROJECT_STAGE2_RESOURCE_GUARD_VERSION,
    PROJECT_STAGE2_RESOURCE_GUARD_SAMPLE_INTERVAL_SECONDS,
    PROJECT_STAGE2_RESOURCE_GUARD_WARNING_SWAP_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_ABORT_SWAP_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_ABORT_HOLD_SECONDS,
    PROJECT_STAGE2_RESOURCE_GUARD_EMERGENCY_SWAP_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_FINALIZE_EVAL_SWAP_ABORT_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_INITIALIZE_ENABLED,
    PROJECT_STAGE2_RESOURCE_GUARD_FINALIZE_EVAL_ADDS_TRANSITION,
    PROJECT_STAGE2_RESOURCE_GUARD_REARM_SWAP_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_REARM_TIMEOUT_SECONDS,
    PROJECT_STAGE2_RESOURCE_GUARD_ABORT_SCOPE,
    PROJECT_STAGE2_RESOURCE_GUARD_TERMINAL_OUTCOME,
    PROJECT_STAGE2_RESOURCE_GUARD_TERMINAL_REWARD,
    PROJECT_STAGE2_RESOURCE_GUARD_ABORT_COUNTS_AS_TRANSITION,
    PROJECT_STAGE2_RESOURCE_GUARD_DENSE_SHAPING_ENABLED,
    PROJECT_STAGE2_RESOURCE_GUARD_IMMEDIATE_CHECKPOINT,
    PROJECT_STAGE2_RESOURCE_GUARD_PREFLIGHT_REARM_REQUIRED,
    PROJECT_STAGE2_RESOURCE_GUARD_COLLECTOR_AUTORESET_POLICY,

    PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_COMPAT_ENABLED,
    PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_PHASE,
    PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_SIGNAL,
    PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_SIGNATURE,
    PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_GUARD_STATE,

    PROTOCOL_VERSION,

    PROJECT_BC_WEIGHT,

    PROJECT_STAGE1_GRADIENT_STEPS,
    PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS,

    PROJECT_STAGE2_EXPO_REPLAY_CAPACITY,
    PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS,
)


FORMAL_CHECKPOINT_VERSION = (
    "loopycuts_formal_checkpoint_v4_cpp_rss_compat"
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


class FormalCheckpointError(
    RuntimeError
):
    pass


def formal_resource_guard_contract():
    return {
        "version":
            PROJECT_STAGE2_RESOURCE_GUARD_VERSION,

        "sample_interval_seconds":
            float(
                PROJECT_STAGE2_RESOURCE_GUARD_SAMPLE_INTERVAL_SECONDS
            ),

        "warning_swap_gib":
            int(
                PROJECT_STAGE2_RESOURCE_GUARD_WARNING_SWAP_GIB
            ),

        "abort_swap_gib":
            int(
                PROJECT_STAGE2_RESOURCE_GUARD_ABORT_SWAP_GIB
            ),

        "abort_hold_seconds":
            float(
                PROJECT_STAGE2_RESOURCE_GUARD_ABORT_HOLD_SECONDS
            ),

        "emergency_swap_gib":
            int(
                PROJECT_STAGE2_RESOURCE_GUARD_EMERGENCY_SWAP_GIB
            ),

        "finalize_eval_swap_abort_gib":
            int(
                PROJECT_STAGE2_RESOURCE_GUARD_FINALIZE_EVAL_SWAP_ABORT_GIB
            ),

        "initialize_guard_enabled":
            bool(
                PROJECT_STAGE2_RESOURCE_GUARD_INITIALIZE_ENABLED
            ),

        "finalize_eval_adds_transition":
            bool(
                PROJECT_STAGE2_RESOURCE_GUARD_FINALIZE_EVAL_ADDS_TRANSITION
            ),

        "rearm_swap_gib":
            int(
                PROJECT_STAGE2_RESOURCE_GUARD_REARM_SWAP_GIB
            ),

        "rearm_timeout_seconds":
            float(
                PROJECT_STAGE2_RESOURCE_GUARD_REARM_TIMEOUT_SECONDS
            ),

        "abort_scope":
            PROJECT_STAGE2_RESOURCE_GUARD_ABORT_SCOPE,

        "terminal_outcome":
            PROJECT_STAGE2_RESOURCE_GUARD_TERMINAL_OUTCOME,

        "terminal_reward":
            float(
                PROJECT_STAGE2_RESOURCE_GUARD_TERMINAL_REWARD
            ),

        "abort_counts_as_transition":
            bool(
                PROJECT_STAGE2_RESOURCE_GUARD_ABORT_COUNTS_AS_TRANSITION
            ),

        "dense_shaping_enabled":
            bool(
                PROJECT_STAGE2_RESOURCE_GUARD_DENSE_SHAPING_ENABLED
            ),

        "immediate_checkpoint":
            bool(
                PROJECT_STAGE2_RESOURCE_GUARD_IMMEDIATE_CHECKPOINT
            ),

        "preflight_rearm_required":
            bool(
                PROJECT_STAGE2_RESOURCE_GUARD_PREFLIGHT_REARM_REQUIRED
            ),

        "collector_autoreset_policy":
            PROJECT_STAGE2_RESOURCE_GUARD_COLLECTOR_AUTORESET_POLICY,

        "cpp_rss_assert_compat_enabled":
            bool(
                PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_COMPAT_ENABLED
            ),

        "cpp_rss_assert_phase":
            PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_PHASE,

        "cpp_rss_assert_signal":
            PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_SIGNAL,

        "cpp_rss_assert_signature":
            PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_SIGNATURE,

        "cpp_rss_assert_guard_state":
            PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_GUARD_STATE,
    }


# ======================================================================
# Git / file identity.
# ======================================================================


def current_git_head() -> str:
    return (
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            text=True,
        )
        .strip()
    )


def repository_is_dirty() -> bool:
    output = (
        subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
            ],
            cwd=PROJECT_ROOT,
            text=True,
        )
    )

    return bool(
        output.strip()
    )


def sha256_file(
    path: Path,
) -> str:
    path = Path(
        path
    )

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


# ======================================================================
# Global RNG state.
# ======================================================================


def capture_global_rng_state():
    return {
        "python_random":
            copy.deepcopy(
                random.getstate()
            ),

        "numpy_global":
            copy.deepcopy(
                np.random.get_state()
            ),

        "torch_cpu":
            torch.get_rng_state()
            .clone(),
    }


def restore_global_rng_state(
    state,
):
    random.setstate(
        state[
            "python_random"
        ]
    )

    np.random.set_state(
        state[
            "numpy_global"
        ]
    )

    torch.set_rng_state(
        state[
            "torch_cpu"
        ]
    )


# ======================================================================
# Replay snapshots.
#
# D_demo:
#   data are immutable and already frozen by provenance.
#   Save only replay execution state / RNG.
#
# D_expo:
#   data are unique online experience, therefore save all USED
#   transitions, but deliberately do not serialize unused capacity.
# ======================================================================


def snapshot_demo_replay(
    buffer,
):
    return {
        "length":
            len(
                buffer
            ),

        "maxsize":
            int(
                buffer.maxsize
            ),

        "stack_num":
            int(
                buffer.stack_num
            ),

        "random_state":
            copy.deepcopy(
                buffer
                ._random_state
                .get_state()
            ),
    }


def restore_demo_replay_state(
    buffer,
    snapshot,
):
    if (
        len(
            buffer
        )
        !=
        int(
            snapshot[
                "length"
            ]
        )
    ):
        raise FormalCheckpointError(
            "Restored D_demo length mismatch"
        )

    if (
        int(
            buffer.maxsize
        )
        !=
        int(
            snapshot[
                "maxsize"
            ]
        )
    ):
        raise FormalCheckpointError(
            "Restored D_demo capacity mismatch"
        )

    if (
        int(
            buffer.stack_num
        )
        !=
        int(
            snapshot[
                "stack_num"
            ]
        )
    ):
        raise FormalCheckpointError(
            "Restored D_demo stack_num mismatch"
        )

    buffer._random_state.set_state(
        snapshot[
            "random_state"
        ]
    )


def snapshot_expo_replay(
    buffer,
):
    size = len(
        buffer
    )

    indices = np.asarray(
        buffer.sample_indices(
            0
        ),
        dtype=np.int64,
    )

    expected = np.arange(
        size,
        dtype=np.int64,
    )

    # D_expo capacity equals the exact global training budget and
    # therefore is never overwritten during a formal run.
    if not np.array_equal(
        indices,
        expected,
    ):
        raise FormalCheckpointError(
            "Formal D_expo indices are not in expected append-only order"
        )

    if size == 0:
        data = None

    else:
        # Advanced indexing + deepcopy ensures the checkpoint owns only
        # the USED transitions, not the entire 25,000-slot backing store.
        data = copy.deepcopy(
            buffer[
                indices
            ]
        )

    return {
        "length":
            size,

        "maxsize":
            int(
                buffer.maxsize
            ),

        "stack_num":
            int(
                buffer.stack_num
            ),

        "data":
            data,

        "random_state":
            copy.deepcopy(
                buffer
                ._random_state
                .get_state()
            ),

        "insertion_idx":
            int(
                buffer._insertion_idx
            ),

        "ep_return":
            copy.deepcopy(
                buffer._ep_return
            ),

        "ep_len":
            int(
                buffer._ep_len
            ),

        "ep_start_idx":
            int(
                buffer._ep_start_idx
            ),

        "last_index":
            np.asarray(
                buffer.last_index
            ).copy(),
    }


def restore_expo_replay(
    *,
    snapshot,
    seed: int,
):
    if (
        int(
            snapshot[
                "maxsize"
            ]
        )
        !=
        PROJECT_STAGE2_EXPO_REPLAY_CAPACITY
        or
        PROJECT_STAGE2_EXPO_REPLAY_CAPACITY
        !=
        25_000
    ):
        raise FormalCheckpointError(
            "Checkpoint D_expo capacity mismatch"
        )

    buffer = ReplayBuffer(
        size=
            PROJECT_STAGE2_EXPO_REPLAY_CAPACITY,

        random_seed=
            int(
                seed
            ),
    )

    data = snapshot[
        "data"
    ]

    expected_length = int(
        snapshot[
            "length"
        ]
    )

    if expected_length == 0:
        if data is not None:
            raise FormalCheckpointError(
                "Empty D_expo snapshot unexpectedly contains data"
            )

    else:
        if data is None:
            raise FormalCheckpointError(
                "Non-empty D_expo snapshot lacks transition data"
            )

        if (
            len(
                data
            )
            !=
            expected_length
        ):
            raise FormalCheckpointError(
                "Compact D_expo transition count mismatch"
            )

        # Replay transitions in chronological order.
        #
        # This reconstructs not only _meta, but also Tianshou's
        # unfinished-episode bookkeeping:
        #
        #   _ep_return
        #   _ep_len
        #   _ep_start_idx
        #
        # Therefore final 25k checkpoints are valid even if the exact
        # budget boundary lies inside a nonterminal episode.
        for index in range(
            expected_length
        ):
            buffer.add(
                data[
                    index
                ]
            )

    if (
        len(
            buffer
        )
        !=
        expected_length
    ):
        raise FormalCheckpointError(
            "Reconstructed D_expo length mismatch"
        )

    if (
        int(
            buffer._insertion_idx
        )
        !=
        int(
            snapshot[
                "insertion_idx"
            ]
        )
    ):
        raise FormalCheckpointError(
            "Reconstructed D_expo insertion index mismatch"
        )

    if (
        int(
            buffer._ep_len
        )
        !=
        int(
            snapshot[
                "ep_len"
            ]
        )
    ):
        raise FormalCheckpointError(
            "Reconstructed D_expo episode-length state mismatch"
        )

    if (
        int(
            buffer._ep_start_idx
        )
        !=
        int(
            snapshot[
                "ep_start_idx"
            ]
        )
    ):
        raise FormalCheckpointError(
            "Reconstructed D_expo episode-start state mismatch"
        )

    if not np.array_equal(
        np.asarray(
            buffer.last_index
        ),
        np.asarray(
            snapshot[
                "last_index"
            ]
        ),
    ):
        raise FormalCheckpointError(
            "Reconstructed D_expo last_index mismatch"
        )

    observed_ep_return = np.asarray(
        buffer._ep_return
    )

    expected_ep_return = np.asarray(
        snapshot[
            "ep_return"
        ]
    )

    if not np.array_equal(
        observed_ep_return,
        expected_ep_return,
    ):
        raise FormalCheckpointError(
            "Reconstructed D_expo episode-return state mismatch"
        )

    buffer._random_state.set_state(
        snapshot[
            "random_state"
        ]
    )

    return buffer


# ======================================================================
# Checkpoint-boundary validation.
# ======================================================================


def assert_checkpoint_boundary(
    core: FormalTrainingCoreV1,
    stage2_state: FormalStage2StateV1 | None,
):
    if (
        core.policy.is_within_training_step
    ):
        raise FormalCheckpointError(
            "Cannot checkpoint while policy is inside a training step"
        )

    if (
        core.stage
        ==
        "STAGE_I"
    ):
        if stage2_state is not None:
            raise FormalCheckpointError(
                "STAGE_I checkpoint must not contain Stage-II state"
            )

        if (
            core.stage1_updates_completed
            !=
            PROJECT_STAGE1_GRADIENT_STEPS
        ):
            raise FormalCheckpointError(
                "STAGE_I checkpoint is allowed only after exact Stage-I completion"
            )

        if (
            core.stage1_sampled_demo_transitions
            !=
            PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS
        ):
            raise FormalCheckpointError(
                "STAGE_I demonstration budget is incomplete"
            )

        if (
            not
            core.algorithm.bc_enabled
        ):
            raise FormalCheckpointError(
                "Completed STAGE_I checkpoint must still have BC enabled"
            )

        return


    if (
        core.stage
        !=
        "STAGE_II"
    ):
        raise FormalCheckpointError(
            f"Unknown formal training stage: {core.stage!r}"
        )

    if stage2_state is None:
        raise FormalCheckpointError(
            "STAGE_II checkpoint requires Stage-II state"
        )

    if (
        core.algorithm.bc_enabled
    ):
        raise FormalCheckpointError(
            "STAGE_II checkpoint cannot have BC enabled"
        )

    if (
        stage2_state.total_environment_steps
        !=
        stage2_state.total_gradient_updates
    ):
        raise FormalCheckpointError(
            "STAGE_II checkpoint requires fully flushed updates"
        )

    if (
        len(
            stage2_state.expo_buffer
        )
        !=
        stage2_state.total_environment_steps
    ):
        raise FormalCheckpointError(
            "D_expo size does not match Stage-II collection counter"
        )

    if (
        stage2_state.total_environment_steps
        >
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
    ):
        raise FormalCheckpointError(
            "Stage-II checkpoint exceeds exact environment budget"
        )

    # Tianshou ReplayBuffer.unfinished_index() assumes that the
    # replay metadata already contains a "done" field.  A freshly
    # constructed empty ReplayBuffer has no transition schema yet, so
    # calling unfinished_index() at size 0 raises AttributeError.
    #
    # Semantically an empty D_expo cannot contain an unfinished
    # episode, therefore size 0 is directly an episode-safe boundary.
    if (
        len(
            stage2_state.expo_buffer
        )
        ==
        0
    ):
        unfinished = np.asarray(
            [],
            dtype=np.int64,
        )

    else:
        unfinished = np.asarray(
            stage2_state
            .expo_buffer
            .unfinished_index()
        )

    # Before the final exact budget, checkpoint only at a real native
    # LoopyCuts episode boundary. We deliberately never serialize a
    # live C++ environment.
    if (
        stage2_state.total_environment_steps
        <
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
        and
        unfinished.size
        !=
        0
    ):
        raise FormalCheckpointError(
            "Intermediate Stage-II checkpoint is not at an episode boundary"
        )

    # Exactly 25,000 transitions may legally end in a partial episode.
    # That prefix is final and does not need a live C++ state to resume
    # further collection.


# ======================================================================
# Payload creation.
# ======================================================================


def build_formal_checkpoint_payload(
    core: FormalTrainingCoreV1,
    stage2_state: FormalStage2StateV1 | None,
):
    assert_checkpoint_boundary(
        core,
        stage2_state,
    )

    stage2_payload = None

    if stage2_state is not None:
        stage2_payload = {
            "seed":
                int(
                    stage2_state.seed
                ),

            "total_environment_steps":
                int(
                    stage2_state.total_environment_steps
                ),

            "total_gradient_updates":
                int(
                    stage2_state.total_gradient_updates
                ),

            "episode_attempts":
                int(
                    stage2_state.episode_attempts
                ),

            "completed_episodes":
                int(
                    stage2_state.completed_episodes
                ),

            "history":
                copy.deepcopy(
                    stage2_state.history
                ),

            "model_names":
                tuple(
                    model.model
                    for model in
                    stage2_state.models
                ),

            "model_complexity_strata":
                tuple(
                    int(
                        model.complexity_stratum
                    )
                    for model in
                    stage2_state.models
                ),

            "curriculum_phase":
                formal_stage2_curriculum_phase(
                    stage2_state
                ),

            "model_rng_state":
                copy.deepcopy(
                    stage2_state
                    .model_rng
                    .bit_generator
                    .state
                ),

            "expo_replay":
                snapshot_expo_replay(
                    stage2_state.expo_buffer
                ),
        }

    payload = {
        "schema_version":
            FORMAL_CHECKPOINT_VERSION,

        "protocol_version":
            PROTOCOL_VERSION,

        "trainer_core_version":
            FORMAL_TRAINER_CORE_VERSION,

        "stage2_online_version":
            FORMAL_STAGE2_ONLINE_VERSION,

        "resource_guard_contract":
            formal_resource_guard_contract(),

        "repository": {
            "head":
                current_git_head(),

            "dirty":
                repository_is_dirty(),
        },

        "software": {
            "python":
                sys.version,

            "numpy":
                np.__version__,

            "torch":
                torch.__version__,

            "tianshou":
                tianshou.__version__,
        },

        "input_provenance": {
            "manifest_sha256":
                core.input_provenance[
                    "manifest_sha256"
                ],

            "train49_aggregate_sha256":
                core.input_provenance[
                    "train49_aggregate_sha256"
                ],

            "selected_bc_weight":
                core.input_provenance[
                    "selected_bc_weight"
                ],
        },

        "core": {
            "seed":
                int(
                    core.seed
                ),

            "stage":
                str(
                    core.stage
                ),

            "stage1_updates_completed":
                int(
                    core.stage1_updates_completed
                ),

            "stage1_sampled_demo_transitions":
                int(
                    core.stage1_sampled_demo_transitions
                ),

            "bc_enabled":
                bool(
                    core.algorithm.bc_enabled
                ),

            "bc_weight":
                float(
                    core.algorithm.bc_weight
                ),

            "policy_deterministic_eval":
                bool(
                    core.policy.deterministic_eval
                ),

            "policy_exploration_epsilon":
                float(
                    core.policy.exploration_epsilon
                ),
        },

        # Tianshou Algorithm.state_dict() includes the registered
        # modules AND its policy/critic optimizer states.
        "algorithm_state":
            copy.deepcopy(
                core.algorithm.state_dict()
            ),

        # MaskedAutoAlphaV1 is a module, therefore _log_alpha itself is
        # present in algorithm_state. Its private Adam optimizer is not
        # one of Algorithm._optimizers, so save it explicitly.
        "alpha_optimizer_state":
            copy.deepcopy(
                core.auto_alpha
                ._optim
                .state_dict()
            ),

        "auto_alpha_last_update":
            copy.deepcopy(
                core.auto_alpha.last_update
            ),

        "demo_replay":
            snapshot_demo_replay(
                core.demo_buffer
            ),

        "policy_exploration_rng_state":
            copy.deepcopy(
                core.policy
                ._exploration_rng
                .bit_generator
                .state
            ),

        "stage2":
            stage2_payload,

        # Capture these after payload construction has completed.
        # None of the operations above intentionally consumes them.
        "global_rng_state":
            capture_global_rng_state(),
    }

    return payload


# ======================================================================
# Atomic save.
# ======================================================================


def save_formal_checkpoint(
    *,
    checkpoint_path: Path,
    core: FormalTrainingCoreV1,
    stage2_state: FormalStage2StateV1 | None,
    require_clean_git: bool = True,
):
    checkpoint_path = Path(
        checkpoint_path
    ).resolve()

    if (
        require_clean_git
        and
        repository_is_dirty()
    ):
        raise FormalCheckpointError(
            "Formal checkpoint requires a clean Git worktree"
        )

    payload = build_formal_checkpoint_payload(
        core,
        stage2_state,
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = checkpoint_path.with_name(
        checkpoint_path.name
        +
        f".tmp.{os.getpid()}"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    try:
        with temporary_path.open(
            "wb"
        ) as f:
            torch.save(
                payload,
                f,
                pickle_protocol=
                    pickle.HIGHEST_PROTOCOL,
            )

            f.flush()

            os.fsync(
                f.fileno()
            )

        os.replace(
            temporary_path,
            checkpoint_path,
        )

        # Persist the directory entry itself.
        directory_fd = os.open(
            str(
                checkpoint_path.parent
            ),
            os.O_DIRECTORY,
        )

        try:
            os.fsync(
                directory_fd
            )

        finally:
            os.close(
                directory_fd
            )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

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
            core.seed,
    }


# ======================================================================
# Load / restore.
# ======================================================================


def _validate_software(
    payload,
):
    expected = payload[
        "software"
    ]

    observed = {
        "python":
            sys.version,

        "numpy":
            np.__version__,

        "torch":
            torch.__version__,

        "tianshou":
            tianshou.__version__,
    }

    if (
        observed
        !=
        expected
    ):
        raise FormalCheckpointError(
            "Checkpoint software/runtime version mismatch: "
            f"expected={expected}, "
            f"observed={observed}"
        )


def load_formal_checkpoint(
    *,
    checkpoint_path: Path,

    executable: Path =
        DEFAULT_EXECUTABLE,

    dataset_manifest: Path =
        DEFAULT_DATASET_MANIFEST,

    demo_quality_manifest: Path =
        DEFAULT_DEMO_QUALITY,

    raw_demo_root: Path =
        DEFAULT_RAW_DEMO_ROOT,

    input_provenance_path: Path =
        DEFAULT_FORMAL_INPUT_PROVENANCE,

    strict_git: bool = True,
):
    checkpoint_path = Path(
        checkpoint_path
    ).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    payload = torch.load(
        checkpoint_path,
        map_location=
            "cpu",
    )

    if (
        payload.get(
            "schema_version"
        )
        !=
        FORMAL_CHECKPOINT_VERSION
    ):
        raise FormalCheckpointError(
            "Checkpoint schema version mismatch"
        )

    if (
        payload.get(
            "protocol_version"
        )
        !=
        PROTOCOL_VERSION
    ):
        raise FormalCheckpointError(
            "Training Protocol version mismatch"
        )

    if (
        payload.get(
            "trainer_core_version"
        )
        !=
        FORMAL_TRAINER_CORE_VERSION
    ):
        raise FormalCheckpointError(
            "Formal Trainer Core version mismatch"
        )

    if (
        payload.get(
            "stage2_online_version"
        )
        !=
        FORMAL_STAGE2_ONLINE_VERSION
    ):
        raise FormalCheckpointError(
            "Formal Stage-II Online version mismatch"
        )

    if (
        payload.get(
            "resource_guard_contract"
        )
        !=
        formal_resource_guard_contract()
    ):
        raise FormalCheckpointError(
            "Checkpoint ResourceGuard contract mismatch"
        )

    _validate_software(
        payload
    )


    # ------------------------------------------------------------------
    # Strict source-code identity for formal resume.
    # ------------------------------------------------------------------

    if strict_git:
        if (
            payload[
                "repository"
            ][
                "dirty"
            ]
        ):
            raise FormalCheckpointError(
                "Formal resume refuses a checkpoint created from a dirty worktree"
            )

        observed_head = (
            current_git_head()
        )

        expected_head = payload[
            "repository"
        ][
            "head"
        ]

        if (
            observed_head
            !=
            expected_head
        ):
            raise FormalCheckpointError(
                "Checkpoint Git HEAD mismatch: "
                f"expected={expected_head}, "
                f"observed={observed_head}"
            )

        if repository_is_dirty():
            raise FormalCheckpointError(
                "Formal resume requires a clean Git worktree"
            )


    seed = int(
        payload[
            "core"
        ][
            "seed"
        ]
    )


    # ------------------------------------------------------------------
    # Reconstruct known-clean formal objects from frozen protocol /
    # provenance first.
    # ------------------------------------------------------------------

    core = prepare_formal_training_core(
        seed=
            seed,

        executable=
            executable,

        dataset_manifest=
            dataset_manifest,

        demo_quality_manifest=
            demo_quality_manifest,

        raw_demo_root=
            raw_demo_root,

        input_provenance_path=
            input_provenance_path,
    )


    # ------------------------------------------------------------------
    # Strict frozen-input identity.
    # ------------------------------------------------------------------

    expected_provenance = payload[
        "input_provenance"
    ]

    if (
        core.input_provenance[
            "manifest_sha256"
        ]
        !=
        expected_provenance[
            "manifest_sha256"
        ]
    ):
        raise FormalCheckpointError(
            "Checkpoint formal-input provenance SHA256 mismatch"
        )

    if (
        core.input_provenance[
            "train49_aggregate_sha256"
        ]
        !=
        expected_provenance[
            "train49_aggregate_sha256"
        ]
    ):
        raise FormalCheckpointError(
            "Checkpoint Train49 provenance mismatch"
        )

    if (
        float(
            expected_provenance[
                "selected_bc_weight"
            ]
        )
        !=
        float(
            PROJECT_BC_WEIGHT
        )
        or
        float(
            PROJECT_BC_WEIGHT
        )
        !=
        3.0
    ):
        raise FormalCheckpointError(
            "Checkpoint lambda_BC provenance mismatch"
        )


    # ------------------------------------------------------------------
    # Neural modules + Actor/Critic optimizer states.
    #
    # Tianshou load_state_dict() pops its optimizer key internally,
    # therefore pass a deepcopy rather than mutating payload.
    # ------------------------------------------------------------------

    algorithm_state = copy.deepcopy(
        payload[
            "algorithm_state"
        ]
    )

    core.algorithm.load_state_dict(
        algorithm_state,
        strict=
            True,
    )


    # ------------------------------------------------------------------
    # Independent alpha Adam state.
    # ------------------------------------------------------------------

    core.auto_alpha._optim.load_state_dict(
        copy.deepcopy(
            payload[
                "alpha_optimizer_state"
            ]
        )
    )

    core.auto_alpha._last_update = (
        copy.deepcopy(
            payload[
                "auto_alpha_last_update"
            ]
        )
    )


    # ------------------------------------------------------------------
    # Plain non-state_dict protocol state.
    # ------------------------------------------------------------------

    core_state = payload[
        "core"
    ]

    core.stage = str(
        core_state[
            "stage"
        ]
    )

    core.stage1_updates_completed = int(
        core_state[
            "stage1_updates_completed"
        ]
    )

    core.stage1_sampled_demo_transitions = int(
        core_state[
            "stage1_sampled_demo_transitions"
        ]
    )

    core.algorithm.bc_enabled = bool(
        core_state[
            "bc_enabled"
        ]
    )

    core.algorithm.bc_weight = float(
        core_state[
            "bc_weight"
        ]
    )

    core.policy.deterministic_eval = bool(
        core_state[
            "policy_deterministic_eval"
        ]
    )

    core.policy.set_exploration_epsilon(
        float(
            core_state[
                "policy_exploration_epsilon"
            ]
        )
    )


    # ------------------------------------------------------------------
    # D_demo data were freshly reloaded from frozen provenance.
    # Restore its independent sampling RNG.
    # ------------------------------------------------------------------

    restore_demo_replay_state(
        core.demo_buffer,
        payload[
            "demo_replay"
        ],
    )


    # ------------------------------------------------------------------
    # Masked epsilon-greedy RNG.
    # ------------------------------------------------------------------

    core.policy._exploration_rng.bit_generator.state = (
        copy.deepcopy(
            payload[
                "policy_exploration_rng_state"
            ]
        )
    )


    # ------------------------------------------------------------------
    # Stage-II state / D_expo.
    # ------------------------------------------------------------------

    stage2_snapshot = payload[
        "stage2"
    ]

    stage2_state = None

    if stage2_snapshot is not None:
        if (
            core.stage
            !=
            "STAGE_II"
        ):
            raise FormalCheckpointError(
                "Checkpoint contains Stage-II state but core stage is not STAGE_II"
            )

        stage2_state = (
            prepare_formal_stage2_state(
                core,

                dataset_manifest=
                    dataset_manifest,
            )
        )

        observed_model_names = tuple(
            model.model
            for model in
            stage2_state.models
        )

        expected_model_names = tuple(
            stage2_snapshot[
                "model_names"
            ]
        )

        if (
            observed_model_names
            !=
            expected_model_names
        ):
            raise FormalCheckpointError(
                "Checkpoint Train49 ordered model list mismatch"
            )

        observed_complexity_strata = tuple(
            int(
                model.complexity_stratum
            )
            for model in
            stage2_state.models
        )

        expected_complexity_strata = tuple(
            int(
                value
            )
            for value in
            stage2_snapshot[
                "model_complexity_strata"
            ]
        )

        if (
            observed_complexity_strata
            !=
            expected_complexity_strata
        ):
            raise FormalCheckpointError(
                "Checkpoint Train49 complexity-stratum list mismatch"
            )

        stage2_state.expo_buffer = (
            restore_expo_replay(
                snapshot=
                    stage2_snapshot[
                        "expo_replay"
                    ],

                seed=
                    seed,
            )
        )

        stage2_state.total_environment_steps = int(
            stage2_snapshot[
                "total_environment_steps"
            ]
        )

        stage2_state.total_gradient_updates = int(
            stage2_snapshot[
                "total_gradient_updates"
            ]
        )

        restored_curriculum_phase = (
            formal_stage2_curriculum_phase(
                stage2_state
            )
        )

        expected_curriculum_phase = str(
            stage2_snapshot[
                "curriculum_phase"
            ]
        )

        if (
            restored_curriculum_phase
            !=
            expected_curriculum_phase
        ):
            raise FormalCheckpointError(
                "Checkpoint Stage-II curriculum phase mismatch: "
                f"expected={expected_curriculum_phase}, "
                f"observed={restored_curriculum_phase}"
            )

        stage2_state.episode_attempts = int(
            stage2_snapshot[
                "episode_attempts"
            ]
        )

        stage2_state.completed_episodes = int(
            stage2_snapshot[
                "completed_episodes"
            ]
        )

        stage2_state.history = copy.deepcopy(
            stage2_snapshot[
                "history"
            ]
        )

        stage2_state.model_rng.bit_generator.state = (
            copy.deepcopy(
                stage2_snapshot[
                    "model_rng_state"
                ]
            )
        )

    else:
        if (
            core.stage
            !=
            "STAGE_I"
        ):
            raise FormalCheckpointError(
                "Checkpoint lacks Stage-II state but core stage is not STAGE_I"
            )


    # ------------------------------------------------------------------
    # Restore global RNGs LAST.
    #
    # Core/network/replay reconstruction above is allowed to consume RNG
    # state; none of it may leak into the resumed training trajectory.
    # ------------------------------------------------------------------

    restore_global_rng_state(
        payload[
            "global_rng_state"
        ]
    )


    assert_checkpoint_boundary(
        core,
        stage2_state,
    )

    return (
        core,
        stage2_state,
        {
            "checkpoint_path":
                str(
                    checkpoint_path
                ),

            "checkpoint_sha256":
                sha256_file(
                    checkpoint_path
                ),

            "checkpoint_bytes":
                int(
                    checkpoint_path
                    .stat()
                    .st_size
                ),

            "repository_head":
                payload[
                    "repository"
                ][
                    "head"
                ],

            "stage":
                core.stage,

            "seed":
                core.seed,
        },
    )
