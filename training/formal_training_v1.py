from __future__ import annotations

import csv
import math
import random
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


import gymnasium as gym
import numpy as np
import torch

from torch.optim import (
    AdamW,
)

from tianshou.algorithm.optim import (
    AdamOptimizerFactory,
    TorchOptimizerFactory,
)

from tianshou.data import (
    Collector,
    ReplayBuffer,
)

from tianshou.env import (
    DummyVectorEnv,
)

from tianshou.utils.torch_utils import (
    policy_within_training_step,
)


from algorithms.demo_guided_discrete_sac_v1 import (
    LoopyCutsDemoGuidedDiscreteSACV1,
)

from bridge.resource_guard_v1 import (
    GIB,
    ResourceGuardPolicyV1,
)

from envs.final_reward_wrapper_v5 import (
    FinalRewardWrapperV5,
)

from envs.finalization_quality_wrapper_v1 import (
    FinalizationQualityWrapperV1,
)

from envs.loopycuts_env import (
    LoopyCutsEnv,
)

from envs.formal_episode_collector_bridge_v1 import (
    FormalEpisodeCollectorBridgeV1,
)

from rewards.reward_v5 import (
    REWARD_V5_VERSION,
)

from imitation.demo_replay import (
    load_main_demo_replay,
)

from networks.loopycuts_actor_critic_v1 import (
    build_loopycuts_actor_critics_v1,
)

from observation.builder import (
    MAX_LOOPS,
)

from policies.masked_discrete_sac import (
    MaskedDiscreteSACPolicy,
)

from training.formal_training_input_provenance_v2 import (
    assert_formal_training_input_provenance_v2,
)

from training.masked_auto_alpha_v1 import (
    MaskedAutoAlphaV1,
)

from training.training_metrics_v1 import (
    TrainingMetricsWriterV1,
)

from training.protocol_v1 import (
    PROJECT_STAGE2_RESOURCE_GUARD_SAMPLE_INTERVAL_SECONDS,
    PROJECT_STAGE2_RESOURCE_GUARD_WARNING_SWAP_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_ABORT_SWAP_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_ABORT_HOLD_SECONDS,
    PROJECT_STAGE2_RESOURCE_GUARD_EMERGENCY_SWAP_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_FINALIZE_QUALITY_SWAP_ABORT_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_REARM_SWAP_GIB,
    PAPER_BATCH_SIZE,
    PAPER_DISCOUNT_FACTOR,
    PAPER_ENTROPY_TARGET_COEFFICIENT,
    PAPER_SOFT_UPDATE_TAU,

    PROJECT_ACTOR_CRITIC_OPTIMIZER_FAMILY,
    PROJECT_ACTOR_CRITIC_BETAS,
    PROJECT_ACTOR_CRITIC_EPS,
    PROJECT_ACTOR_CRITIC_WEIGHT_DECAY,

    PROJECT_ALPHA_OPTIMIZER_FAMILY,
    PROJECT_ALPHA_ADAM_BETAS,
    PROJECT_ALPHA_ADAM_EPS,
    PROJECT_ALPHA_ADAM_WEIGHT_DECAY,

    PROJECT_ACTOR_LEARNING_RATE,
    PROJECT_CRITIC1_LEARNING_RATE,
    PROJECT_CRITIC2_LEARNING_RATE,
    PROJECT_ALPHA_LEARNING_RATE,

    PROJECT_INITIAL_ALPHA,
    PROJECT_N_STEP_RETURN_HORIZON,

    PROJECT_MAIN_DEMO_EPISODES,
    PROJECT_MAIN_DEMO_TRANSITIONS,
    PROJECT_MAIN_DEMO_MODELS,

    PROJECT_BC_WEIGHT,

    PROJECT_FORMAL_TRAINING_SEEDS,
    PROJECT_FORMAL_TRAINING_DEVICE,
    PROJECT_FORMAL_TRAINING_TORCH_NUM_THREADS,
    PROJECT_FORMAL_TRAINING_TORCH_NUM_INTEROP_THREADS,
    PROJECT_FORMAL_TRAINING_TORCH_DETERMINISTIC_ALGORITHMS,

    PROJECT_STAGE1_GRADIENT_STEPS,
    PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS,

    PROJECT_STAGE2_BC_ENABLED,
    PROJECT_STAGE2_EXPLORATION_EPSILON,
    PROJECT_STAGE2_COLLECTOR_EXPLORATION_NOISE,
    PROJECT_STAGE2_SAMPLES_PER_BUFFER,
    PROJECT_STAGE2_EXPO_REPLAY_CAPACITY,

    PROJECT_STAGE2_MODEL_SPLIT,
    PROJECT_STAGE2_MODEL_COUNT,
    PROJECT_STAGE2_MODEL_SAMPLING,
    PROJECT_STAGE2_MODEL_SAMPLING_RNG,

    PROJECT_STAGE2_CURRICULUM_VERSION,
    PROJECT_STAGE2_CURRICULUM_WARMUP_ENV_STEPS,
    PROJECT_STAGE2_CURRICULUM_WARMUP_MAX_STRATUM,
    PROJECT_STAGE2_CURRICULUM_WARMUP_MODEL_COUNT,
    PROJECT_STAGE2_CURRICULUM_FULL_MODEL_COUNT,
    PROJECT_STAGE2_CURRICULUM_WARMUP_POOL,
    PROJECT_STAGE2_CURRICULUM_FULL_POOL,
    PROJECT_STAGE2_CURRICULUM_PHASE_SELECTION,
    PROJECT_STAGE2_CURRICULUM_BOUNDARY_POLICY,
    PROJECT_STAGE2_CURRICULUM_SAMPLING_WITHIN_POOL,

    PROJECT_STAGE2_DEV_ALLOWED,
    PROJECT_STAGE2_BLIND_ALLOWED,

    PROJECT_STAGE2_COLLECTION_UPDATE_RATIO,
    PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS,
    PROJECT_STAGE2_UPDATE_SCHEDULING,
    PROJECT_STAGE2_BUDGET_BOUNDARY_POLICY,

    PROJECT_NETWORK_REINITIALIZATION,

    assert_formal_training_ready,
)


FORMAL_TRAINER_CORE_VERSION = (
    "loopycuts_formal_training_core_v1"
)


DEFAULT_EXECUTABLE = Path(
    "/home/yjk/codes/LoopyCuts_v5/"
    "volumetric_cutter/"
    "volumetric_cutter"
)

DEFAULT_DATASET_MANIFEST = Path(
    "data/manifests/"
    "dataset_split_v2.csv"
)

DEFAULT_DEMO_QUALITY = Path(
    "data/manifests/"
    "demo_quality_v1.csv"
)

DEFAULT_RAW_DEMO_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "demonstrations_v1/raw/train"
)

DEFAULT_FORMAL_INPUT_PROVENANCE = Path(
    "data/manifests/"
    "formal_training_input_provenance_v1.json"
)


class FormalTrainingCoreError(
    RuntimeError
):
    pass


@dataclass
class FormalTrainingCoreV1:
    seed: int

    algorithm: Any
    policy: Any
    auto_alpha: Any

    demo_buffer: Any
    demo_records: list
    demo_provenance: dict

    input_provenance: dict
    runtime: dict

    stage: str = (
        "STAGE_I"
    )

    stage1_updates_completed: int = 0

    stage1_sampled_demo_transitions: int = 0


def assert_formal_core_protocol():
    assert_formal_training_ready()

    if (
        PAPER_BATCH_SIZE
        !=
        64
    ):
        raise FormalTrainingCoreError(
            "Formal batch size must be 64"
        )

    if not math.isclose(
        PROJECT_BC_WEIGHT,
        3.0,
    ):
        raise FormalTrainingCoreError(
            "Formal lambda_BC must be 3.0"
        )

    if (
        PROJECT_FORMAL_TRAINING_SEEDS
        !=
        (
            42,
            43,
            44,
        )
    ):
        raise FormalTrainingCoreError(
            "Unexpected formal seed set"
        )

    if (
        PROJECT_FORMAL_TRAINING_DEVICE
        !=
        "cpu"
    ):
        raise FormalTrainingCoreError(
            "Formal training device must be CPU"
        )

    if (
        PROJECT_STAGE1_GRADIENT_STEPS
        !=
        782
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-I update budget must be 782"
        )

    if (
        PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS
        !=
        50_048
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-I sampled-demo budget mismatch"
        )

    if (
        PROJECT_STAGE2_BC_ENABLED
        is not
        False
    ):
        raise FormalTrainingCoreError(
            "Stage-II BC must be disabled"
        )

    if (
        PROJECT_NETWORK_REINITIALIZATION
        is not
        False
    ):
        raise FormalTrainingCoreError(
            "Stage transition must not reinitialize networks"
        )

    if not math.isclose(
        PROJECT_STAGE2_EXPLORATION_EPSILON,
        0.05,
    ):
        raise FormalTrainingCoreError(
            "Stage-II epsilon must be 0.05"
        )

    if (
        PROJECT_STAGE2_SAMPLES_PER_BUFFER
        !=
        32
    ):
        raise FormalTrainingCoreError(
            "Stage-II equal replay must use 32+32"
        )


def configure_formal_training_runtime():
    expected_threads = int(
        PROJECT_FORMAL_TRAINING_TORCH_NUM_THREADS
    )

    expected_interop = int(
        PROJECT_FORMAL_TRAINING_TORCH_NUM_INTEROP_THREADS
    )

    expected_deterministic = bool(
        PROJECT_FORMAL_TRAINING_TORCH_DETERMINISTIC_ALGORITHMS
    )

    if (
        torch.get_num_threads()
        !=
        expected_threads
    ):
        torch.set_num_threads(
            expected_threads
        )

    if (
        torch.get_num_interop_threads()
        !=
        expected_interop
    ):
        try:
            torch.set_num_interop_threads(
                expected_interop
            )

        except RuntimeError as exc:
            raise FormalTrainingCoreError(
                "Cannot apply frozen torch inter-op thread policy"
            ) from exc

    torch.use_deterministic_algorithms(
        expected_deterministic
    )

    observed = {
        "device":
            PROJECT_FORMAL_TRAINING_DEVICE,

        "torch_num_threads":
            torch.get_num_threads(),

        "torch_num_interop_threads":
            torch.get_num_interop_threads(),

        "torch_deterministic_algorithms":
            torch.are_deterministic_algorithms_enabled(),
    }

    expected = {
        "device":
            "cpu",

        "torch_num_threads":
            8,

        "torch_num_interop_threads":
            8,

        "torch_deterministic_algorithms":
            False,
    }

    if (
        observed
        !=
        expected
    ):
        raise FormalTrainingCoreError(
            "Formal numerical runtime mismatch: "
            f"expected={expected}, "
            f"observed={observed}"
        )

    return observed


def set_formal_training_seed(
    seed: int,
):
    seed = int(
        seed
    )

    if (
        seed
        not in
        PROJECT_FORMAL_TRAINING_SEEDS
    ):
        raise FormalTrainingCoreError(
            f"Formal seed is not frozen: {seed}"
        )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


def make_actor_critic_optimizer_factory(
    *,
    lr: float,
):
    if (
        PROJECT_ACTOR_CRITIC_OPTIMIZER_FAMILY
        !=
        "ADAMW"
    ):
        raise FormalTrainingCoreError(
            "Actor/Critic optimizer must be AdamW"
        )

    return TorchOptimizerFactory(
        AdamW,

        lr=
            float(
                lr
            ),

        betas=
            PROJECT_ACTOR_CRITIC_BETAS,

        eps=
            PROJECT_ACTOR_CRITIC_EPS,

        weight_decay=
            PROJECT_ACTOR_CRITIC_WEIGHT_DECAY,
    )


def make_alpha_optimizer_factory(
    *,
    lr: float,
):
    if (
        PROJECT_ALPHA_OPTIMIZER_FAMILY
        !=
        "ADAM"
    ):
        raise FormalTrainingCoreError(
            "Alpha optimizer must be Adam"
        )

    return AdamOptimizerFactory(
        lr=
            float(
                lr
            ),

        betas=
            PROJECT_ALPHA_ADAM_BETAS,

        eps=
            PROJECT_ALPHA_ADAM_EPS,

        weight_decay=
            PROJECT_ALPHA_ADAM_WEIGHT_DECAY,
    )


def build_formal_algorithm(
    *,
    seed: int,
):
    set_formal_training_seed(
        seed
    )

    actor, critic1, critic2 = (
        build_loopycuts_actor_critics_v1(
            device=
                PROJECT_FORMAL_TRAINING_DEVICE
        )
    )

    # During Stage-I there is no environment collection.
    #
    # deterministic_eval=True is retained so Stage-II exploitation
    # selects the maximum-probability legal action.  Epsilon noise
    # is enabled only when entering Stage-II.
    policy = MaskedDiscreteSACPolicy(
        actor=
            actor,

        action_space=
            gym.spaces.Discrete(
                MAX_LOOPS
            ),

        deterministic_eval=
            True,

        exploration_epsilon=
            0.0,

        exploration_seed=
            int(
                seed
            ),
    )

    auto_alpha = MaskedAutoAlphaV1(
        target_coefficient=
            PAPER_ENTROPY_TARGET_COEFFICIENT,

        initial_alpha=
            PROJECT_INITIAL_ALPHA,

        optim=
            make_alpha_optimizer_factory(
                lr=
                    PROJECT_ALPHA_LEARNING_RATE
            ),

        device=
            PROJECT_FORMAL_TRAINING_DEVICE,
    )

    algorithm = LoopyCutsDemoGuidedDiscreteSACV1(
        policy=
            policy,

        policy_optim=
            make_actor_critic_optimizer_factory(
                lr=
                    PROJECT_ACTOR_LEARNING_RATE
            ),

        critic=
            critic1,

        critic_optim=
            make_actor_critic_optimizer_factory(
                lr=
                    PROJECT_CRITIC1_LEARNING_RATE
            ),

        critic2=
            critic2,

        critic2_optim=
            make_actor_critic_optimizer_factory(
                lr=
                    PROJECT_CRITIC2_LEARNING_RATE
            ),

        tau=
            PAPER_SOFT_UPDATE_TAU,

        gamma=
            PAPER_DISCOUNT_FACTOR,

        alpha=
            auto_alpha,

        n_step_return_horizon=
            PROJECT_N_STEP_RETURN_HORIZON,

        bc_weight=
            PROJECT_BC_WEIGHT,

        bc_enabled=
            True,
    )

    return (
        algorithm,
        policy,
        auto_alpha,
    )


def prepare_formal_training_core(
    *,
    seed: int,

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
):
    assert_formal_core_protocol()

    seed = int(
        seed
    )

    if (
        seed
        not in
        PROJECT_FORMAL_TRAINING_SEEDS
    ):
        raise FormalTrainingCoreError(
            f"Formal seed is not frozen: {seed}"
        )

    runtime = (
        configure_formal_training_runtime()
    )

    input_provenance = (
        assert_formal_training_input_provenance_v2(
            historical_provenance_path=
                input_provenance_path,

            executable=
                executable,

            dataset_manifest=
                dataset_manifest,

            demo_quality_manifest=
                demo_quality_manifest,

            quality_ref_root=
                DEFAULT_QUALITY_REF_ROOT,
        )
    )

    (
        demo_buffer,
        demo_records,
        demo_provenance,
    ) = load_main_demo_replay(
        raw_root=
            raw_demo_root,

        quality_manifest=
            demo_quality_manifest,

        random_seed=
            seed,
    )

    if (
        len(
            demo_records
        )
        !=
        PROJECT_MAIN_DEMO_EPISODES
        or
        PROJECT_MAIN_DEMO_EPISODES
        !=
        30
    ):
        raise FormalTrainingCoreError(
            "Formal D_demo episode count mismatch"
        )

    if (
        len(
            demo_buffer
        )
        !=
        PROJECT_MAIN_DEMO_TRANSITIONS
        or
        PROJECT_MAIN_DEMO_TRANSITIONS
        !=
        605
    ):
        raise FormalTrainingCoreError(
            "Formal D_demo transition count mismatch"
        )

    observed_models = tuple(
        record[
            "model"
        ]
        for record in
        demo_records
    )

    if (
        observed_models
        !=
        PROJECT_MAIN_DEMO_MODELS
    ):
        raise FormalTrainingCoreError(
            "Formal D_demo model order/content mismatch"
        )

    (
        algorithm,
        policy,
        auto_alpha,
    ) = build_formal_algorithm(
        seed=
            seed
    )

    if (
        not
        algorithm.bc_enabled
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-I must begin with BC enabled"
        )

    if not math.isclose(
        algorithm.bc_weight,
        PROJECT_BC_WEIGHT,
    ):
        raise FormalTrainingCoreError(
            "Formal algorithm lambda_BC mismatch"
        )

    if not math.isclose(
        policy.exploration_epsilon,
        0.0,
    ):
        raise FormalTrainingCoreError(
            "Stage-I exploration epsilon must begin at zero"
        )

    return FormalTrainingCoreV1(
        seed=
            seed,

        algorithm=
            algorithm,

        policy=
            policy,

        auto_alpha=
            auto_alpha,

        demo_buffer=
            demo_buffer,

        demo_records=
            demo_records,

        demo_provenance=
            demo_provenance,

        input_provenance=
            input_provenance,

        runtime=
            runtime,
    )


def training_stats_snapshot(
    stats,
):
    names = (
        "actor_loss",
        "critic1_loss",
        "critic2_loss",
        "alpha",
        "alpha_loss",

        "sac_actor_loss",
        "bc_loss",
        "total_actor_loss",

        "bc_unfiltered_loss",
        "bc_selected_count",
        "bc_filter_fraction",
        "bc_mean_expert_probability",
        "bc_top1_accuracy",
        "bc_mean_q_margin",

        "bc_weight",
    )

    result = {}

    for name in names:
        if not hasattr(
            stats,
            name,
        ):
            continue

        value = getattr(
            stats,
            name,
        )

        if isinstance(
            value,
            (
                int,
                np.integer,
            ),
        ):
            result[
                name
            ] = int(
                value
            )

            continue

        try:
            numeric = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if not math.isfinite(
            numeric
        ):
            raise FormalTrainingCoreError(
                f"Non-finite training statistic: "
                f"{name}={numeric}"
            )

        result[
            name
        ] = numeric

    return result


def run_formal_stage1(
    core: FormalTrainingCoreV1,
    *,
    metrics_writer:
        TrainingMetricsWriterV1
        | None
        = None,
):
    if (
        core.stage
        !=
        "STAGE_I"
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-I may only run from STAGE_I state"
        )

    if (
        core.stage1_updates_completed
        !=
        0
        or
        core.stage1_sampled_demo_transitions
        !=
        0
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-I is not at its initial state"
        )

    if (
        not
        core.algorithm.bc_enabled
    ):
        raise FormalTrainingCoreError(
            "BC must remain enabled throughout Stage-I"
        )

    total_updates = int(
        PROJECT_STAGE1_GRADIENT_STEPS
    )

    final_snapshot = None

    start = time.time()

    with policy_within_training_step(
        core.policy
    ):
        for update_index in range(
            1,
            total_updates + 1,
        ):
            stats = core.algorithm.update(
                core.demo_buffer,

                sample_size=
                    PAPER_BATCH_SIZE,
            )

            snapshot = (
                training_stats_snapshot(
                    stats
                )
            )

            core.stage1_updates_completed = (
                update_index
            )

            core.stage1_sampled_demo_transitions = (
                update_index
                *
                PAPER_BATCH_SIZE
            )

            if metrics_writer is not None:
                metrics_writer.append(
                    seed=
                        core.seed,

                    stage=
                        "STAGE_I",

                    gradient_update=
                        update_index,

                    sampled_demo_transitions=
                        core.stage1_sampled_demo_transitions,

                    stats=
                        snapshot,
                )

            if (
                update_index
                ==
                1
                or
                update_index
                ==
                total_updates
                or
                update_index
                %
                100
                ==
                0
            ):
                print(
                    "formal-stage1 "
                    f"update={update_index:4d}/"
                    f"{total_updates} "
                    f"actor_loss="
                    f"{snapshot.get('actor_loss')} "
                    f"bc_loss="
                    f"{snapshot.get('bc_loss')} "
                    f"alpha="
                    f"{snapshot.get('alpha')}"
                )

            final_snapshot = (
                snapshot
            )

    elapsed = (
        time.time()
        -
        start
    )

    if (
        core.stage1_updates_completed
        !=
        PROJECT_STAGE1_GRADIENT_STEPS
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-I update count mismatch"
        )

    if (
        core.stage1_sampled_demo_transitions
        !=
        PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-I sampled-demo count mismatch"
        )

    if final_snapshot is None:
        raise FormalTrainingCoreError(
            "Formal Stage-I produced no update statistics"
        )

    return {
        "gradient_updates":
            core.stage1_updates_completed,

        "sampled_demo_transitions":
            core.stage1_sampled_demo_transitions,

        "elapsed_seconds":
            float(
                elapsed
            ),

        "alpha_after_stage1":
            float(
                core.auto_alpha.value
            ),

        "final_training_stats":
            final_snapshot,
    }


def enter_formal_stage2(
    core: FormalTrainingCoreV1,
):
    if (
        core.stage
        !=
        "STAGE_I"
    ):
        raise FormalTrainingCoreError(
            "Stage-II transition requires STAGE_I state"
        )

    if (
        core.stage1_updates_completed
        !=
        PROJECT_STAGE1_GRADIENT_STEPS
    ):
        raise FormalTrainingCoreError(
            "Cannot enter Stage-II before exact Stage-I completion"
        )

    if (
        core.stage1_sampled_demo_transitions
        !=
        PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS
    ):
        raise FormalTrainingCoreError(
            "Stage-I sampled-demo budget is incomplete"
        )

    if (
        not
        core.algorithm.bc_enabled
    ):
        raise FormalTrainingCoreError(
            "BC was unexpectedly disabled before Stage-II transition"
        )

    identities_before = {
        "algorithm":
            id(
                core.algorithm
            ),

        "policy":
            id(
                core.policy
            ),

        "actor":
            id(
                core.policy.actor
            ),

        "critic1":
            id(
                core.algorithm.critic
            ),

        "critic2":
            id(
                core.algorithm.critic2
            ),

        "auto_alpha":
            id(
                core.auto_alpha
            ),
    }

    # This is the actual Stage-I -> Stage-II transition.
    #
    # No Actor, Critic, target network, alpha, or optimizer is
    # re-created here.
    core.algorithm.set_bc_enabled(
        False
    )

    core.policy.set_exploration_epsilon(
        PROJECT_STAGE2_EXPLORATION_EPSILON
    )

    identities_after = {
        "algorithm":
            id(
                core.algorithm
            ),

        "policy":
            id(
                core.policy
            ),

        "actor":
            id(
                core.policy.actor
            ),

        "critic1":
            id(
                core.algorithm.critic
            ),

        "critic2":
            id(
                core.algorithm.critic2
            ),

        "auto_alpha":
            id(
                core.auto_alpha
            ),
    }

    if (
        identities_after
        !=
        identities_before
    ):
        raise FormalTrainingCoreError(
            "Stage-II transition reinitialized formal training objects"
        )

    if (
        core.algorithm.bc_enabled
    ):
        raise FormalTrainingCoreError(
            "BC must be disabled in Stage-II"
        )

    if not math.isclose(
        core.policy.exploration_epsilon,
        PROJECT_STAGE2_EXPLORATION_EPSILON,
    ):
        raise FormalTrainingCoreError(
            "Stage-II exploration epsilon mismatch"
        )

    core.stage = (
        "STAGE_II"
    )

    return {
        "stage":
            core.stage,

        "bc_enabled":
            bool(
                core.algorithm.bc_enabled
            ),

        "exploration_epsilon":
            float(
                core.policy.exploration_epsilon
            ),

        "samples_per_replay_source":
            int(
                PROJECT_STAGE2_SAMPLES_PER_BUFFER
            ),

        "object_identities":
            identities_after,
    }


# ======================================================================
# PHASE 18.3 -- FORMAL STAGE-II ONLINE TRAINING
# ======================================================================

FORMAL_STAGE2_ONLINE_VERSION = (
    "loopycuts_formal_stage2_online_v5_quality_aware"
)


DEFAULT_QUALITY_REF_ROOT = (
    Path.home()
    /
    "loopycuts_test"
    /
    "quality_refs_train49_v1"
    /
    "refs"
)


@dataclass(
    frozen=True
)
class FormalStage2ModelV1:
    model: str

    mesh_file: Path
    loop_file: Path

    header_loops: int
    actionable_nonconvex: int

    complexity_stratum: int

    quality_ref_file: Path | None = None


@dataclass
class FormalStage2StateV1:
    seed: int

    expo_buffer: Any

    models: tuple[
        FormalStage2ModelV1,
        ...
    ]

    model_rng: np.random.Generator

    total_environment_steps: int = 0
    total_gradient_updates: int = 0

    episode_attempts: int = 0
    completed_episodes: int = 0

    history: list[
        dict
    ] = field(
        default_factory=list
    )


def assert_formal_stage2_protocol():
    if (
        PROJECT_STAGE2_MODEL_SPLIT
        !=
        "train"
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-II must use split=train"
        )

    if (
        PROJECT_STAGE2_MODEL_COUNT
        !=
        49
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-II model count must be 49"
        )

    if (
        PROJECT_STAGE2_MODEL_SAMPLING
        !=
        "COMPLEXITY_CURRICULUM_UNIFORM_IID_PER_EPISODE"
    ):
        raise FormalTrainingCoreError(
            "Unexpected formal curriculum model-sampling semantics"
        )

    if (
        PROJECT_STAGE2_MODEL_SAMPLING_RNG
        !=
        "NUMPY_GENERATOR_SEEDED_BY_FORMAL_RUN_SEED"
    ):
        raise FormalTrainingCoreError(
            "Unexpected formal model-sampling RNG semantics"
        )

    if (
        PROJECT_STAGE2_CURRICULUM_VERSION
        !=
        "complexity_curriculum_v1"
    ):
        raise FormalTrainingCoreError(
            "Unexpected Stage-II curriculum version"
        )

    if (
        PROJECT_STAGE2_CURRICULUM_WARMUP_ENV_STEPS
        !=
        5_000
    ):
        raise FormalTrainingCoreError(
            "Stage-II curriculum warmup boundary must be 5000"
        )

    if (
        PROJECT_STAGE2_CURRICULUM_WARMUP_MAX_STRATUM
        !=
        7
    ):
        raise FormalTrainingCoreError(
            "Stage-II curriculum warmup max stratum must be 7"
        )

    if (
        PROJECT_STAGE2_CURRICULUM_WARMUP_MODEL_COUNT
        !=
        39
    ):
        raise FormalTrainingCoreError(
            "Stage-II curriculum warmup model count must be 39"
        )

    if (
        PROJECT_STAGE2_CURRICULUM_FULL_MODEL_COUNT
        !=
        PROJECT_STAGE2_MODEL_COUNT
        or
        PROJECT_STAGE2_CURRICULUM_FULL_MODEL_COUNT
        !=
        49
    ):
        raise FormalTrainingCoreError(
            "Stage-II curriculum full model count must be 49"
        )

    if (
        PROJECT_STAGE2_CURRICULUM_WARMUP_POOL
        !=
        "TRAIN_MODELS_WITH_COMPLEXITY_STRATUM_LE_7"
    ):
        raise FormalTrainingCoreError(
            "Unexpected Stage-II curriculum warmup-pool semantics"
        )

    if (
        PROJECT_STAGE2_CURRICULUM_FULL_POOL
        !=
        "ALL_TRAIN49"
    ):
        raise FormalTrainingCoreError(
            "Unexpected Stage-II curriculum full-pool semantics"
        )

    if (
        PROJECT_STAGE2_CURRICULUM_PHASE_SELECTION
        !=
        "AT_EPISODE_START_FROM_TOTAL_ENVIRONMENT_STEPS"
    ):
        raise FormalTrainingCoreError(
            "Unexpected Stage-II curriculum phase-selection semantics"
        )

    if (
        PROJECT_STAGE2_CURRICULUM_BOUNDARY_POLICY
        !=
        "NO_MID_EPISODE_PHASE_SWITCH"
    ):
        raise FormalTrainingCoreError(
            "Unexpected Stage-II curriculum boundary semantics"
        )

    if (
        PROJECT_STAGE2_CURRICULUM_SAMPLING_WITHIN_POOL
        !=
        "UNIFORM_IID_PER_EPISODE"
    ):
        raise FormalTrainingCoreError(
            "Unexpected within-pool curriculum sampling semantics"
        )

    if (
        PROJECT_STAGE2_DEV_ALLOWED
        is not
        False
    ):
        raise FormalTrainingCoreError(
            "Dev models must not enter formal Stage-II"
        )

    if (
        PROJECT_STAGE2_BLIND_ALLOWED
        is not
        False
    ):
        raise FormalTrainingCoreError(
            "Blind models must not enter formal Stage-II"
        )

    if (
        PROJECT_STAGE2_COLLECTOR_EXPLORATION_NOISE
        is not
        True
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-II Collector exploration must be enabled"
        )

    if (
        PROJECT_STAGE2_EXPO_REPLAY_CAPACITY
        !=
        25_000
    ):
        raise FormalTrainingCoreError(
            "Formal D_expo capacity must be 25000"
        )

    if (
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
        !=
        25_000
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-II environment budget must be 25000"
        )

    if not math.isclose(
        PROJECT_STAGE2_COLLECTION_UPDATE_RATIO,
        1.0,
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-II update ratio must be 1.0"
        )

    if (
        PROJECT_STAGE2_UPDATE_SCHEDULING
        !=
        "AFTER_EACH_COMPLETED_EPISODE"
    ):
        raise FormalTrainingCoreError(
            "Unexpected Stage-II update scheduling"
        )

    if (
        PROJECT_STAGE2_BUDGET_BOUNDARY_POLICY
        !=
        (
            "STOP_AT_EXACT_TRANSITION_BUDGET_WITHOUT_"
            "SYNTHETIC_TERMINAL_OR_TRUNCATION"
        )
    ):
        raise FormalTrainingCoreError(
            "Unexpected Stage-II budget-boundary semantics"
        )


def load_formal_stage2_models(
    *,
    dataset_manifest: Path =
        DEFAULT_DATASET_MANIFEST,
):
    dataset_manifest = Path(
        dataset_manifest
    )

    with dataset_manifest.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    train_rows = [
        row
        for row in rows
        if (
            row[
                "split"
            ]
            ==
            PROJECT_STAGE2_MODEL_SPLIT
        )
    ]

    if (
        len(
            train_rows
        )
        !=
        PROJECT_STAGE2_MODEL_COUNT
        or
        PROJECT_STAGE2_MODEL_COUNT
        !=
        49
    ):
        raise FormalTrainingCoreError(
            "Formal Train49 model count mismatch"
        )

    names = [
        row[
            "model"
        ]
        for row in train_rows
    ]

    if (
        len(
            set(
                names
            )
        )
        !=
        49
    ):
        raise FormalTrainingCoreError(
            "Formal Train49 contains duplicate model names"
        )

    models = []

    for row in train_rows:
        model_name = str(
            row[
                "model"
            ]
        )

        quality_ref_file = (
            DEFAULT_QUALITY_REF_ROOT
            /
            (
                model_name
                +
                ".quality_ref_v1"
            )
        ).resolve()

        if not quality_ref_file.is_file():
            raise FileNotFoundError(
                quality_ref_file
            )

        mesh_file = Path(
            row[
                "mesh_file"
            ]
        ).resolve()

        loop_file = Path(
            row[
                "loop_file"
            ]
        ).resolve()

        if not mesh_file.is_file():
            raise FileNotFoundError(
                mesh_file
            )

        if not loop_file.is_file():
            raise FileNotFoundError(
                loop_file
            )

        models.append(
            FormalStage2ModelV1(
                model=
                    str(
                        row[
                            "model"
                        ]
                    ),

                mesh_file=
                    mesh_file,

                loop_file=
                    loop_file,

                quality_ref_file=
                    quality_ref_file,

                header_loops=
                    int(
                        row[
                            "header_loops"
                        ]
                    ),

                actionable_nonconvex=
                    int(
                        row[
                            "actionable_nonconvex"
                        ]
                    ),

                complexity_stratum=
                    int(
                        row[
                            "complexity_stratum"
                        ]
                    ),
            )
        )

    models.sort(
        key=lambda x:
            x.model
    )

    return tuple(
        models
    )


def prepare_formal_stage2_state(
    core: FormalTrainingCoreV1,
    *,
    dataset_manifest: Path =
        DEFAULT_DATASET_MANIFEST,
):
    assert_formal_stage2_protocol()

    if (
        core.stage
        !=
        "STAGE_II"
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-II state requires core.stage == STAGE_II"
        )

    if (
        core.algorithm.bc_enabled
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-II must begin with BC disabled"
        )

    if (
        core.stage1_updates_completed
        !=
        PROJECT_STAGE1_GRADIENT_STEPS
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-I update budget is incomplete"
        )

    if (
        core.stage1_sampled_demo_transitions
        !=
        PROJECT_STAGE1_ACTUAL_SAMPLED_DEMO_TRANSITIONS
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-I demonstration exposure is incomplete"
        )

    models = (
        load_formal_stage2_models(
            dataset_manifest=
                dataset_manifest
        )
    )

    expo_buffer = ReplayBuffer(
        size=
            PROJECT_STAGE2_EXPO_REPLAY_CAPACITY,

        random_seed=
            int(
                core.seed
            ),
    )

    model_rng = np.random.default_rng(
        int(
            core.seed
        )
    )

    return FormalStage2StateV1(
        seed=
            int(
                core.seed
            ),

        expo_buffer=
            expo_buffer,

        models=
            models,

        model_rng=
            model_rng,
    )


def formal_stage2_curriculum_phase(
    state: FormalStage2StateV1,
):
    """
    Resolve the curriculum phase at the START of a new episode.

    No phase transition is performed during an episode.  Therefore
    an episode sampled while total_environment_steps < 5000 remains
    a WARMUP episode even if that episode crosses the 5000-transition
    boundary.
    """

    environment_steps = int(
        state.total_environment_steps
    )

    if environment_steps < 0:
        raise FormalTrainingCoreError(
            "Stage-II environment-step counter cannot be negative"
        )

    if (
        environment_steps
        <
        PROJECT_STAGE2_CURRICULUM_WARMUP_ENV_STEPS
    ):
        return "WARMUP"

    return "FULL"


def eligible_formal_stage2_models(
    state: FormalStage2StateV1,
):
    if (
        len(
            state.models
        )
        !=
        PROJECT_STAGE2_MODEL_COUNT
        or
        PROJECT_STAGE2_MODEL_COUNT
        !=
        49
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-II model count mismatch"
        )

    phase = (
        formal_stage2_curriculum_phase(
            state
        )
    )

    if phase == "WARMUP":
        candidates = tuple(
            model
            for model in state.models
            if (
                model.complexity_stratum
                <=
                PROJECT_STAGE2_CURRICULUM_WARMUP_MAX_STRATUM
            )
        )

        if (
            len(
                candidates
            )
            !=
            PROJECT_STAGE2_CURRICULUM_WARMUP_MODEL_COUNT
            or
            len(
                candidates
            )
            !=
            39
        ):
            raise FormalTrainingCoreError(
                "Formal Stage-II warmup pool must contain exactly 39 models"
            )

        if any(
            model.complexity_stratum > 7
            for model in candidates
        ):
            raise FormalTrainingCoreError(
                "High-complexity model leaked into Stage-II warmup pool"
            )

    elif phase == "FULL":
        candidates = tuple(
            state.models
        )

        if (
            len(
                candidates
            )
            !=
            PROJECT_STAGE2_CURRICULUM_FULL_MODEL_COUNT
            or
            len(
                candidates
            )
            !=
            49
        ):
            raise FormalTrainingCoreError(
                "Formal Stage-II full pool must contain exactly 49 models"
            )

    else:
        raise FormalTrainingCoreError(
            f"Unknown Stage-II curriculum phase: {phase}"
        )

    return (
        phase,
        candidates,
    )


def sample_formal_stage2_model(
    state: FormalStage2StateV1,
):
    phase, candidates = (
        eligible_formal_stage2_models(
            state
        )
    )

    if len(candidates) <= 0:
        raise FormalTrainingCoreError(
            f"Stage-II curriculum phase {phase} has no eligible models"
        )

    # IMPORTANT:
    #
    # Sample DIRECTLY from the currently eligible pool.
    # Do not sample from Train49 and rejection-resample excluded
    # models, because that would make RNG consumption dependent on
    # the number of rejected draws.
    index = int(
        state.model_rng.integers(
            low=0,
            high=len(
                candidates
            ),
        )
    )

    return candidates[
        index
    ]


def build_formal_stage2_vector_env(
    *,
    model: FormalStage2ModelV1,
    executable: Path =
        DEFAULT_EXECUTABLE,

    resource_guard_policy=None,

    resource_guard_sample_interval_seconds: float =
        PROJECT_STAGE2_RESOURCE_GUARD_SAMPLE_INTERVAL_SECONDS,

    resource_snapshot_reader=None,

    finalize_eval_swap_abort_bytes=None,
):
    """
    Build one formal Stage-II LoopyCuts environment.

    Formal V5 production behavior enables ResourceGuard by default:

        STEP:
            warning      8 GiB SwapUsed
            hard abort  10 GiB continuously for 8 seconds
            emergency   12 GiB immediately

        FINALIZE_QUALITY:
            hard system-swap cap 25 GiB

        INITIALIZE:
            unguarded by design; every new model is already
            gated by the <=6 GiB global preflight/re-arm rule.

    RESOURCE_ABORT terminates only the current model episode.

    Test-only callers may inject a policy/snapshot reader to trigger
    deterministic synthetic resource events without consuming real
    system swap.
    """

    executable = Path(
        executable
    )

    if not executable.is_file():
        raise FileNotFoundError(
            executable
        )

    if model.quality_ref_file is None:
        raise FormalTrainingCoreError(
            "Formal V5 Stage-II model lacks quality_ref_file: "
            f"{model.model}"
        )

    quality_ref_file = Path(
        model.quality_ref_file
    ).resolve()

    if not quality_ref_file.is_file():
        raise FileNotFoundError(
            quality_ref_file
        )

    if resource_guard_policy is None:
        resource_guard_policy = (
            ResourceGuardPolicyV1(
                warning_swap_used_bytes=
                    PROJECT_STAGE2_RESOURCE_GUARD_WARNING_SWAP_GIB
                    *
                    GIB,

                abort_swap_used_bytes=
                    PROJECT_STAGE2_RESOURCE_GUARD_ABORT_SWAP_GIB
                    *
                    GIB,

                emergency_swap_used_bytes=
                    PROJECT_STAGE2_RESOURCE_GUARD_EMERGENCY_SWAP_GIB
                    *
                    GIB,

                rearm_swap_used_bytes=
                    PROJECT_STAGE2_RESOURCE_GUARD_REARM_SWAP_GIB
                    *
                    GIB,

                abort_hold_seconds=
                    PROJECT_STAGE2_RESOURCE_GUARD_ABORT_HOLD_SECONDS,
            )
        )

    if finalize_eval_swap_abort_bytes is None:
        finalize_eval_swap_abort_bytes = (
            PROJECT_STAGE2_RESOURCE_GUARD_FINALIZE_QUALITY_SWAP_ABORT_GIB
            *
            GIB
        )

    finalize_eval_swap_abort_bytes = int(
        finalize_eval_swap_abort_bytes
    )

    if finalize_eval_swap_abort_bytes <= 0:
        raise ValueError(
            "finalize_eval_swap_abort_bytes must be positive"
        )

    if not isinstance(
        resource_guard_policy,
        ResourceGuardPolicyV1,
    ):
        raise TypeError(
            "resource_guard_policy must be "
            "ResourceGuardPolicyV1"
        )

    def make_env():
        return (
            FormalEpisodeCollectorBridgeV1(
                FinalRewardWrapperV5(
                    FinalizationQualityWrapperV1(
                        LoopyCutsEnv(
                            executable=
                                executable,

                            mesh_file=
                                model.mesh_file,

                            loop_file=
                                model.loop_file,

                            echo_logs=
                                False,

                            resource_guard_policy=
                                resource_guard_policy,

                            resource_guard_sample_interval_seconds=
                                resource_guard_sample_interval_seconds,

                            resource_snapshot_reader=
                                resource_snapshot_reader,

                            finalize_eval_swap_abort_bytes=
                                finalize_eval_swap_abort_bytes,
                        ),

                        quality_ref_path=
                            quality_ref_file,

                        expected_model=
                            model.model,
                    )
                )
            )
        )

    return DummyVectorEnv(
        [
            make_env
        ]
    )

def _collect_stats_steps(
    stats,
):
    if isinstance(
        stats,
        dict,
    ):
        return int(
            stats[
                "n_collected_steps"
            ]
        )

    return int(
        stats.n_collected_steps
    )


def _single_int(
    value,
):
    return int(
        np.asarray(
            value
        )
        .reshape(
            -1
        )[
            0
        ]
    )


def _single_float(
    value,
):
    return float(
        np.asarray(
            value
        )
        .reshape(
            -1
        )[
            0
        ]
    )


def _single_bool(
    value,
):
    return bool(
        np.asarray(
            value
        )
        .reshape(
            -1
        )[
            0
        ]
    )


def _single_str(
    value,
):
    return str(
        np.asarray(
            value
        )
        .reshape(
            -1
        )[
            0
        ]
    )


def _empty_terminal_quality_episode_record():
    return {
        "available": False,
        "model": "",
        "hex": -1,
        "total_polys": -1,
        "nonhex": -1,
        "d_c": 0.0,
        "q_missing": 0.0,
        "q_spurious": 0.0,
        "q_shape": 0.0,
        "sharp_active": 0,
        "sharp_metrics_valid": 0,
        "q_sharp_available": False,
        "q_sharp": 0.0,
        "q_fidelity": 0.0,
        "utility": 0.0,
    }


def _empty_terminal_reward_v5_episode_record():
    return {
        "available": False,
        "step": 0.0,
        "tet_growth": 0.0,
        "revert": 0.0,
        "convergence": 0.0,
        "quality_available": False,
        "utility": 0.0,
        "terminal": 0.0,
        "total": 0.0,
    }


def _extract_terminal_v5_episode_telemetry(
    *,
    info,
    expected_model,
    reward,
    finalization_outcome,
):
    if not hasattr(
        info,
        "terminal_quality",
    ):
        raise FormalTrainingCoreError(
            "Terminal Stage-II transition lacks "
            "terminal_quality telemetry"
        )

    if not hasattr(
        info,
        "reward_v5_breakdown",
    ):
        raise FormalTrainingCoreError(
            "Terminal Stage-II transition lacks "
            "reward_v5_breakdown telemetry"
        )

    quality = (
        info.terminal_quality
    )

    breakdown = (
        info.reward_v5_breakdown
    )

    terminal_quality = {
        "available":
            _single_bool(
                quality.available
            ),

        "model":
            _single_str(
                quality.model
            ),

        "hex":
            _single_int(
                quality.hex
            ),

        "total_polys":
            _single_int(
                quality.total_polys
            ),

        "nonhex":
            _single_int(
                quality.nonhex
            ),

        "d_c":
            _single_float(
                quality.d_c
            ),

        "q_missing":
            _single_float(
                quality.q_missing
            ),

        "q_spurious":
            _single_float(
                quality.q_spurious
            ),

        "q_shape":
            _single_float(
                quality.q_shape
            ),

        "sharp_active":
            _single_int(
                quality.sharp_active
            ),

        "sharp_metrics_valid":
            _single_int(
                quality.sharp_metrics_valid
            ),

        "q_sharp_available":
            _single_bool(
                quality.q_sharp_available
            ),

        "q_sharp":
            _single_float(
                quality.q_sharp
            ),

        "q_fidelity":
            _single_float(
                quality.q_fidelity
            ),

        "utility":
            _single_float(
                quality.utility
            ),
    }

    terminal_reward_v5 = {
        "available":
            True,

        "step":
            _single_float(
                breakdown.step
            ),

        "tet_growth":
            _single_float(
                breakdown.tet_growth
            ),

        "revert":
            _single_float(
                breakdown.revert
            ),

        "convergence":
            _single_float(
                breakdown.convergence
            ),

        "quality_available":
            _single_bool(
                breakdown.quality_available
            ),

        "utility":
            _single_float(
                breakdown.utility
            ),

        "terminal":
            _single_float(
                breakdown.terminal
            ),

        "total":
            _single_float(
                breakdown.total
            ),
    }

    # The persisted reward must be the exact reward inserted
    # into D_expo for this terminal transition.
    if (
        terminal_reward_v5[
            "total"
        ]
        !=
        float(
            reward
        )
    ):
        raise FormalTrainingCoreError(
            "Terminal Reward V5 telemetry does not "
            "exactly match replay reward"
        )

    successful = (
        finalization_outcome
        in {
            "FULL_HEX",
            "NON_FULL_HEX",
        }
    )

    if successful:
        if not terminal_quality[
            "available"
        ]:
            raise FormalTrainingCoreError(
                "Successful finalization lacks "
                "terminal quality"
            )

        if (
            terminal_quality[
                "model"
            ]
            !=
            str(
                expected_model
            )
        ):
            raise FormalTrainingCoreError(
                "Terminal quality model mismatch"
            )

        expected_utility = (
            terminal_quality[
                "d_c"
            ]
            *
            terminal_quality[
                "q_fidelity"
            ]
        )

        if (
            terminal_quality[
                "utility"
            ]
            !=
            expected_utility
        ):
            raise FormalTrainingCoreError(
                "Terminal quality utility is not "
                "exactly D_C * Q_fidelity"
            )

        if not terminal_reward_v5[
            "quality_available"
        ]:
            raise FormalTrainingCoreError(
                "Successful Reward V5 terminal "
                "lacks quality"
            )

        if (
            terminal_reward_v5[
                "utility"
            ]
            !=
            terminal_quality[
                "utility"
            ]
        ):
            raise FormalTrainingCoreError(
                "Reward V5 utility does not exactly "
                "match terminal quality utility"
            )

    else:
        if (
            finalization_outcome
            not in {
                "FINALIZATION_CRASH",
                "RESOURCE_ABORT",
            }
        ):
            raise FormalTrainingCoreError(
                "Unknown terminal outcome while "
                "extracting V5 telemetry"
            )

        if terminal_quality[
            "available"
        ]:
            raise FormalTrainingCoreError(
                "Fatal terminal outcome must not "
                "carry terminal quality"
            )

        if terminal_reward_v5[
            "quality_available"
        ]:
            raise FormalTrainingCoreError(
                "Fatal terminal Reward V5 must not "
                "claim quality availability"
            )

    return (
        terminal_quality,
        terminal_reward_v5,
    )


def flush_formal_stage2_updates(
    core: FormalTrainingCoreV1,
    state: FormalStage2StateV1,
    *,
    metrics_writer:
        TrainingMetricsWriterV1
        | None
        = None,
    episode_index:
        int
        | None
        = None,
):
    if (
        core.stage
        !=
        "STAGE_II"
    ):
        raise FormalTrainingCoreError(
            "Stage-II updates require STAGE_II state"
        )

    if (
        core.algorithm.bc_enabled
    ):
        raise FormalTrainingCoreError(
            "Stage-II updates require BC disabled"
        )

    pending_updates = (
        state.total_environment_steps
        -
        state.total_gradient_updates
    )

    if pending_updates < 0:
        raise FormalTrainingCoreError(
            "Stage-II gradient updates exceed collected transitions"
        )

    if pending_updates == 0:
        return {
            "gradient_updates":
                0,

            "final_training_stats":
                None,
        }

    if (
        len(
            state.expo_buffer
        )
        <=
        0
    ):
        raise FormalTrainingCoreError(
            "Cannot update from empty D_expo"
        )

    final_snapshot = None

    with policy_within_training_step(
        core.policy
    ):
        for episode_update_index in range(
            1,
            pending_updates + 1,
        ):
            stats, mix = (
                core.algorithm.update_equal_replay(
                    demo_buffer=
                        core.demo_buffer,

                    expo_buffer=
                        state.expo_buffer,

                    samples_per_buffer=
                        PROJECT_STAGE2_SAMPLES_PER_BUFFER,
                )
            )

            if (
                mix[
                    "demo_samples"
                ]
                !=
                PROJECT_STAGE2_SAMPLES_PER_BUFFER
            ):
                raise FormalTrainingCoreError(
                    "Stage-II D_demo sample count mismatch"
                )

            if (
                mix[
                    "expo_samples"
                ]
                !=
                PROJECT_STAGE2_SAMPLES_PER_BUFFER
            ):
                raise FormalTrainingCoreError(
                    "Stage-II D_expo sample count mismatch"
                )

            if (
                mix[
                    "total_samples"
                ]
                !=
                2
                *
                PROJECT_STAGE2_SAMPLES_PER_BUFFER
            ):
                raise FormalTrainingCoreError(
                    "Stage-II mixed minibatch size mismatch"
                )

            snapshot = training_stats_snapshot(
                stats
            )

            if (
                float(
                    snapshot.get(
                        "bc_loss",
                        0.0,
                    )
                )
                !=
                0.0
            ):
                raise FormalTrainingCoreError(
                    "BC loss became non-zero during Stage-II"
                )

            state.total_gradient_updates += 1

            if metrics_writer is not None:
                metrics_writer.append(
                    seed=
                        core.seed,

                    stage=
                        "STAGE_II",

                    gradient_update=
                        state.total_gradient_updates,

                    environment_steps=
                        state.total_environment_steps,

                    episode_index=(
                        int(episode_index)
                        if episode_index is not None
                        else None
                    ),

                    episode_update_index=
                        episode_update_index,

                    episode_update_count=
                        pending_updates,

                    stats=
                        snapshot,
                )

            final_snapshot = snapshot

    if (
        state.total_gradient_updates
        >
        state.total_environment_steps
    ):
        raise FormalTrainingCoreError(
            "Stage-II update/collection accounting mismatch"
        )

    return {
        "gradient_updates":
            pending_updates,

        "final_training_stats":
            final_snapshot,
    }


def collect_formal_stage2_model_episode(
    core: FormalTrainingCoreV1,
    state: FormalStage2StateV1,
    *,
    model: FormalStage2ModelV1,
    executable: Path =
        DEFAULT_EXECUTABLE,

    metrics_writer:
        TrainingMetricsWriterV1
        | None
        = None,

    resource_guard_policy=None,

    resource_guard_sample_interval_seconds: float =
        PROJECT_STAGE2_RESOURCE_GUARD_SAMPLE_INTERVAL_SECONDS,

    resource_snapshot_reader=None,

    finalize_eval_swap_abort_bytes=None,
):
    """
    Collect exactly one native LoopyCuts episode, except when the
    frozen global 25,000-transition budget is reached first.

    A budget-boundary prefix is NOT marked terminated/truncated.

    After collection stops, perform exactly one Stage-II SAC update
    for every newly collected transition.

    Normal case:
        completed episode of N transitions
        -> N updates

    Final budget case:
        M-transition partial prefix reaches exactly 25,000
        -> M updates

    The final flush preserves the already-frozen global 1.0
    gradient-update / collected-transition ratio.
    """

    if (
        core.stage
        !=
        "STAGE_II"
    ):
        raise FormalTrainingCoreError(
            "Online collection requires STAGE_II"
        )

    if (
        core.algorithm.bc_enabled
    ):
        raise FormalTrainingCoreError(
            "Online Stage-II collection requires BC OFF"
        )

    if (
        state.total_environment_steps
        >=
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-II environment budget is already exhausted"
        )

    if (
        len(
            state.expo_buffer
        )
        !=
        state.total_environment_steps
    ):
        raise FormalTrainingCoreError(
            "D_expo size does not match formal collection counter"
        )

    episode_index = (
        state.episode_attempts
        +
        1
    )

    state.episode_attempts = (
        episode_index
    )

    environment_steps_before = (
        state.total_environment_steps
    )

    # Curriculum phase is frozen at episode start.
    #
    # Even if this episode crosses env=5000, the phase and eligible
    # pool recorded for this episode do not change.
    (
        curriculum_phase,
        eligible_models,
    ) = eligible_formal_stage2_models(
        state
    )

    eligible_model_count = len(
        eligible_models
    )

    if (
        model
        not in
        eligible_models
    ):
        raise FormalTrainingCoreError(
            "Requested Stage-II model is not eligible in current "
            f"curriculum phase: phase={curriculum_phase}, "
            f"model={model.model}, "
            f"stratum={model.complexity_stratum}"
        )

    replay_size_before = len(
        state.expo_buffer
    )

    actions = []

    episode_return = 0.0

    terminated = False
    truncated = False

    finalization_outcome = (
        "NONE"
    )

    resource_abort = False

    resource_guard_state = ""
    resource_guard_phase = ""

    resource_guard_swap_used_bytes = 0
    resource_guard_mem_available_bytes = 0

    resource_guard_python_rss_bytes = 0
    resource_guard_python_swap_bytes = 0

    resource_guard_cpp_rss_bytes = 0
    resource_guard_cpp_swap_bytes = 0

    terminal_quality_record = (
        _empty_terminal_quality_episode_record()
    )

    terminal_reward_v5_record = (
        _empty_terminal_reward_v5_episode_record()
    )

    vector_env = (
        build_formal_stage2_vector_env(
            model=
                model,

            executable=
                executable,

            resource_guard_policy=
                resource_guard_policy,

            resource_guard_sample_interval_seconds=
                resource_guard_sample_interval_seconds,

            resource_snapshot_reader=
                resource_snapshot_reader,

            finalize_eval_swap_abort_bytes=
                finalize_eval_swap_abort_bytes,
        )
    )

    collector = Collector(
        core.algorithm,
        vector_env,
        state.expo_buffer,

        exploration_noise=
            PROJECT_STAGE2_COLLECTOR_EXPLORATION_NOISE,
    )

    try:
        # CRITICAL TIanshou 2.0.1 semantic:
        #
        # Collector.reset() defaults reset_buffer=True.
        # D_expo must accumulate across all Train49 episodes.
        collector.reset(
            reset_buffer=
                False,

            reset_stats=
                True,
        )

        if (
            len(
                state.expo_buffer
            )
            !=
            replay_size_before
        ):
            raise FormalTrainingCoreError(
                "Collector reset unexpectedly changed D_expo"
            )

        while (
            state.total_environment_steps
            <
            PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
        ):
            before = len(
                state.expo_buffer
            )

            stats = collector.collect(
                n_step=1
            )

            after = len(
                state.expo_buffer
            )

            if (
                _collect_stats_steps(
                    stats
                )
                !=
                1
            ):
                raise FormalTrainingCoreError(
                    "collect(n_step=1) did not collect exactly one step"
                )

            if (
                after
                !=
                before + 1
            ):
                raise FormalTrainingCoreError(
                    "D_expo did not grow by exactly one transition"
                )

            transition_index = (
                after
                -
                1
            )

            transition = state.expo_buffer[
                np.asarray(
                    [
                        transition_index
                    ],
                    dtype=np.int64,
                )
            ]

            action = _single_int(
                transition.act
            )

            reward = _single_float(
                transition.rew
            )

            terminated = _single_bool(
                transition.terminated
            )

            truncated = _single_bool(
                transition.truncated
            )

            if (
                truncated
                is not
                False
            ):
                raise FormalTrainingCoreError(
                    "LoopyCuts formal Stage-II must not use truncation"
                )

            current_mask = np.asarray(
                transition.obs.mask,
                dtype=np.bool_,
            ).reshape(
                1,
                -1,
            )

            if not bool(
                current_mask[
                    0,
                    action,
                ]
            ):
                raise FormalTrainingCoreError(
                    "Collector stored an action illegal under its dynamic mask"
                )

            if not math.isfinite(
                reward
            ):
                raise FormalTrainingCoreError(
                    "Non-finite Stage-II reward"
                )

            if not hasattr(
                transition.info,
                "reward_version",
            ):
                raise FormalTrainingCoreError(
                    "Stage-II transition lacks reward_version"
                )

            reward_version = str(
                np.asarray(
                    transition.info.reward_version
                )
                .reshape(
                    -1
                )[
                    0
                ]
            )

            if (
                reward_version
                !=
                REWARD_V5_VERSION
            ):
                raise FormalTrainingCoreError(
                    "Formal Stage-II did not collect Reward V5"
                )

            if not hasattr(
                transition.info,
                "resource_guard",
            ):
                raise FormalTrainingCoreError(
                    "Stage-II transition lacks fixed "
                    "ResourceGuard record"
                )

            resource_abort_step = (
                _single_bool(
                    transition
                    .info
                    .resource_guard
                    .triggered
                )
            )

            if (
                resource_abort_step
                and
                not terminated
            ):
                raise FormalTrainingCoreError(
                    "RESOURCE_ABORT transition must be terminal"
                )

            if resource_abort_step:
                if resource_abort:
                    raise FormalTrainingCoreError(
                        "Episode contains multiple "
                        "RESOURCE_ABORT transitions"
                    )

                resource_abort = True

                resource_guard_phase = str(
                    np.asarray(
                        transition
                        .info
                        .resource_guard
                        .phase
                    )
                    .reshape(
                        -1
                    )[
                        0
                    ]
                )

                if resource_guard_phase not in {
                    "STEP",
                    "FINALIZE_QUALITY",
                }:
                    raise FormalTrainingCoreError(
                        "Unknown ResourceGuard phase: "
                        f"{resource_guard_phase!r}"
                    )

                resource_guard_state = str(
                    np.asarray(
                        transition
                        .info
                        .resource_guard
                        .guard_state
                    )
                    .reshape(
                        -1
                    )[
                        0
                    ]
                )

                resource_guard_swap_used_bytes = (
                    _single_int(
                        transition
                        .info
                        .resource_guard
                        .swap_used_bytes
                    )
                )

                resource_guard_mem_available_bytes = (
                    _single_int(
                        transition
                        .info
                        .resource_guard
                        .mem_available_bytes
                    )
                )

                resource_guard_python_rss_bytes = (
                    _single_int(
                        transition
                        .info
                        .resource_guard
                        .python_rss_bytes
                    )
                )

                resource_guard_python_swap_bytes = (
                    _single_int(
                        transition
                        .info
                        .resource_guard
                        .python_swap_bytes
                    )
                )

                resource_guard_cpp_rss_bytes = (
                    _single_int(
                        transition
                        .info
                        .resource_guard
                        .cpp_rss_bytes
                    )
                )

                resource_guard_cpp_swap_bytes = (
                    _single_int(
                        transition
                        .info
                        .resource_guard
                        .cpp_swap_bytes
                    )
                )

                if not math.isclose(
                    reward,
                    -4.0,
                    rel_tol=0.0,
                    abs_tol=0.0,
                ):
                    raise FormalTrainingCoreError(
                        "RESOURCE_ABORT formal reward "
                        "must be exactly -4"
                    )

            actions.append(
                action
            )

            episode_return += (
                reward
            )

            state.total_environment_steps += 1

            if (
                len(
                    state.expo_buffer
                )
                !=
                state.total_environment_steps
            ):
                raise FormalTrainingCoreError(
                    "D_expo/global-step accounting mismatch"
                )

            if terminated:
                if not hasattr(
                    transition.info,
                    "finalization_outcome",
                ):
                    raise FormalTrainingCoreError(
                        "Terminal Stage-II transition lacks finalization outcome"
                    )

                finalization_outcome = str(
                    np.asarray(
                        transition
                        .info
                        .finalization_outcome
                        .outcome
                    )
                    .reshape(
                        -1
                    )[
                        0
                    ]
                )

                if finalization_outcome not in {
                    "FULL_HEX",
                    "NON_FULL_HEX",
                    "FINALIZATION_CRASH",
                    "RESOURCE_ABORT",
                }:
                    raise FormalTrainingCoreError(
                        "Unknown terminal outcome: "
                        f"{finalization_outcome}"
                    )

                if (
                    finalization_outcome
                    ==
                    "RESOURCE_ABORT"
                    and
                    not resource_abort
                ):
                    raise FormalTrainingCoreError(
                        "RESOURCE_ABORT outcome lacks "
                        "ResourceGuard trigger"
                    )

                if (
                    resource_abort
                    and
                    finalization_outcome
                    !=
                    "RESOURCE_ABORT"
                ):
                    raise FormalTrainingCoreError(
                        "ResourceGuard trigger has inconsistent "
                        "terminal outcome"
                    )

                (
                    terminal_quality_record,
                    terminal_reward_v5_record,
                ) = (
                    _extract_terminal_v5_episode_telemetry(
                        info=
                            transition.info,

                        expected_model=
                            model.model,

                        reward=
                            reward,

                        finalization_outcome=
                            finalization_outcome,
                    )
                )

                break

            if (
                state.total_environment_steps
                ==
                PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
            ):
                # Exact budget boundary.
                #
                # Deliberately do NOT change the stored transition
                # to terminated=True or truncated=True.
                break

    finally:
        collector.close()

    episode_steps = (
        state.total_environment_steps
        -
        environment_steps_before
    )

    if episode_steps <= 0:
        raise FormalTrainingCoreError(
            "Stage-II episode collected zero transitions"
        )

    budget_exhausted = (
        state.total_environment_steps
        ==
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
    )

    completed = bool(
        terminated
    )

    if completed:
        state.completed_episodes += 1

    update_result = (
        flush_formal_stage2_updates(
            core,
            state,

            metrics_writer=
                metrics_writer,

            episode_index=
                episode_index,
        )
    )

    if (
        update_result[
            "gradient_updates"
        ]
        !=
        episode_steps
    ):
        raise FormalTrainingCoreError(
            "Stage-II updates do not equal newly collected transitions"
        )

    if (
        state.total_gradient_updates
        !=
        state.total_environment_steps
    ):
        raise FormalTrainingCoreError(
            "Stage-II global update ratio is not exactly 1.0"
        )

    record = {
        "episode_index":
            episode_index,

        "model":
            model.model,

        "model_complexity_stratum":
            int(
                model.complexity_stratum
            ),

        "curriculum_phase":
            curriculum_phase,

        "eligible_model_count":
            int(
                eligible_model_count
            ),

        "environment_steps_before":
            int(
                environment_steps_before
            ),

        "mesh_file":
            str(
                model.mesh_file
            ),

        "loop_file":
            str(
                model.loop_file
            ),

        "completed":
            completed,

        "budget_exhausted":
            budget_exhausted,

        "terminated":
            bool(
                terminated
            ),

        "truncated":
            bool(
                truncated
            ),

        "steps":
            episode_steps,

        "actions":
            [
                int(
                    action
                )
                for action in actions
            ],

        "episode_return":
            float(
                episode_return
            ),

        "finalization_outcome":
            finalization_outcome,

        "terminal_quality":
            terminal_quality_record,

        "terminal_reward_v5":
            terminal_reward_v5_record,

        "resource_abort":
            bool(
                resource_abort
            ),

        "resource_guard_phase":
            str(
                resource_guard_phase
            ),

        "resource_guard_state":
            str(
                resource_guard_state
            ),

        "resource_guard_swap_used_bytes":
            int(
                resource_guard_swap_used_bytes
            ),

        "resource_guard_mem_available_bytes":
            int(
                resource_guard_mem_available_bytes
            ),

        "resource_guard_python_rss_bytes":
            int(
                resource_guard_python_rss_bytes
            ),

        "resource_guard_python_swap_bytes":
            int(
                resource_guard_python_swap_bytes
            ),

        "resource_guard_cpp_rss_bytes":
            int(
                resource_guard_cpp_rss_bytes
            ),

        "resource_guard_cpp_swap_bytes":
            int(
                resource_guard_cpp_swap_bytes
            ),

        "gradient_updates":
            int(
                update_result[
                    "gradient_updates"
                ]
            ),

        "total_environment_steps":
            state.total_environment_steps,

        "total_gradient_updates":
            state.total_gradient_updates,

        "expo_buffer_size":
            len(
                state.expo_buffer
            ),

        "final_training_stats":
            update_result[
                "final_training_stats"
            ],
    }

    state.history.append(
        record
    )

    return record


def run_next_formal_stage2_episode(
    core: FormalTrainingCoreV1,
    state: FormalStage2StateV1,
    *,
    executable: Path =
        DEFAULT_EXECUTABLE,

    metrics_writer:
        TrainingMetricsWriterV1
        | None
        = None,
):
    model = (
        sample_formal_stage2_model(
            state
        )
    )

    return (
        collect_formal_stage2_model_episode(
            core,
            state,

            model=
                model,

            executable=
                executable,

            metrics_writer=
                metrics_writer,
        )
    )


def run_formal_stage2_to_budget(
    core: FormalTrainingCoreV1,
    state: FormalStage2StateV1,
    *,
    executable: Path =
        DEFAULT_EXECUTABLE,

    metrics_writer:
        TrainingMetricsWriterV1
        | None
        = None,
):
    while (
        state.total_environment_steps
        <
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
    ):
        record = (
            run_next_formal_stage2_episode(
                core,
                state,

                executable=
                    executable,

                metrics_writer=
                    metrics_writer,
            )
        )

        print(
            "formal-stage2 "
            f"episode={record['episode_index']} "
            f"model={record['model']} "
            f"steps={record['steps']} "
            f"outcome={record['finalization_outcome']} "
            f"total_env={record['total_environment_steps']}/"
            f"{PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS} "
            f"total_updates={record['total_gradient_updates']}"
        )

    if (
        state.total_environment_steps
        !=
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-II did not stop at exact transition budget"
        )

    if (
        state.total_gradient_updates
        !=
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
    ):
        raise FormalTrainingCoreError(
            "Formal Stage-II did not execute exact 1.0 update ratio"
        )

    if (
        len(
            state.expo_buffer
        )
        !=
        PROJECT_STAGE2_TOTAL_ENVIRONMENT_STEPS
    ):
        raise FormalTrainingCoreError(
            "Formal D_expo did not end at exact frozen capacity"
        )

    return {
        "total_environment_steps":
            state.total_environment_steps,

        "total_gradient_updates":
            state.total_gradient_updates,

        "episode_attempts":
            state.episode_attempts,

        "completed_episodes":
            state.completed_episodes,

        "expo_buffer_size":
            len(
                state.expo_buffer
            ),

        "history":
            state.history,
    }
