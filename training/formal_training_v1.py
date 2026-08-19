from __future__ import annotations

import math
import random
import time

from dataclasses import dataclass
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

from tianshou.utils.torch_utils import (
    policy_within_training_step,
)


from algorithms.demo_guided_discrete_sac_v1 import (
    LoopyCutsDemoGuidedDiscreteSACV1,
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

from training.formal_training_input_provenance_v1 import (
    assert_formal_training_input_provenance,
)

from training.masked_auto_alpha_v1 import (
    MaskedAutoAlphaV1,
)

from training.protocol_v1 import (
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
    PROJECT_STAGE2_SAMPLES_PER_BUFFER,

    PROJECT_NETWORK_REINITIALIZATION,

    assert_formal_training_ready,
)


FORMAL_TRAINER_CORE_VERSION = (
    "loopycuts_formal_training_core_v1"
)


DEFAULT_EXECUTABLE = Path(
    "/home/yjk/codes/LoopyCuts/"
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
        assert_formal_training_input_provenance(
            provenance_path=
                input_provenance_path,

            executable=
                executable,

            dataset_manifest=
                dataset_manifest,

            demo_quality_manifest=
                demo_quality_manifest,

            raw_demo_root=
                raw_demo_root,
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
