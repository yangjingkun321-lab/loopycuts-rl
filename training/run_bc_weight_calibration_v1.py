from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
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


import gymnasium as gym
import numpy as np
import torch
import tianshou

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

from tianshou.utils.torch_utils import (
    policy_within_training_step,
)


from algorithms.demo_guided_discrete_sac_v1 import (
    LoopyCutsDemoGuidedDiscreteSACV1,
)

from envs.final_reward_wrapper import (
    FinalRewardWrapper,
)

from envs.finalization_eval_wrapper import (
    FinalizationEvalWrapper,
)

from envs.loopycuts_env import (
    LoopyCutsEnv,
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

from training.bc_weight_calibration_v1 import (
    CALIBRATION_RESULT_SCHEMA_VERSION,
    CalibrationEpisodeResult,
    select_best_bc_weight,
    validate_complete_result_grid,
    validate_episode_result,
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

    PROJECT_BC_WEIGHT_CALIBRATION_VERSION,
    PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES,
    PROJECT_BC_WEIGHT_CALIBRATION_SEEDS,
    PROJECT_BC_WEIGHT_CALIBRATION_SPLIT,
    PROJECT_BC_WEIGHT_CALIBRATION_MODELS,
    PROJECT_BC_WEIGHT_CALIBRATION_STAGE1_GRADIENT_STEPS,
    PROJECT_BC_WEIGHT_CALIBRATION_STAGE2_ENABLED,
    PROJECT_BC_WEIGHT_CALIBRATION_EVAL_DETERMINISTIC,
    PROJECT_BC_WEIGHT_CALIBRATION_EVAL_EPSILON,
    PROJECT_BC_WEIGHT_CALIBRATION_DEVICE,

    UNRESOLVED_BC_WEIGHT,
    FORMAL_TRAINING_BLOCKERS,
    formal_training_ready,
)


RUNNER_VERSION = (
    "bc_weight_calibration_runner_v2"
)


DEFAULT_EXECUTABLE = Path(
    "/home/yjk/codes/LoopyCuts/"
    "volumetric_cutter/"
    "volumetric_cutter"
)

DEFAULT_MANIFEST = Path(
    "data/manifests/"
    "dataset_split_v2.csv"
)

DEFAULT_RAW_DEMO_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "demonstrations_v1/raw/train"
)

DEFAULT_DEMO_QUALITY = Path(
    "data/manifests/"
    "demo_quality_v1.csv"
)

DEFAULT_SMOKE_OUTPUT_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "bc_weight_calibration_smoke_v2"
)

DEFAULT_FORMAL_OUTPUT_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "bc_weight_calibration_v2/"
    "formal_pairs"
)

DEFAULT_SELECTION_OUTPUT = Path(
    "/home/yjk/loopycuts_test/"
    "bc_weight_calibration_v2/"
    "bc_weight_selection_v1.json"
)


class CalibrationRunnerError(
    RuntimeError
):
    pass


def sha256_file(
    path: Path,
):
    path = Path(
        path
    )

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        for chunk in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def canonical_formal_candidate(
    value: float,
):
    value = float(
        value
    )

    matches = [
        float(
            candidate
        )
        for candidate in
        PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES
        if math.isclose(
            value,
            float(
                candidate
            ),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ]

    if len(
        matches
    ) != 1:
        raise CalibrationRunnerError(
            "Formal bc_weight must be exactly "
            "one of the frozen candidates: "
            f"{PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES}"
        )

    return matches[
        0
    ]


def formal_pair_output_name(
    *,
    bc_weight: float,
    seed: int,
):
    bc_weight = (
        canonical_formal_candidate(
            bc_weight
        )
    )

    weight_text = (
        f"{bc_weight:g}"
        .replace(
            ".",
            "p",
        )
    )

    return (
        f"formal_lambda_{weight_text}_"
        f"seed_{int(seed)}.json"
    )


def atomic_write_json(
    *,
    path: Path,
    payload,
):
    path = Path(
        path
    )

    if path.exists():
        raise CalibrationRunnerError(
            "Refusing to overwrite existing "
            f"artifact: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        path.with_suffix(
            path.suffix
            +
            ".tmp"
        )
    )

    if temp_path.exists():
        raise CalibrationRunnerError(
            "Temporary artifact already exists; "
            "diagnose the previous interrupted run: "
            f"{temp_path}"
        )

    temp_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    temp_path.replace(
        path
    )


def git_head() -> str:
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


def assert_clean_repository():
    status = (
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

    if status.strip():
        raise CalibrationRunnerError(
            "Calibration execution requires "
            "a clean Git working tree"
        )


def assert_protocol():
    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_VERSION
        ==
        "bc_weight_calibration_v2"
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES
        ==
        (
            0.1,
            0.3,
            0.5,
            1.0,
            3.0,
        )
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_SEEDS
        ==
        (
            42,
            43,
            44,
        )
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_STAGE1_GRADIENT_STEPS
        ==
        782
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_STAGE2_ENABLED
        is False
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_EVAL_DETERMINISTIC
        is True
    )

    assert math.isclose(
        PROJECT_BC_WEIGHT_CALIBRATION_EVAL_EPSILON,
        0.0,
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_DEVICE
        ==
        "cpu"
    )

    assert PAPER_BATCH_SIZE == 64

    assert math.isclose(
        PAPER_DISCOUNT_FACTOR,
        0.95,
    )

    assert math.isclose(
        PAPER_SOFT_UPDATE_TAU,
        0.005,
    )

    assert PROJECT_N_STEP_RETURN_HORIZON == 1

    assert PROJECT_MAIN_DEMO_EPISODES == 30
    assert PROJECT_MAIN_DEMO_TRANSITIONS == 605

    assert UNRESOLVED_BC_WEIGHT is None

    assert FORMAL_TRAINING_BLOCKERS == (
        "bc_weight",
    )

    assert formal_training_ready() is False


def set_run_seed(
    seed: int,
):
    seed = int(
        seed
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


def make_actor_critic_adamw_factory(
    *,
    lr: float,
):
    if (
        PROJECT_ACTOR_CRITIC_OPTIMIZER_FAMILY
        !=
        "ADAMW"
    ):
        raise CalibrationRunnerError(
            "Frozen actor/critic optimizer must be ADAMW"
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


def make_alpha_adam_factory(
    *,
    lr: float,
):
    if (
        PROJECT_ALPHA_OPTIMIZER_FAMILY
        !=
        "ADAM"
    ):
        raise CalibrationRunnerError(
            "Frozen alpha optimizer must be ADAM"
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


def build_stage1_algorithm(
    *,
    bc_weight: float,
    seed: int,
):
    set_run_seed(
        seed
    )

    device = (
        PROJECT_BC_WEIGHT_CALIBRATION_DEVICE
    )

    actor, critic1, critic2 = (
        build_loopycuts_actor_critics_v1(
            device=
                device
        )
    )

    policy = (
        MaskedDiscreteSACPolicy(
            actor=
                actor,

            action_space=
                gym.spaces.Discrete(
                    MAX_LOOPS
                ),

            deterministic_eval=
                PROJECT_BC_WEIGHT_CALIBRATION_EVAL_DETERMINISTIC,

            exploration_epsilon=
                PROJECT_BC_WEIGHT_CALIBRATION_EVAL_EPSILON,

            exploration_seed=
                int(
                    seed
                ),
        )
    )

    auto_alpha = (
        MaskedAutoAlphaV1(
            target_coefficient=
                PAPER_ENTROPY_TARGET_COEFFICIENT,

            initial_alpha=
                PROJECT_INITIAL_ALPHA,

            optim=
                make_alpha_adam_factory(
                    lr=
                        PROJECT_ALPHA_LEARNING_RATE
                ),

            device=
                device,
        )
    )

    algorithm = (
        LoopyCutsDemoGuidedDiscreteSACV1(
            policy=
                policy,

            policy_optim=
                make_actor_critic_adamw_factory(
                    lr=
                        PROJECT_ACTOR_LEARNING_RATE
                ),

            critic=
                critic1,

            critic_optim=
                make_actor_critic_adamw_factory(
                    lr=
                        PROJECT_CRITIC1_LEARNING_RATE
                ),

            critic2=
                critic2,

            critic2_optim=
                make_actor_critic_adamw_factory(
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
                float(
                    bc_weight
                ),

            bc_enabled=
                True,
        )
    )

    return (
        algorithm,
        policy,
        auto_alpha,
    )


def stats_snapshot(
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
            raise CalibrationRunnerError(
                f"non-finite training stat "
                f"{name}={numeric}"
            )

        result[
            name
        ] = numeric

    return result


def run_stage1(
    *,
    algorithm,
    policy,
    demo_buffer,
):
    total_updates = int(
        PROJECT_BC_WEIGHT_CALIBRATION_STAGE1_GRADIENT_STEPS
    )

    final_snapshot = None

    start = time.time()

    with policy_within_training_step(
        policy
    ):
        for update_index in range(
            1,
            total_updates + 1,
        ):
            stats = (
                algorithm.update(
                    demo_buffer,
                    sample_size=
                        PAPER_BATCH_SIZE,
                )
            )

            snapshot = (
                stats_snapshot(
                    stats
                )
            )

            if (
                update_index == 1
                or
                update_index == total_updates
                or
                update_index % 100 == 0
            ):
                print(
                    "stage1"
                    f" update={update_index:4d}/"
                    f"{total_updates}"
                    f" actor_loss="
                    f"{snapshot.get('actor_loss')}"
                    f" bc_loss="
                    f"{snapshot.get('bc_loss')}"
                    f" alpha="
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

    if final_snapshot is None:
        raise CalibrationRunnerError(
            "Stage-I produced no updates"
        )

    return {
        "gradient_updates":
            total_updates,

        "sampled_demo_transitions":
            (
                total_updates
                *
                PAPER_BATCH_SIZE
            ),

        "elapsed_seconds":
            float(
                elapsed
            ),

        "final_training_stats":
            final_snapshot,
    }


def load_engineering_model_row(
    *,
    manifest: Path,
    model: str,
):
    manifest = Path(
        manifest
    )

    if not manifest.is_file():
        raise FileNotFoundError(
            manifest
        )

    with manifest.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = [
            dict(
                row
            )
            for row in csv.DictReader(
                f
            )
            if row[
                "model"
            ] == model
        ]

    if len(
        rows
    ) != 1:
        raise CalibrationRunnerError(
            f"Expected exactly one manifest "
            f"row for {model}; got {len(rows)}"
        )

    row = rows[
        0
    ]

    if (
        row[
            "split"
        ]
        !=
        PROJECT_BC_WEIGHT_CALIBRATION_SPLIT
    ):
        raise CalibrationRunnerError(
            f"{model}: split={row['split']!r}, "
            f"expected "
            f"{PROJECT_BC_WEIGHT_CALIBRATION_SPLIT!r}"
        )

    if (
        model
        not in
        PROJECT_BC_WEIGHT_CALIBRATION_MODELS
    ):
        raise CalibrationRunnerError(
            f"{model} is not in the frozen "
            "engineering-calibration model set"
        )

    mesh_file = Path(
        row[
            "mesh_file"
        ]
    )

    loop_file = Path(
        row[
            "loop_file"
        ]
    )

    if not mesh_file.is_file():
        raise FileNotFoundError(
            mesh_file
        )

    if not loop_file.is_file():
        raise FileNotFoundError(
            loop_file
        )

    return (
        row,
        mesh_file,
        loop_file,
    )


def batch_field_value(
    container,
    field: str,
    index: int,
):
    if isinstance(
        container,
        dict,
    ):
        value = container[
            field
        ]

    else:
        value = getattr(
            container,
            field,
        )

    array = np.asarray(
        value
    ).reshape(
        -1
    )

    return array[
        index
    ]


def evaluate_engineering_model(
    *,
    algorithm,
    executable: Path,
    manifest: Path,
    model: str,
    bc_weight: float,
    seed: int,
):
    (
        row,
        mesh_file,
        loop_file,
    ) = load_engineering_model_row(
        manifest=
            manifest,

        model=
            model,
    )

    executable = Path(
        executable
    )

    if not executable.is_file():
        raise FileNotFoundError(
            executable
        )

    env = (
        FinalRewardWrapper(
            FinalizationEvalWrapper(
                LoopyCutsEnv(
                    executable=
                        executable,

                    mesh_file=
                        mesh_file,

                    loop_file=
                        loop_file,

                    echo_logs=
                        False,
                )
            )
        )
    )

    eval_buffer = (
        ReplayBuffer(
            size=
                MAX_LOOPS
                +
                8,

            random_seed=
                int(
                    seed
                ),
        )
    )

    collector = (
        Collector(
            algorithm,
            env,
            eval_buffer,

            exploration_noise=
                False,
        )
    )

    try:
        collector.reset()

        collect_stats = (
            collector.collect(
                n_episode=
                    1
            )
        )

    finally:
        env.close()

    if len(
        eval_buffer
    ) <= 0:
        raise CalibrationRunnerError(
            "Engineering evaluation produced "
            "an empty replay buffer"
        )

    indices = (
        eval_buffer.sample_indices(
            0
        )
    )

    data = eval_buffer[
        indices
    ]

    rewards = np.asarray(
        data.rew,
        dtype=np.float64,
    ).reshape(
        -1
    )

    terminated = np.asarray(
        data.terminated,
        dtype=np.bool_,
    ).reshape(
        -1
    )

    truncated = np.asarray(
        data.truncated,
        dtype=np.bool_,
    ).reshape(
        -1
    )

    if bool(
        truncated.any()
    ):
        raise CalibrationRunnerError(
            "Engineering evaluation "
            "unexpectedly truncated"
        )

    terminal_indices = (
        np.flatnonzero(
            terminated
        )
    )

    if len(
        terminal_indices
    ) != 1:
        raise CalibrationRunnerError(
            "Engineering evaluation requires "
            "exactly one terminal transition; "
            f"got {len(terminal_indices)}"
        )

    terminal_index = int(
        terminal_indices[
            0
        ]
    )

    if not hasattr(
        data.info,
        "finalization_outcome",
    ):
        raise CalibrationRunnerError(
            "Terminal replay data is missing "
            "finalization_outcome"
        )

    final_record = (
        data.info
        .finalization_outcome
    )

    outcome = str(
        batch_field_value(
            final_record,
            "outcome",
            terminal_index,
        )
    )

    final_hex_raw = int(
        batch_field_value(
            final_record,
            "final_hex",
            terminal_index,
        )
    )

    final_total_raw = int(
        batch_field_value(
            final_record,
            "final_total_polys",
            terminal_index,
        )
    )

    if (
        outcome
        ==
        "FINALIZATION_CRASH"
    ):
        final_hex = None
        final_total = None

    else:
        final_hex = (
            final_hex_raw
        )

        final_total = (
            final_total_raw
        )

    episode_return = float(
        rewards.sum()
    )

    result = (
        CalibrationEpisodeResult(
            bc_weight=
                float(
                    bc_weight
                ),

            seed=
                int(
                    seed
                ),

            model=
                str(
                    model
                ),

            outcome=
                outcome,

            episode_return=
                episode_return,

            final_hex=
                final_hex,

            final_total_polys=
                final_total,
        )
    )

    validate_episode_result(
        result
    )

    return (
        result,
        {
            "episode_steps":
                int(
                    len(
                        eval_buffer
                    )
                ),

            "collect_stats":
                str(
                    collect_stats
                ),

            "mesh_file":
                str(
                    mesh_file
                    .resolve()
                ),

            "loop_file":
                str(
                    loop_file
                    .resolve()
                ),
        },
    )


def smoke_output_name(
    *,
    bc_weight: float,
    seed: int,
    model: str,
):
    weight_text = (
        f"{float(bc_weight):g}"
        .replace(
            ".",
            "p",
        )
    )

    return (
        f"smoke_lambda_{weight_text}_"
        f"seed_{int(seed)}_"
        f"{model}.json"
    )


def run_smoke(
    args,
):
    assert_clean_repository()
    assert_protocol()

    bc_weight = float(
        args.bc_weight
    )

    seed = int(
        args.seed
    )

    model = str(
        args.model
    )

    # A smoke test MUST NOT expose one of the frozen formal
    # candidate lambdas before the formal grid is launched.
    if any(
        math.isclose(
            bc_weight,
            candidate,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for candidate in
        PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES
    ):
        raise CalibrationRunnerError(
            "Smoke bc_weight must NOT be one of "
            "the frozen formal candidates"
        )

    if (
        model
        not in
        PROJECT_BC_WEIGHT_CALIBRATION_MODELS
    ):
        raise CalibrationRunnerError(
            "Smoke model must belong to the "
            "engineering_calibration split"
        )

    print("=" * 112)
    print("BC-WEIGHT CALIBRATION RUNNER V2 -- NON-CANDIDATE SMOKE")
    print("=" * 112)

    print(
        "runner version :",
        RUNNER_VERSION,
    )

    print(
        "git commit     :",
        git_head(),
    )

    print(
        "bc_weight      :",
        bc_weight,
    )

    print(
        "seed           :",
        seed,
    )

    print(
        "eval model     :",
        model,
    )

    print(
        "Stage-I updates:",
        PROJECT_BC_WEIGHT_CALIBRATION_STAGE1_GRADIENT_STEPS,
    )

    print(
        "device         :",
        PROJECT_BC_WEIGHT_CALIBRATION_DEVICE,
    )

    print()

    set_run_seed(
        seed
    )

    (
        demo_buffer,
        demo_records,
        demo_provenance,
    ) = load_main_demo_replay(
        raw_root=
            Path(
                args.raw_demo_root
            ),

        quality_manifest=
            Path(
                args.demo_quality
            ),

        random_seed=
            seed,
    )

    assert len(
        demo_buffer
    ) == PROJECT_MAIN_DEMO_TRANSITIONS

    assert len(
        demo_records
    ) == PROJECT_MAIN_DEMO_EPISODES

    loaded_models = tuple(
        record[
            "model"
        ]
        for record in
        demo_records
    )

    assert (
        loaded_models
        ==
        PROJECT_MAIN_DEMO_MODELS
    )

    print(
        "D_demo episodes    :",
        len(
            demo_records
        ),
    )

    print(
        "D_demo transitions :",
        len(
            demo_buffer
        ),
    )

    (
        algorithm,
        policy,
        auto_alpha,
    ) = build_stage1_algorithm(
        bc_weight=
            bc_weight,

        seed=
            seed,
    )

    assert (
        algorithm.bc_enabled
        is True
    )

    assert math.isclose(
        algorithm.bc_weight,
        bc_weight,
    )

    assert math.isclose(
        auto_alpha.value,
        PROJECT_INITIAL_ALPHA,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )

    print()
    print("=" * 112)
    print("STAGE-I OFFLINE TRAINING")
    print("=" * 112)

    stage1_record = (
        run_stage1(
            algorithm=
                algorithm,

            policy=
                policy,

            demo_buffer=
                demo_buffer,
        )
    )

    print()
    print("=" * 112)
    print("DETERMINISTIC ENGINEERING EVALUATION")
    print("=" * 112)

    (
        episode_result,
        evaluation_provenance,
    ) = evaluate_engineering_model(
        algorithm=
            algorithm,

        executable=
            Path(
                args.executable
            ),

        manifest=
            Path(
                args.manifest
            ),

        model=
            model,

        bc_weight=
            bc_weight,

        seed=
            seed,
    )

    print(
        "model          :",
        episode_result.model,
    )

    print(
        "outcome        :",
        episode_result.outcome,
    )

    print(
        "episode return :",
        episode_result.episode_return,
    )

    print(
        "final hex      :",
        episode_result.final_hex,
    )

    print(
        "final polys    :",
        episode_result.final_total_polys,
    )

    output_root = Path(
        args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_root
        /
        smoke_output_name(
            bc_weight=
                bc_weight,

            seed=
                seed,

            model=
                model,
        )
    )

    payload = {
        "runner_version":
            RUNNER_VERSION,

        "calibration_version":
            PROJECT_BC_WEIGHT_CALIBRATION_VERSION,

        "run_kind":
            "NON_CANDIDATE_FULL_BUDGET_SMOKE",

        "selector_eligible":
            False,

        "formal_grid_member":
            False,

        "git_commit":
            git_head(),

        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "torch_version":
            torch.__version__,

        "tianshou_version":
            tianshou.__version__,

        "torch_num_threads":
            torch.get_num_threads(),

        "bc_weight":
            bc_weight,

        "seed":
            seed,

        "device":
            PROJECT_BC_WEIGHT_CALIBRATION_DEVICE,

        "demo_provenance":
            demo_provenance,

        "stage1":
            stage1_record,

        "evaluation":
            {
                **asdict(
                    episode_result
                ),

                **evaluation_provenance,
            },
    }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        "smoke result:",
        output_path,
    )

    print()
    print("=" * 112)
    print(
        "PASS: non-candidate lambda completed "
        "the full 782-update Stage-I path"
    )
    print(
        "PASS: deterministic engineering evaluation completed"
    )
    print(
        "PASS: smoke result is explicitly ineligible "
        "for the formal 75-row calibration grid"
    )
    print("=" * 112)


def run_formal_pair(
    args,
):
    assert_clean_repository()
    assert_protocol()

    bc_weight = (
        canonical_formal_candidate(
            args.bc_weight
        )
    )

    seed = int(
        args.seed
    )

    if (
        seed
        not in
        PROJECT_BC_WEIGHT_CALIBRATION_SEEDS
    ):
        raise CalibrationRunnerError(
            "Formal seed must be one of the "
            "frozen calibration seeds: "
            f"{PROJECT_BC_WEIGHT_CALIBRATION_SEEDS}"
        )

    output_root = Path(
        args.formal_output_root
    )

    output_path = (
        output_root
        /
        formal_pair_output_name(
            bc_weight=
                bc_weight,

            seed=
                seed,
        )
    )

    if output_path.exists():
        raise CalibrationRunnerError(
            "Formal pair artifact already exists; "
            "refusing silent overwrite: "
            f"{output_path}"
        )

    commit = (
        git_head()
    )

    print("=" * 112)
    print("BC-WEIGHT CALIBRATION RUNNER V2 -- FORMAL PAIR")
    print("=" * 112)

    print(
        "git commit     :",
        commit,
    )

    print(
        "bc_weight      :",
        bc_weight,
    )

    print(
        "seed           :",
        seed,
    )

    print(
        "Stage-I updates:",
        PROJECT_BC_WEIGHT_CALIBRATION_STAGE1_GRADIENT_STEPS,
    )

    print(
        "eval models    :",
        PROJECT_BC_WEIGHT_CALIBRATION_MODELS,
    )

    print(
        "output         :",
        output_path,
    )

    print()


    # ============================================================
    # Frozen formal D_demo.
    # ============================================================

    set_run_seed(
        seed
    )

    (
        demo_buffer,
        demo_records,
        demo_provenance,
    ) = load_main_demo_replay(
        raw_root=
            Path(
                args.raw_demo_root
            ),

        quality_manifest=
            Path(
                args.demo_quality
            ),

        random_seed=
            seed,
    )

    if (
        len(
            demo_buffer
        )
        !=
        PROJECT_MAIN_DEMO_TRANSITIONS
    ):
        raise CalibrationRunnerError(
            "Formal D_demo transition-count mismatch"
        )

    if (
        len(
            demo_records
        )
        !=
        PROJECT_MAIN_DEMO_EPISODES
    ):
        raise CalibrationRunnerError(
            "Formal D_demo episode-count mismatch"
        )

    loaded_models = tuple(
        record[
            "model"
        ]
        for record in
        demo_records
    )

    if (
        loaded_models
        !=
        PROJECT_MAIN_DEMO_MODELS
    ):
        raise CalibrationRunnerError(
            "Formal D_demo model order/content mismatch"
        )


    # ============================================================
    # Exactly one Stage-I run for this (lambda, seed) pair.
    # ============================================================

    (
        algorithm,
        policy,
        auto_alpha,
    ) = build_stage1_algorithm(
        bc_weight=
            bc_weight,

        seed=
            seed,
    )

    if not math.isclose(
        auto_alpha.value,
        PROJECT_INITIAL_ALPHA,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    ):
        raise CalibrationRunnerError(
            "Initial alpha mismatch"
        )

    print("=" * 112)
    print("STAGE-I OFFLINE TRAINING")
    print("=" * 112)

    stage1_record = (
        run_stage1(
            algorithm=
                algorithm,

            policy=
                policy,

            demo_buffer=
                demo_buffer,
        )
    )

    stage1_record[
        "alpha_after_stage1"
    ] = float(
        auto_alpha.value
    )


    # ============================================================
    # Five deterministic engineering evaluations using the SAME
    # post-Stage-I algorithm.  No further learning occurs here.
    # ============================================================

    print()
    print("=" * 112)
    print("ENGINEERING-CALIBRATION EVALUATIONS")
    print("=" * 112)

    evaluation_payloads = []
    selector_rows = []

    for model in (
        PROJECT_BC_WEIGHT_CALIBRATION_MODELS
    ):
        print()
        print(
            f"[EVAL] {model}"
        )

        (
            episode_result,
            evaluation_provenance,
        ) = evaluate_engineering_model(
            algorithm=
                algorithm,

            executable=
                Path(
                    args.executable
                ),

            manifest=
                Path(
                    args.manifest
                ),

            model=
                model,

            bc_weight=
                bc_weight,

            seed=
                seed,
        )

        selector_rows.append(
            episode_result
        )

        evaluation_payloads.append(
            {
                **asdict(
                    episode_result
                ),

                **evaluation_provenance,
            }
        )

        print(
            "outcome       :",
            episode_result.outcome,
        )

        print(
            "episode return:",
            episode_result.episode_return,
        )

        print(
            "hex / total   :",
            episode_result.final_hex,
            "/",
            episode_result.final_total_polys,
        )


    observed_models = tuple(
        row.model
        for row in
        selector_rows
    )

    if (
        observed_models
        !=
        PROJECT_BC_WEIGHT_CALIBRATION_MODELS
    ):
        raise CalibrationRunnerError(
            "Formal pair evaluation model order mismatch"
        )

    if (
        len(
            selector_rows
        )
        !=
        len(
            PROJECT_BC_WEIGHT_CALIBRATION_MODELS
        )
    ):
        raise CalibrationRunnerError(
            "Formal pair must contain exactly "
            "five engineering evaluations"
        )

    for row in selector_rows:
        validate_episode_result(
            row
        )

        if not math.isclose(
            row.bc_weight,
            bc_weight,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise CalibrationRunnerError(
                "Evaluation bc_weight mismatch"
            )

        if row.seed != seed:
            raise CalibrationRunnerError(
                "Evaluation seed mismatch"
            )


    # ============================================================
    # Write only AFTER all five evaluations have completed.
    # This prevents a partial pair from entering the formal grid.
    # ============================================================

    payload = {
        "runner_version":
            RUNNER_VERSION,

        "calibration_version":
            PROJECT_BC_WEIGHT_CALIBRATION_VERSION,

        "result_schema_version":
            CALIBRATION_RESULT_SCHEMA_VERSION,

        "run_kind":
            "FORMAL_BC_WEIGHT_CALIBRATION_PAIR",

        "selector_eligible":
            True,

        "formal_grid_member":
            True,

        "git_commit":
            commit,

        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "torch_version":
            torch.__version__,

        "tianshou_version":
            tianshou.__version__,

        "torch_num_threads":
            torch.get_num_threads(),

        "bc_weight":
            bc_weight,

        "seed":
            seed,

        "device":
            PROJECT_BC_WEIGHT_CALIBRATION_DEVICE,

        "demo_provenance":
            demo_provenance,

        "stage1":
            stage1_record,

        "evaluations":
            evaluation_payloads,
    }

    atomic_write_json(
        path=
            output_path,

        payload=
            payload,
    )

    print()
    print(
        "formal pair artifact:",
        output_path,
    )

    print(
        "sha256:",
        sha256_file(
            output_path
        ),
    )

    print()
    print("=" * 112)
    print(
        "PASS: formal pair completed one 782-update "
        "Stage-I run and all five engineering evaluations"
    )
    print(
        "PASS: artifact is selector-eligible and "
        "was written only after the pair completed"
    )
    print("=" * 112)


def _episode_result_from_payload(
    data,
):
    final_hex = data.get(
        "final_hex"
    )

    final_total = data.get(
        "final_total_polys"
    )

    if final_hex is not None:
        final_hex = int(
            final_hex
        )

    if final_total is not None:
        final_total = int(
            final_total
        )

    row = (
        CalibrationEpisodeResult(
            bc_weight=
                float(
                    data[
                        "bc_weight"
                    ]
                ),

            seed=
                int(
                    data[
                        "seed"
                    ]
                ),

            model=
                str(
                    data[
                        "model"
                    ]
                ),

            outcome=
                str(
                    data[
                        "outcome"
                    ]
                ),

            episode_return=
                float(
                    data[
                        "episode_return"
                    ]
                ),

            final_hex=
                final_hex,

            final_total_polys=
                final_total,
        )
    )

    validate_episode_result(
        row
    )

    return row


def load_formal_grid_from_pair_artifacts(
    *,
    output_root: Path,
    expected_git_commit: str | None = None,
):
    output_root = Path(
        output_root
    )

    if not output_root.is_dir():
        raise CalibrationRunnerError(
            "Formal pair directory does not exist: "
            f"{output_root}"
        )

    if expected_git_commit is None:
        expected_git_commit = (
            git_head()
        )

    expected_paths = []

    for bc_weight in (
        PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES
    ):
        for seed in (
            PROJECT_BC_WEIGHT_CALIBRATION_SEEDS
        ):
            expected_paths.append(
                output_root
                /
                formal_pair_output_name(
                    bc_weight=
                        bc_weight,

                    seed=
                        seed,
                )
            )

    expected_names = {
        path.name
        for path in
        expected_paths
    }

    actual_paths = tuple(
        sorted(
            output_root.glob(
                "formal_lambda_*_seed_*.json"
            )
        )
    )

    actual_names = {
        path.name
        for path in
        actual_paths
    }

    missing_names = (
        expected_names
        -
        actual_names
    )

    extra_names = (
        actual_names
        -
        expected_names
    )

    if missing_names:
        raise CalibrationRunnerError(
            "Formal calibration grid is incomplete; "
            "missing pair artifacts: "
            +
            ", ".join(
                sorted(
                    missing_names
                )
            )
        )

    if extra_names:
        raise CalibrationRunnerError(
            "Unexpected formal pair artifacts: "
            +
            ", ".join(
                sorted(
                    extra_names
                )
            )
        )

    if len(
        actual_paths
    ) != 15:
        raise CalibrationRunnerError(
            "Formal calibration requires exactly "
            f"15 pair artifacts; got {len(actual_paths)}"
        )

    rows = []
    artifact_records = []

    for expected_path in (
        expected_paths
    ):
        payload = json.loads(
            expected_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            payload.get(
                "runner_version"
            )
            !=
            RUNNER_VERSION
        ):
            raise CalibrationRunnerError(
                f"{expected_path}: runner version mismatch"
            )

        if (
            payload.get(
                "calibration_version"
            )
            !=
            PROJECT_BC_WEIGHT_CALIBRATION_VERSION
        ):
            raise CalibrationRunnerError(
                f"{expected_path}: calibration version mismatch"
            )

        if (
            payload.get(
                "result_schema_version"
            )
            !=
            CALIBRATION_RESULT_SCHEMA_VERSION
        ):
            raise CalibrationRunnerError(
                f"{expected_path}: result schema mismatch"
            )

        if (
            payload.get(
                "run_kind"
            )
            !=
            "FORMAL_BC_WEIGHT_CALIBRATION_PAIR"
        ):
            raise CalibrationRunnerError(
                f"{expected_path}: wrong run kind"
            )

        if (
            payload.get(
                "selector_eligible"
            )
            is not True
        ):
            raise CalibrationRunnerError(
                f"{expected_path}: not selector eligible"
            )

        if (
            payload.get(
                "formal_grid_member"
            )
            is not True
        ):
            raise CalibrationRunnerError(
                f"{expected_path}: not a formal grid member"
            )

        if (
            payload.get(
                "git_commit"
            )
            !=
            expected_git_commit
        ):
            raise CalibrationRunnerError(
                f"{expected_path}: Git commit mismatch; "
                "all formal pairs must come from exactly "
                "the same committed runner/protocol state"
            )

        bc_weight = (
            canonical_formal_candidate(
                payload[
                    "bc_weight"
                ]
            )
        )

        seed = int(
            payload[
                "seed"
            ]
        )

        expected_name = (
            formal_pair_output_name(
                bc_weight=
                    bc_weight,

                seed=
                    seed,
            )
        )

        if (
            expected_name
            !=
            expected_path.name
        ):
            raise CalibrationRunnerError(
                f"{expected_path}: filename/payload mismatch"
            )

        if (
            seed
            not in
            PROJECT_BC_WEIGHT_CALIBRATION_SEEDS
        ):
            raise CalibrationRunnerError(
                f"{expected_path}: invalid formal seed"
            )


        demo = payload[
            "demo_provenance"
        ]

        if int(
            demo[
                "episodes"
            ]
        ) != PROJECT_MAIN_DEMO_EPISODES:
            raise CalibrationRunnerError(
                f"{expected_path}: D_demo episode mismatch"
            )

        if int(
            demo[
                "transitions"
            ]
        ) != PROJECT_MAIN_DEMO_TRANSITIONS:
            raise CalibrationRunnerError(
                f"{expected_path}: D_demo transition mismatch"
            )

        if int(
            demo[
                "random_seed"
            ]
        ) != seed:
            raise CalibrationRunnerError(
                f"{expected_path}: replay seed mismatch"
            )


        stage1 = payload[
            "stage1"
        ]

        if int(
            stage1[
                "gradient_updates"
            ]
        ) != PROJECT_BC_WEIGHT_CALIBRATION_STAGE1_GRADIENT_STEPS:
            raise CalibrationRunnerError(
                f"{expected_path}: Stage-I update mismatch"
            )

        if int(
            stage1[
                "sampled_demo_transitions"
            ]
        ) != (
            PROJECT_BC_WEIGHT_CALIBRATION_STAGE1_GRADIENT_STEPS
            *
            PAPER_BATCH_SIZE
        ):
            raise CalibrationRunnerError(
                f"{expected_path}: Stage-I exposure mismatch"
            )


        evaluations = payload.get(
            "evaluations"
        )

        if not isinstance(
            evaluations,
            list,
        ):
            raise CalibrationRunnerError(
                f"{expected_path}: evaluations must be a list"
            )

        if (
            len(
                evaluations
            )
            !=
            len(
                PROJECT_BC_WEIGHT_CALIBRATION_MODELS
            )
        ):
            raise CalibrationRunnerError(
                f"{expected_path}: expected exactly "
                "five engineering evaluations"
            )

        pair_rows = [
            _episode_result_from_payload(
                evaluation
            )
            for evaluation in
            evaluations
        ]

        if tuple(
            row.model
            for row in
            pair_rows
        ) != PROJECT_BC_WEIGHT_CALIBRATION_MODELS:
            raise CalibrationRunnerError(
                f"{expected_path}: evaluation model set/order mismatch"
            )

        for row in pair_rows:
            if not math.isclose(
                row.bc_weight,
                bc_weight,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise CalibrationRunnerError(
                    f"{expected_path}: row lambda mismatch"
                )

            if row.seed != seed:
                raise CalibrationRunnerError(
                    f"{expected_path}: row seed mismatch"
                )

        rows.extend(
            pair_rows
        )

        artifact_records.append(
            {
                "path":
                    str(
                        expected_path
                        .resolve()
                    ),

                "sha256":
                    sha256_file(
                        expected_path
                    ),

                "bc_weight":
                    bc_weight,

                "seed":
                    seed,
            }
        )

    rows = (
        validate_complete_result_grid(
            rows
        )
    )

    if len(
        rows
    ) != 75:
        raise CalibrationRunnerError(
            "Validated formal grid must contain "
            f"75 rows; got {len(rows)}"
        )

    return (
        rows,
        tuple(
            artifact_records
        ),
    )


def _summary_json(
    summary,
):
    data = asdict(
        summary
    )

    metric = float(
        data[
            "aggregate_nonhex_fraction"
        ]
    )

    if not math.isfinite(
        metric
    ):
        data[
            "aggregate_nonhex_fraction"
        ] = None

    return data


def run_select(
    args,
):
    assert_clean_repository()
    assert_protocol()

    selection_output = Path(
        args.selection_output
    )

    if selection_output.exists():
        raise CalibrationRunnerError(
            "Selection artifact already exists; "
            "refusing silent overwrite: "
            f"{selection_output}"
        )

    current_commit = (
        git_head()
    )

    (
        rows,
        artifact_records,
    ) = load_formal_grid_from_pair_artifacts(
        output_root=
            Path(
                args.formal_output_root
            ),

        expected_git_commit=
            current_commit,
    )

    (
        winner,
        summaries,
    ) = select_best_bc_weight(
        rows
    )

    payload = {
        "runner_version":
            RUNNER_VERSION,

        "calibration_version":
            PROJECT_BC_WEIGHT_CALIBRATION_VERSION,

        "result_schema_version":
            CALIBRATION_RESULT_SCHEMA_VERSION,

        "run_kind":
            "FORMAL_BC_WEIGHT_CALIBRATION_SELECTION",

        "git_commit":
            current_commit,

        "pair_artifact_count":
            len(
                artifact_records
            ),

        "episode_result_count":
            len(
                rows
            ),

        "pair_artifacts":
            list(
                artifact_records
            ),

        "winner":
            _summary_json(
                winner
            ),

        "candidate_summaries":
            [
                _summary_json(
                    summary
                )
                for summary in
                summaries
            ],
    }

    atomic_write_json(
        path=
            selection_output,

        payload=
            payload,
    )

    print("=" * 112)
    print("FORMAL BC-WEIGHT CALIBRATION SELECTION")
    print("=" * 112)

    print(
        "Git commit:",
        current_commit,
    )

    print(
        "pair artifacts:",
        len(
            artifact_records
        ),
    )

    print(
        "episode rows:",
        len(
            rows
        ),
    )

    print()

    for summary in summaries:
        print(
            "lambda="
            f"{summary.bc_weight:<4g}"
            " full_hex="
            f"{summary.full_hex_count:2d}"
            " crashes="
            f"{summary.finalization_crash_count:2d}"
            " nonhex="
            f"{summary.aggregate_nonhex_fraction}"
            " mean_return="
            f"{summary.mean_episode_return}"
        )

    print()
    print(
        "WINNER lambda_BC:",
        winner.bc_weight,
    )

    print(
        "selection artifact:",
        selection_output,
    )

    print(
        "selection sha256:",
        sha256_file(
            selection_output
        ),
    )

    print()
    print(
        "PASS: complete frozen 75-row grid "
        "was selected using the committed selector"
    )


def print_plan():
    assert_protocol()

    print("=" * 112)
    print("BC-WEIGHT CALIBRATION RUNNER V2 PLAN")
    print("=" * 112)

    print(
        "formal candidates:",
        PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES,
    )

    print(
        "formal seeds:",
        PROJECT_BC_WEIGHT_CALIBRATION_SEEDS,
    )

    print(
        "engineering models:",
        PROJECT_BC_WEIGHT_CALIBRATION_MODELS,
    )

    print(
        "Stage-I updates/run:",
        PROJECT_BC_WEIGHT_CALIBRATION_STAGE1_GRADIENT_STEPS,
    )

    print(
        "formal result count:",
        (
            len(
                PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES
            )
            *
            len(
                PROJECT_BC_WEIGHT_CALIBRATION_SEEDS
            )
            *
            len(
                PROJECT_BC_WEIGHT_CALIBRATION_MODELS
            )
        ),
    )

    print(
        "formal training ready:",
        formal_training_ready(),
    )

    print()
    print(
        "PASS: runner plan matches frozen protocol"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=(
            "plan",
            "smoke",
            "formal-pair",
            "select",
        ),
        required=True,
    )

    parser.add_argument(
        "--bc-weight",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--model",
        default="bimba",
    )

    parser.add_argument(
        "--executable",
        type=Path,
        default=
            DEFAULT_EXECUTABLE,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=
            DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--raw-demo-root",
        type=Path,
        default=
            DEFAULT_RAW_DEMO_ROOT,
    )

    parser.add_argument(
        "--demo-quality",
        type=Path,
        default=
            DEFAULT_DEMO_QUALITY,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=
            DEFAULT_SMOKE_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--formal-output-root",
        type=Path,
        default=
            DEFAULT_FORMAL_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--selection-output",
        type=Path,
        default=
            DEFAULT_SELECTION_OUTPUT,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "plan":
        print_plan()
        return

    if args.mode == "smoke":
        run_smoke(
            args
        )
        return

    if args.mode == "formal-pair":
        run_formal_pair(
            args
        )
        return

    if args.mode == "select":
        run_select(
            args
        )
        return

    raise AssertionError(
        args.mode
    )


if __name__ == "__main__":
    main()
