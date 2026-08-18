from __future__ import annotations

import argparse
import csv
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

from tianshou.algorithm.optim import (
    AdamOptimizerFactory,
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
    CalibrationEpisodeResult,
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

    PROJECT_ADAM_BETAS,
    PROJECT_ADAM_EPS,
    PROJECT_ADAM_WEIGHT_DECAY,

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
    "bc_weight_calibration_runner_v1"
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
    "bc_weight_calibration_smoke_v1"
)


class CalibrationRunnerError(
    RuntimeError
):
    pass


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
        "bc_weight_calibration_v1"
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


def make_adam_factory(
    *,
    lr: float,
):
    return AdamOptimizerFactory(
        lr=
            float(
                lr
            ),

        betas=
            PROJECT_ADAM_BETAS,

        eps=
            PROJECT_ADAM_EPS,

        weight_decay=
            PROJECT_ADAM_WEIGHT_DECAY,
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
                make_adam_factory(
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
                make_adam_factory(
                    lr=
                        PROJECT_ACTOR_LEARNING_RATE
                ),

            critic=
                critic1,

            critic_optim=
                make_adam_factory(
                    lr=
                        PROJECT_CRITIC1_LEARNING_RATE
                ),

            critic2=
                critic2,

            critic2_optim=
                make_adam_factory(
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
    print("BC-WEIGHT CALIBRATION RUNNER V1 -- NON-CANDIDATE SMOKE")
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


def print_plan():
    assert_protocol()

    print("=" * 112)
    print("BC-WEIGHT CALIBRATION RUNNER V1 PLAN")
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

    raise AssertionError(
        args.mode
    )


if __name__ == "__main__":
    main()
