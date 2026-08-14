from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


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


from envs.final_reward_wrapper import (
    FinalRewardWrapper,
)

from envs.finalization_eval_wrapper import (
    FinalizationEvalWrapper,
)

from envs.loopycuts_env import (
    LoopyCutsEnv,
)

from evaluation.baseline_audit import (
    EXECUTABLE,
    load_manifest,
    select_models,
    validate_initial_action_space,
    validate_model_files,
    verify_frozen_manifest,
)

from imitation.demo_v1 import (
    copy_observation,
    save_episode,
)

from policies.simple import (
    OriginalOrderPolicy,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "demonstrations_v1/raw"
)


def current_git_commit():
    return (
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=
                PROJECT_ROOT,
            text=True,
        )
        .strip()
    )


def make_environment(
    *,
    mesh_file,
    loop_file,
    echo_logs,
):
    return FinalRewardWrapper(
        FinalizationEvalWrapper(
            LoopyCutsEnv(
                executable=
                    EXECUTABLE,

                mesh_file=
                    mesh_file,

                loop_file=
                    loop_file,

                echo_logs=
                    echo_logs,
            )
        )
    )


def collect_model(
    *,
    row,
    output_root,
    overwrite,
    echo_logs,
    max_steps,
):
    model = str(
        row[
            "model"
        ]
    )

    mesh_file, loop_file = (
        validate_model_files(
            row
        )
    )

    model_dir = (
        Path(
            output_root
        )
        /
        "train"
        /
        model
    )

    stem = (
        f"{model}_"
        "original_demo_v1"
    )

    output_paths = [
        model_dir
        /
        f"{stem}.npz",

        model_dir
        /
        f"{stem}.json",

        model_dir
        /
        f"{stem}.jsonl",
    ]

    existing = [
        path
        for path
        in output_paths
        if path.exists()
    ]

    if (
        existing
        and
        not overwrite
    ):
        raise FileExistsError(
            f"{model}: demonstration "
            "output already exists. "
            "Use --overwrite only "
            "for intentional regeneration."
        )

    if overwrite:
        for path in output_paths:
            path.unlink(
                missing_ok=True
            )

    env = make_environment(
        mesh_file=
            mesh_file,

        loop_file=
            loop_file,

        echo_logs=
            echo_logs,
    )

    teacher = (
        OriginalOrderPolicy()
    )

    teacher.reset()

    observations = []
    actions = []
    rewards = []
    terminated_flags = []
    truncated_flags = []
    audit_records = []

    terminal_info = None

    try:
        (
            observation,
            _,
        ) = env.reset()

        initial_actions = (
            validate_initial_action_space(
                model=
                    model,

                row=
                    row,

                loop_file=
                    loop_file,

                client=
                    env.unwrapped.client,
            )
        )

        initial_actionable = len(
            initial_actions
        )

        observations.append(
            copy_observation(
                observation
            )
        )

        initial_mask_count = int(
            np.count_nonzero(
                observation[
                    "mask"
                ]
            )
        )

        if (
            initial_mask_count
            !=
            initial_actionable
        ):
            raise RuntimeError(
                f"{model}: Observation V1 "
                "initial mask count does "
                "not match authoritative "
                "C++ initial ACTIONS. "
                f"mask={initial_mask_count}, "
                "actions="
                f"{initial_actionable}"
            )

        step_index = 0

        while True:
            if (
                step_index
                >=
                max_steps
            ):
                raise RuntimeError(
                    f"{model}: exceeded "
                    f"max_steps={max_steps}. "
                    "Demo V1 refuses to "
                    "write truncated episodes."
                )

            state = (
                env.unwrapped
                .current_state
            )

            legal_actions = tuple(
                env.unwrapped
                .legal_actions
            )

            if state is None:
                raise RuntimeError(
                    f"{model}: missing "
                    "current C++ state"
                )

            if not legal_actions:
                raise RuntimeError(
                    f"{model}: collector "
                    "reached a non-stepped "
                    "state with no legal "
                    "actions"
                )

            action = (
                teacher.select(
                    state,
                    legal_actions,
                )
            )

            current_observation = (
                observations[
                    -1
                ]
            )

            if not bool(
                current_observation[
                    "mask"
                ][
                    action
                ]
            ):
                raise RuntimeError(
                    f"{model}: Original "
                    f"teacher action {action} "
                    "is illegal under stored "
                    "Observation V1 mask"
                )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            if truncated:
                raise RuntimeError(
                    f"{model}: Demo V1 "
                    "does not permit "
                    "truncation"
                )

            actions.append(
                int(
                    action
                )
            )

            rewards.append(
                float(
                    reward
                )
            )

            terminated_flags.append(
                bool(
                    terminated
                )
            )

            truncated_flags.append(
                bool(
                    truncated
                )
            )

            observations.append(
                copy_observation(
                    next_observation
                )
            )

            step_result = dict(
                info[
                    "step_result"
                ]
            )

            transition_metrics = dict(
                info[
                    "transition_metrics"
                ]
            )

            reward_breakdown = dict(
                info[
                    "reward_v2_breakdown"
                ]
            )

            audit_records.append(
                {
                    "step":
                        int(
                            step_index
                        ),

                    "action":
                        int(
                            action
                        ),

                    "status":
                        str(
                            step_result[
                                "status"
                            ]
                        ),

                    "reward":
                        float(
                            reward
                        ),

                    "terminated":
                        bool(
                            terminated
                        ),

                    "transition_metrics":
                        transition_metrics,

                    "reward_v2_breakdown":
                        reward_breakdown,
                }
            )

            next_legal_count = int(
                np.count_nonzero(
                    next_observation[
                        "mask"
                    ]
                )
            )

            print(
                f"{model}: "
                f"step={step_index + 1} "
                f"action={action} "
                "status="
                f"{step_result['status']} "
                f"reward={float(reward):.6f} "
                "legal_next="
                f"{next_legal_count}",
                flush=True,
            )

            step_index += 1

            if terminated:
                terminal_info = dict(
                    info
                )

                break

        if terminal_info is None:
            raise RuntimeError(
                f"{model}: terminal "
                "transition was not "
                "captured"
            )

        finalization_outcome = dict(
            terminal_info[
                "finalization_outcome"
            ]
        )

        if not bool(
            finalization_outcome[
                "attempted"
            ]
        ):
            raise RuntimeError(
                f"{model}: terminal "
                "demonstration did not "
                "attempt FINALIZE_EVAL"
            )

        outcome = str(
            finalization_outcome[
                "outcome"
            ]
        )

        allowed_outcomes = {
            "FULL_HEX",
            "NON_FULL_HEX",
            "FINALIZATION_CRASH",
        }

        if (
            outcome
            not in
            allowed_outcomes
        ):
            raise RuntimeError(
                f"{model}: unknown "
                "finalization outcome "
                f"{outcome!r}"
            )

        result = save_episode(
            output_dir=
                model_dir,

            model=
                model,

            split=
                "train",

            mesh_file=
                mesh_file,

            loop_file=
                loop_file,

            source_git_commit=
                current_git_commit(),

            observations=
                observations,

            actions=
                actions,

            rewards=
                rewards,

            terminated=
                terminated_flags,

            truncated=
                truncated_flags,

            audit_records=
                audit_records,

            finalization_outcome=
                finalization_outcome,

            initial_actionable=
                initial_actionable,
        )

    finally:
        env.close()

    print()
    print(
        "="
        *
        88
    )

    print(
        "DEMONSTRATION V1 COMPLETE"
    )

    print(
        "="
        *
        88
    )

    print(
        "model:",
        model,
    )

    print(
        "steps:",
        result[
            "record"
        ][
            "num_steps"
        ],
    )

    print(
        "return:",
        result[
            "record"
        ][
            "total_return"
        ],
    )

    print(
        "outcome:",
        result[
            "record"
        ][
            "finalization_outcome"
        ][
            "outcome"
        ],
    )

    print(
        "npz:",
        result[
            "npz"
        ],
    )

    print(
        "metadata:",
        result[
            "metadata"
        ],
    )

    print(
        "audit:",
        result[
            "audit"
        ],
    )

    return result


def main():
    parser = (
        argparse.ArgumentParser(
            description=(
                "Collect frozen "
                "Demonstration V1 episodes "
                "using the LoopyCuts "
                "Original runtime policy."
            )
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=
            DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--echo-logs",
        action="store_true",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
    )

    args = (
        parser.parse_args()
    )

    if (
        args.max_steps
        <=
        0
    ):
        parser.error(
            "--max-steps must "
            "be positive"
        )

    # ------------------------------------------------------------
    # Respect the frozen Dataset Split V2 protocol.
    # ------------------------------------------------------------

    verify_frozen_manifest()

    manifest = (
        load_manifest()
    )

    selected = (
        select_models(
            manifest,
            split=
                "train",
            requested_models=
                args.models,
            allow_held_out_test=
                False,
        )
    )

    print(
        "="
        *
        88
    )

    print(
        "LOOPYCUTS ORIGINAL "
        "DEMONSTRATION V1"
    )

    print(
        "="
        *
        88
    )

    print(
        "teacher:",
        "OriginalOrderPolicy",
    )

    print(
        "models:",
        ", ".join(
            row[
                "model"
            ]
            for row
            in selected
        ),
    )

    print(
        "output:",
        args.output_root,
    )

    print(
        "="
        *
        88
    )

    for row in selected:
        collect_model(
            row=
                row,

            output_root=
                args.output_root,

            overwrite=
                args.overwrite,

            echo_logs=
                args.echo_logs,

            max_steps=
                args.max_steps,
        )


if __name__ == "__main__":
    main()
