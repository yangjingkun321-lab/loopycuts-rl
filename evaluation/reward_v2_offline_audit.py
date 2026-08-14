from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import fields
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    "/home/yjk/codes/loopycuts_rl"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from rewards.reward_v2 import (
    RewardV2Weights,
    compute_reward_v2,
)

from rewards.transition_metrics import (
    TransitionMetrics,
)


# ============================================================
# Candidate terminal scales.
#
# Dense terms are identical across all profiles.
# Only the real finalization outcome changes.
# ============================================================

PROFILES = {
    "conservative": RewardV2Weights(
        final_full_hex=2.0,
        final_non_full_hex=2.0,
        final_crash=3.0,
    ),

    "balanced": RewardV2Weights(
        final_full_hex=3.0,
        final_non_full_hex=3.0,
        final_crash=4.0,
    ),

    "strong": RewardV2Weights(
        final_full_hex=4.0,
        final_non_full_hex=4.0,
        final_crash=5.0,
    ),
}


GAMMAS = (
    0.99,
    0.995,
    0.997,
)


INT_FIELDS = {
    "step",
    "loop_id",
    "committed",
    "reverted",
    "convergence_delta",
    "first_convergence",
    "phase_closed_this_step",
    "terminal",
    "selection_success",
    "terminal_failure",
    "diagnostics_delta_valid",
    "available_before",
    "available_after",
    "available_drop",
}


STRING_FIELDS = {
    "status",
}


def row_to_metrics(
    row,
) -> TransitionMetrics:
    kwargs = {}

    for field in fields(
        TransitionMetrics
    ):
        name = field.name

        if name not in row.index:
            raise RuntimeError(
                f"Transition CSV is missing "
                f"TransitionMetrics field {name!r}"
            )

        value = row[
            name
        ]

        if name in INT_FIELDS:
            kwargs[
                name
            ] = int(
                value
            )

        elif name in STRING_FIELDS:
            kwargs[
                name
            ] = str(
                value
            )

        else:
            kwargs[
                name
            ] = float(
                value
            )

    return TransitionMetrics(
        **kwargs
    )


def discounted_return(
    rewards,
    gamma,
):
    total = 0.0

    for index, reward in enumerate(
        rewards
    ):
        total += (
            gamma ** index
        ) * float(
            reward
        )

    return float(
        total
    )


def require_unique_cases(
    df,
    *,
    name,
):
    duplicates = (
        df[
            "case"
        ]
        .duplicated(
            keep=False
        )
    )

    if bool(
        duplicates.any()
    ):
        bad = (
            df.loc[
                duplicates,
                "case",
            ]
            .tolist()
        )

        raise RuntimeError(
            f"{name} has duplicate cases: "
            f"{bad}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reward-audit-dir",
        type=Path,
        default=Path(
            "/home/yjk/loopycuts_test/"
            "reward_scale_audit_all"
        ),
    )

    parser.add_argument(
        "--finalization-csv",
        type=Path,
        default=Path(
            "/home/yjk/loopycuts_test/"
            "finalization_outcome_audit_v1/"
            "finalization_outcomes.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/yjk/loopycuts_test/"
            "reward_v2_offline_audit"
        ),
    )

    args = parser.parse_args()

    transition_path = (
        args.reward_audit_dir
        /
        "transition_metrics.csv"
    )

    episode_path = (
        args.reward_audit_dir
        /
        "episode_summary.csv"
    )

    reward_v1_path = (
        args.reward_audit_dir
        /
        "reward_v1_episode_summary.csv"
    )

    for path in (
        transition_path,
        episode_path,
        args.finalization_csv,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    transitions = pd.read_csv(
        transition_path
    )

    episodes = pd.read_csv(
        episode_path
    )

    finals = pd.read_csv(
        args.finalization_csv
    )

    require_unique_cases(
        episodes,
        name="episode_summary",
    )

    require_unique_cases(
        finals,
        name="finalization_outcomes",
    )

    # ============================================================
    # The selection trajectories and finalization audit must describe
    # exactly the same engineering/calibration cases.
    # ============================================================

    transition_cases = set(
        transitions[
            "case"
        ].unique()
    )

    episode_cases = set(
        episodes[
            "case"
        ].unique()
    )

    final_cases = set(
        finals[
            "case"
        ].unique()
    )

    if (
        transition_cases
        !=
        episode_cases
    ):
        raise RuntimeError(
            "transition_metrics and episode_summary "
            "case sets do not match"
        )

    if (
        transition_cases
        !=
        final_cases
    ):
        raise RuntimeError(
            "transition metrics and finalization outcome "
            "case sets do not match"
        )

    initial_actionable_by_case = {
        str(
            row[
                "case"
            ]
        ):
        int(
            row[
                "initial_actionable"
            ]
        )
        for _, row in episodes.iterrows()
    }

    outcome_by_case = {
        str(
            row[
                "case"
            ]
        ):
        str(
            row[
                "outcome"
            ]
        )
        for _, row in finals.iterrows()
    }

    final_steps_by_case = {
        str(
            row[
                "case"
            ]
        ):
        int(
            row[
                "num_steps"
            ]
        )
        for _, row in finals.iterrows()
    }

    # ============================================================
    # Structural trajectory checks
    # ============================================================

    for case in sorted(
        transition_cases
    ):
        case_df = (
            transitions[
                transitions[
                    "case"
                ]
                ==
                case
            ]
            .sort_values(
                "step"
            )
        )

        expected_steps = int(
            episodes.loc[
                episodes[
                    "case"
                ]
                ==
                case,
                "steps",
            ].iloc[
                0
            ]
        )

        if (
            len(
                case_df
            )
            !=
            expected_steps
        ):
            raise RuntimeError(
                f"{case}: transition count "
                f"{len(case_df)} != "
                f"episode steps {expected_steps}"
            )

        if (
            expected_steps
            !=
            final_steps_by_case[
                case
            ]
        ):
            raise RuntimeError(
                f"{case}: selection step count does not "
                "match finalization audit"
            )

        terminal_rows = (
            case_df[
                case_df[
                    "terminal"
                ]
                ==
                1
            ]
        )

        if (
            len(
                terminal_rows
            )
            !=
            1
        ):
            raise RuntimeError(
                f"{case}: expected exactly one "
                "terminal transition"
            )

        terminal_step = int(
            terminal_rows.iloc[
                0
            ][
                "step"
            ]
        )

        if (
            terminal_step
            !=
            expected_steps
        ):
            raise RuntimeError(
                f"{case}: terminal transition is not "
                "the final selection step"
            )

    # ============================================================
    # Score every transition under every candidate profile.
    # ============================================================

    transition_rows = []
    episode_rows = []

    for (
        profile_name,
        weights,
    ) in PROFILES.items():

        for case in sorted(
            transition_cases
        ):
            case_df = (
                transitions[
                    transitions[
                        "case"
                    ]
                    ==
                    case
                ]
                .sort_values(
                    "step"
                )
            )

            initial_actionable = (
                initial_actionable_by_case[
                    case
                ]
            )

            final_outcome = (
                outcome_by_case[
                    case
                ]
            )

            rewards = []

            sum_step = 0.0
            sum_tet = 0.0
            sum_revert = 0.0
            sum_convergence = 0.0
            sum_finalization = 0.0

            for _, row in (
                case_df.iterrows()
            ):
                metrics = (
                    row_to_metrics(
                        row
                    )
                )

                outcome = (
                    final_outcome
                    if int(
                        metrics.terminal
                    )
                    else None
                )

                breakdown = (
                    compute_reward_v2(
                        metrics=metrics,
                        initial_actionable_count=(
                            initial_actionable
                        ),
                        finalization_outcome=(
                            outcome
                        ),
                        weights=weights,
                    )
                )

                rewards.append(
                    breakdown.total
                )

                sum_step += (
                    breakdown.step
                )

                sum_tet += (
                    breakdown.tet_growth
                )

                sum_revert += (
                    breakdown.revert
                )

                sum_convergence += (
                    breakdown.convergence
                )

                sum_finalization += (
                    breakdown.finalization
                )

                transition_rows.append(
                    {
                        "profile":
                            profile_name,

                        "case":
                            case,

                        "step":
                            metrics.step,

                        "loop_id":
                            metrics.loop_id,

                        "status":
                            metrics.status,

                        "terminal":
                            metrics.terminal,

                        "selection_success":
                            metrics.selection_success,

                        "finalization_outcome":
                            (
                                outcome
                                if outcome is not None
                                else ""
                            ),

                        "r_step":
                            breakdown.step,

                        "r_tet":
                            breakdown.tet_growth,

                        "r_revert":
                            breakdown.revert,

                        "r_convergence":
                            breakdown.convergence,

                        "r_finalization":
                            breakdown.finalization,

                        "reward_v2":
                            breakdown.total,
                    }
                )

            undiscounted = float(
                sum(
                    rewards
                )
            )

            episode_row = {
                "profile":
                    profile_name,

                "case":
                    case,

                "finalization_outcome":
                    final_outcome,

                "steps":
                    len(
                        rewards
                    ),

                "initial_actionable":
                    initial_actionable,

                "r_step":
                    sum_step,

                "r_tet":
                    sum_tet,

                "r_revert":
                    sum_revert,

                "r_convergence":
                    sum_convergence,

                "r_dense":
                    (
                        sum_step
                        +
                        sum_tet
                        +
                        sum_revert
                        +
                        sum_convergence
                    ),

                "r_finalization":
                    sum_finalization,

                "reward_v2":
                    undiscounted,
            }

            for gamma in GAMMAS:
                key = (
                    "return_gamma_"
                    +
                    str(
                        gamma
                    ).replace(
                        ".",
                        "_"
                    )
                )

                episode_row[
                    key
                ] = (
                    discounted_return(
                        rewards,
                        gamma,
                    )
                )

            episode_rows.append(
                episode_row
            )

    transition_scored = (
        pd.DataFrame(
            transition_rows
        )
    )

    episode_scored = (
        pd.DataFrame(
            episode_rows
        )
    )

    # ============================================================
    # Cross-check:
    #
    # V2 dense return MUST equal:
    #
    #     Reward V1 return - Reward V1 terminal component
    #
    # because V2 deliberately preserves all V1 dense components.
    # ============================================================

    if reward_v1_path.is_file():
        reward_v1 = pd.read_csv(
            reward_v1_path
        )

        v1_by_case = {
            str(
                row[
                    "case"
                ]
            ):
            (
                float(
                    row[
                        "reward_v1"
                    ]
                )
                -
                float(
                    row[
                        "r_terminal"
                    ]
                )
            )
            for _, row in reward_v1.iterrows()
        }

        for _, row in (
            episode_scored.iterrows()
        ):
            case = str(
                row[
                    "case"
                ]
            )

            if case not in v1_by_case:
                raise RuntimeError(
                    f"{case}: missing from "
                    "reward_v1_episode_summary.csv"
                )

            if not math.isclose(
                float(
                    row[
                        "r_dense"
                    ]
                ),
                v1_by_case[
                    case
                ],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"{case}: Reward V2 dense components "
                    "do not match Reward V1 minus "
                    "its selection-terminal component"
                )

    # ============================================================
    # Candidate profile diagnostics.
    # ============================================================

    sanity_rows = []

    for profile_name in (
        PROFILES
    ):
        df = (
            episode_scored[
                episode_scored[
                    "profile"
                ]
                ==
                profile_name
            ]
            .copy()
        )

        cylinder_original = float(
            df.loc[
                df[
                    "case"
                ]
                ==
                "cylinder_original",
                "reward_v2",
            ].iloc[
                0
            ]
        )

        cylinder_seed3 = float(
            df.loc[
                df[
                    "case"
                ]
                ==
                "cylinder_seed3",
                "reward_v2",
            ].iloc[
                0
            ]
        )

        full_returns = (
            df.loc[
                df[
                    "finalization_outcome"
                ]
                ==
                "FULL_HEX",
                "reward_v2",
            ]
        )

        failure_returns = (
            df.loc[
                df[
                    "finalization_outcome"
                ]
                !=
                "FULL_HEX",
                "reward_v2",
            ]
        )

        min_full = float(
            full_returns.min()
        )

        max_failure = float(
            failure_returns.max()
        )

        nonfull_return = float(
            df.loc[
                df[
                    "finalization_outcome"
                ]
                ==
                "NON_FULL_HEX",
                "reward_v2",
            ].iloc[
                0
            ]
        )

        crash_return = float(
            df.loc[
                df[
                    "finalization_outcome"
                ]
                ==
                "FINALIZATION_CRASH",
                "reward_v2",
            ].iloc[
                0
            ]
        )

        sanity_rows.append(
            {
                "profile":
                    profile_name,

                "cylinder_original":
                    cylinder_original,

                "cylinder_seed3":
                    cylinder_seed3,

                "original_beats_seed3":
                    int(
                        cylinder_original
                        >
                        cylinder_seed3
                    ),

                "min_full_hex_return":
                    min_full,

                "max_failure_return":
                    max_failure,

                "all_full_hex_beat_failures":
                    int(
                        min_full
                        >
                        max_failure
                    ),

                "non_full_hex_return":
                    nonfull_return,

                "crash_return":
                    crash_return,

                "crash_worse_than_nonfull":
                    int(
                        crash_return
                        <
                        nonfull_return
                    ),
            }
        )

    sanity = pd.DataFrame(
        sanity_rows
    )

    # ============================================================
    # Structural sanity must hold for every candidate profile.
    # ============================================================

    required_flags = (
        "original_beats_seed3",
        "all_full_hex_beat_failures",
        "crash_worse_than_nonfull",
    )

    for flag in required_flags:
        if not bool(
            (
                sanity[
                    flag
                ]
                ==
                1
            ).all()
        ):
            raise RuntimeError(
                f"Candidate reward sanity failed: "
                f"{flag}"
            )

    # ============================================================
    # Save
    # ============================================================

    transition_output = (
        args.output_dir
        /
        "reward_v2_candidate_transition_scored.csv"
    )

    episode_output = (
        args.output_dir
        /
        "reward_v2_candidate_episode_summary.csv"
    )

    sanity_output = (
        args.output_dir
        /
        "reward_v2_candidate_sanity.csv"
    )

    transition_scored.to_csv(
        transition_output,
        index=False,
    )

    episode_scored.to_csv(
        episode_output,
        index=False,
    )

    sanity.to_csv(
        sanity_output,
        index=False,
    )

    # ============================================================
    # Console summary
    # ============================================================

    print()
    print(
        "=" * 92
    )
    print(
        "FINAL-AWARE REWARD V2 CANDIDATE AUDIT"
    )
    print(
        "=" * 92
    )

    print()

    print(
        episode_scored[
            [
                "profile",
                "case",
                "finalization_outcome",
                "steps",
                "r_dense",
                "r_finalization",
                "reward_v2",
                "return_gamma_0_99",
                "return_gamma_0_995",
                "return_gamma_0_997",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 92
    )
    print(
        "SANITY"
    )
    print(
        "=" * 92
    )

    print(
        sanity.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:"
    )

    print(
        transition_output
    )

    print(
        episode_output
    )

    print(
        sanity_output
    )

    print()
    print(
        "PASS: Reward V2 candidate profiles "
        "preserve dense Reward V1 components, "
        "replace selection-terminal proxy reward "
        "with real finalization outcomes, and "
        "satisfy the required trajectory ordering."
    )


if __name__ == "__main__":
    main()
