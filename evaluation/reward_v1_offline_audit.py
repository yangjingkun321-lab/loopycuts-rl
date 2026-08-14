from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    "/home/yjk/codes/loopycuts_rl"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


AUDIT_DIR = Path(
    "/home/yjk/loopycuts_test/"
    "reward_scale_audit_all"
)

TRANSITION_CSV = (
    AUDIT_DIR
    /
    "transition_metrics.csv"
)

EPISODE_SUMMARY_CSV = (
    AUDIT_DIR
    /
    "episode_summary.csv"
)


def main():
    transitions = pd.read_csv(
        TRANSITION_CSV
    )

    episodes = pd.read_csv(
        EPISODE_SUMMARY_CSV
    )

    # ============================================================
    # Validate transition-level source data.
    # ============================================================

    required_transition = {
        "case",
        "step",
        "loop_id",
        "status",
        "log_tet_growth",
        "reverted",
        "convergence_delta",
        "first_convergence",
        "selection_success",
        "terminal_failure",
    }

    missing = (
        required_transition
        -
        set(
            transitions.columns
        )
    )

    if missing:
        raise RuntimeError(
            "transition_metrics.csv is missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    # ============================================================
    # initial_actionable is an episode-level constant.
    #
    # Do NOT duplicate this data during the original geometry audit
    # and do NOT rerun LoopyCuts. Merge it from episode_summary.csv.
    # ============================================================

    required_episode = {
        "case",
        "initial_actionable",
    }

    missing = (
        required_episode
        -
        set(
            episodes.columns
        )
    )

    if missing:
        raise RuntimeError(
            "episode_summary.csv is missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    episode_actionable = (
        episodes[
            [
                "case",
                "initial_actionable",
            ]
        ]
        .copy()
    )

    if episode_actionable[
        "case"
    ].duplicated().any():
        duplicates = (
            episode_actionable[
                episode_actionable[
                    "case"
                ].duplicated(
                    keep=False
                )
            ][
                "case"
            ]
            .tolist()
        )

        raise RuntimeError(
            "episode_summary.csv has duplicate case rows: "
            f"{duplicates}"
        )

    df = transitions.merge(
        episode_actionable,
        on="case",
        how="left",
        validate="many_to_one",
    )

    if df[
        "initial_actionable"
    ].isna().any():
        missing_cases = (
            df.loc[
                df[
                    "initial_actionable"
                ].isna(),
                "case",
            ]
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Could not resolve initial_actionable for cases: "
            f"{missing_cases}"
        )

    df[
        "initial_actionable"
    ] = (
        df[
            "initial_actionable"
        ]
        .astype(
            int
        )
    )

    if (
        df[
            "initial_actionable"
        ]
        <= 0
    ).any():
        raise RuntimeError(
            "initial_actionable must be positive"
        )

    print(
        "transitions:",
        len(df),
    )

    print(
        "cases:",
        df[
            "case"
        ].nunique(),
    )

    print()

    print(
        "initial_actionable by case:"
    )

    print(
        df.groupby(
            "case",
            sort=False,
        )[
            "initial_actionable"
        ]
        .first()
        .to_string()
    )

    # ============================================================
    # Selection Reward V1
    #
    # r =
    #   - 1 / initial_actionable
    #   - log_tet_growth
    #   - 0.10 * reverted
    #   - 1.00 * convergence_loss
    #   + 1.00 * convergence_recovery
    #   + 3.00 * selection_success
    #   - 3.00 * terminal_failure
    #
    # First convergence itself receives no convergence bonus.
    # ============================================================

    df[
        "r_step"
    ] = (
        -1.0
        /
        df[
            "initial_actionable"
        ]
    )

    df[
        "r_tet"
    ] = (
        -df[
            "log_tet_growth"
        ]
    )

    df[
        "r_revert"
    ] = (
        -0.10
        *
        df[
            "reverted"
        ]
    )

    df[
        "r_convergence"
    ] = 0.0

    convergence_loss = (
        df[
            "convergence_delta"
        ]
        ==
        -1
    )

    convergence_recovery = (
        (
            df[
                "convergence_delta"
            ]
            ==
            1
        )
        &
        (
            df[
                "first_convergence"
            ]
            ==
            0
        )
    )

    df.loc[
        convergence_loss,
        "r_convergence",
    ] = -1.0

    df.loc[
        convergence_recovery,
        "r_convergence",
    ] = 1.0

    df[
        "r_terminal"
    ] = (
        3.0
        *
        df[
            "selection_success"
        ]
        -
        3.0
        *
        df[
            "terminal_failure"
        ]
    )

    df[
        "reward_v1"
    ] = (
        df[
            "r_step"
        ]
        +
        df[
            "r_tet"
        ]
        +
        df[
            "r_revert"
        ]
        +
        df[
            "r_convergence"
        ]
        +
        df[
            "r_terminal"
        ]
    )

    # ============================================================
    # Episode-level component totals.
    # ============================================================

    summary = (
        df.groupby(
            "case",
            sort=False,
        )[
            [
                "r_step",
                "r_tet",
                "r_revert",
                "r_convergence",
                "r_terminal",
                "reward_v1",
            ]
        ]
        .sum()
    )

    print()

    print(
        "===================================="
    )

    print(
        "SELECTION REWARD V1 OFFLINE AUDIT"
    )

    print(
        "===================================="
    )

    print()

    print(
        summary.to_string()
    )

    print()

    print(
        "===================================="
    )

    print(
        "EPISODE RETURN RANKING"
    )

    print(
        "===================================="
    )

    ranking = (
        summary[
            "reward_v1"
        ]
        .sort_values(
            ascending=False
        )
    )

    print()

    print(
        ranking.to_string()
    )

    # ============================================================
    # Important sanity relations.
    # ============================================================

    expected_cases = {
        "cylinder_original",
        "cylinder_seed3",
        "bracket_original",
        "deckel_original",
        "eraser_ball_original",
        "bimba_original",
    }

    actual_cases = set(
        summary.index
    )

    if actual_cases != expected_cases:
        raise RuntimeError(
            "Unexpected case set. "
            f"Expected {sorted(expected_cases)}, "
            f"got {sorted(actual_cases)}"
        )

    #
    # Same model, both selection-successful:
    # original Stage-2 order should outrank the adverse seed3 order.
    #
    if not (
        summary.loc[
            "cylinder_original",
            "reward_v1",
        ]
        >
        summary.loc[
            "cylinder_seed3",
            "reward_v1",
        ]
    ):
        raise RuntimeError(
            "Reward V1 fails Cylinder order-quality sanity check"
        )

    #
    # Known terminal selection failure should be negative.
    #
    if not (
        summary.loc[
            "bracket_original",
            "reward_v1",
        ]
        <
        0.0
    ):
        raise RuntimeError(
            "Reward V1 fails Bracket terminal-failure sanity check"
        )

    #
    # All five known selection-success trajectories in this audit
    # should remain positive under the current V1 calibration.
    #
    successful_cases = (
        "cylinder_original",
        "cylinder_seed3",
        "deckel_original",
        "eraser_ball_original",
        "bimba_original",
    )

    for case in successful_cases:
        if not (
            summary.loc[
                case,
                "reward_v1",
            ]
            >
            0.0
        ):
            raise RuntimeError(
                "Reward V1 gives non-positive return to "
                f"known selection-success case {case}"
            )

    # ============================================================
    # Save scored transitions and episode summary.
    # ============================================================

    transition_output = (
        AUDIT_DIR
        /
        "reward_v1_transition_scored.csv"
    )

    summary_output = (
        AUDIT_DIR
        /
        "reward_v1_episode_summary.csv"
    )

    df.to_csv(
        transition_output,
        index=False,
    )

    summary.reset_index().to_csv(
        summary_output,
        index=False,
    )

    print()

    print(
        "Saved transition scores:",
        transition_output,
    )

    print(
        "Saved episode scores:",
        summary_output,
    )

    print()

    print(
        "PASS: Selection Reward V1 satisfies "
        "the current offline trajectory sanity checks."
    )


if __name__ == "__main__":
    main()
