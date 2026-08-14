import math
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


from envs.loopycuts_env import LoopyCutsEnv
from rewards.reward_v1 import (
    compute_reward_v1,
)
from rewards.transition_metrics import (
    extract_transition_metrics,
)


EXECUTABLE = (
    "/home/yjk/codes/LoopyCuts/"
    "volumetric_cutter/"
    "volumetric_cutter"
)

AUDIT_CSV = Path(
    "/home/yjk/loopycuts_test/"
    "reward_scale_audit_all/"
    "reward_v1_transition_scored.csv"
)


CASES = {
    "cylinder_original": {
        "mesh": (
            "/home/yjk/loopycuts_inputs/"
            "cylinder_plate_clean/"
            "cylinder_plate_rem_splitted.obj"
        ),
        "loop": (
            "/home/yjk/loopycuts_inputs/"
            "cylinder_plate_clean/"
            "cylinder_plate_rem_loop.txt"
        ),
        "expected_actions": [
            0,
            1,
            2,
            3,
        ],
    },

    "bracket_original": {
        "mesh": (
            "/home/yjk/codes/LoopyCuts/"
            "test_data/BracketInches/"
            "BracketInches_rem_rem_splitted.obj"
        ),
        "loop": (
            "/home/yjk/codes/LoopyCuts/"
            "test_data/BracketInches/"
            "BracketInches_rem_rem_loop.txt"
        ),
        "expected_actions": (
            list(
                range(
                    29
                )
            )
            +
            list(
                range(
                    81,
                    90,
                )
            )
        ),
    },
}


COMPONENT_MAP = {
    "step":
        "r_step",

    "tet_growth":
        "r_tet",

    "revert":
        "r_revert",

    "convergence":
        "r_convergence",

    "terminal":
        "r_terminal",

    "total":
        "reward_v1",
}


def assert_close(
    actual,
    expected,
    *,
    case,
    step,
    name,
):
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError(
            f"{case} step={step}: "
            f"{name} mismatch: "
            f"online={actual}, "
            f"offline={expected}"
        )


def run_case(
    *,
    case_name,
    spec,
    offline_df,
):
    offline = (
        offline_df[
            offline_df[
                "case"
            ]
            ==
            case_name
        ]
        .sort_values(
            "step"
        )
        .reset_index(
            drop=True
        )
    )

    expected_actions = (
        spec[
            "expected_actions"
        ]
    )

    assert len(
        offline
    ) == len(
        expected_actions
    )

    env = LoopyCutsEnv(
        executable=EXECUTABLE,
        mesh_file=spec[
            "mesh"
        ],
        loop_file=spec[
            "loop"
        ],
        echo_logs=False,
    )

    online_records = []

    try:
        observation, info = (
            env.reset(
                seed=123
            )
        )

        initial_actionable_count = len(
            env.legal_actions
        )

        print()
        print(
            "=" * 70
        )
        print(
            "CASE:",
            case_name,
        )
        print(
            "initial_actionable:",
            initial_actionable_count,
        )
        print(
            "=" * 70
        )

        for (
            index,
            expected_action,
        ) in enumerate(
            expected_actions
        ):
            if int(
                env.current_state[
                    "terminal"
                ]
            ):
                raise AssertionError(
                    f"{case_name}: "
                    "environment became terminal "
                    "before expected trajectory ended"
                )

            action = min(
                env.legal_actions
            )

            if (
                action
                !=
                expected_action
            ):
                raise AssertionError(
                    f"{case_name}: "
                    f"expected action "
                    f"{expected_action}, "
                    f"but min legal action "
                    f"is {action}"
                )

            state_before = dict(
                env.current_state
            )

            (
                observation,
                env_reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            assert (
                truncated
                is False
            )

            state_after = dict(
                env.current_state
            )

            metrics = (
                extract_transition_metrics(
                    state_before=state_before,
                    step_result=(
                        info[
                            "step_result"
                        ]
                    ),
                    state_after=state_after,
                )
            )

            reward = (
                compute_reward_v1(
                    metrics=metrics,
                    initial_actionable_count=(
                        initial_actionable_count
                    ),
                )
            )

            assert_close(
                env_reward,
                reward.total,
                case=case_name,
                step=metrics.step,
                name="env_reward",
            )

            offline_row = (
                offline.iloc[
                    index
                ]
            )

            # ========================================================
            # Transition identity must match first.
            # ========================================================

            assert (
                int(
                    offline_row[
                        "step"
                    ]
                )
                ==
                metrics.step
            )

            assert (
                int(
                    offline_row[
                        "loop_id"
                    ]
                )
                ==
                metrics.loop_id
            )

            assert (
                str(
                    offline_row[
                        "status"
                    ]
                )
                ==
                metrics.status
            )

            assert (
                int(
                    offline_row[
                        "reverted"
                    ]
                )
                ==
                metrics.reverted
            )

            assert (
                int(
                    offline_row[
                        "convergence_delta"
                    ]
                )
                ==
                metrics.convergence_delta
            )

            assert (
                int(
                    offline_row[
                        "first_convergence"
                    ]
                )
                ==
                metrics.first_convergence
            )

            assert (
                int(
                    offline_row[
                        "selection_success"
                    ]
                )
                ==
                metrics.selection_success
            )

            assert (
                int(
                    offline_row[
                        "terminal_failure"
                    ]
                )
                ==
                metrics.terminal_failure
            )

            # ========================================================
            # Geometry-derived transition metric must match offline
            # audit trajectory exactly.
            # ========================================================

            assert_close(
                metrics.log_tet_growth,
                offline_row[
                    "log_tet_growth"
                ],
                case=case_name,
                step=metrics.step,
                name="log_tet_growth",
            )

            # ========================================================
            # Every Reward V1 component must match the offline scorer.
            # ========================================================

            for (
                online_name,
                offline_name,
            ) in (
                COMPONENT_MAP.items()
            ):
                assert_close(
                    getattr(
                        reward,
                        online_name,
                    ),
                    offline_row[
                        offline_name
                    ],
                    case=case_name,
                    step=metrics.step,
                    name=online_name,
                )

            online_records.append(
                reward
            )

            print(
                f"step={metrics.step:2d} "
                f"id={metrics.loop_id:3d} "
                f"status={metrics.status:9s} "
                f"tet={reward.tet_growth:+.6f} "
                f"revert={reward.revert:+.3f} "
                f"conv={reward.convergence:+.3f} "
                f"terminal={reward.terminal:+.3f} "
                f"reward={reward.total:+.6f}"
            )

        # ============================================================
        # Trajectory must terminate exactly at expected end.
        # ============================================================

        assert int(
            env.current_state[
                "terminal"
            ]
        ) == 1

        assert len(
            online_records
        ) == len(
            offline
        )

        online_total = sum(
            record.total
            for record in online_records
        )

        offline_total = float(
            offline[
                "reward_v1"
            ].sum()
        )

        assert_close(
            online_total,
            offline_total,
            case=case_name,
            step="episode",
            name="episode_return",
        )

        print()
        print(
            "online episode return :",
            online_total,
        )
        print(
            "offline episode return:",
            offline_total,
        )

        if (
            case_name
            ==
            "cylinder_original"
        ):
            assert int(
                env.current_state[
                    "converged"
                ]
            ) == 1

            assert int(
                env.current_state[
                    "selection_success"
                ]
            ) == 1

        elif (
            case_name
            ==
            "bracket_original"
        ):
            assert int(
                env.current_state[
                    "converged"
                ]
            ) == 0

            assert int(
                env.current_state[
                    "selection_success"
                ]
            ) == 0

            #
            # Bracket must contain exactly one convergence-loss
            # transition:
            #
            #     loop 87, step 36
            #
            convergence_penalties = [
                (
                    expected_actions[i],
                    record.convergence,
                )
                for i, record
                in enumerate(
                    online_records
                )
                if record.convergence
                != 0.0
            ]

            assert (
                convergence_penalties
                ==
                [
                    (
                        87,
                        -1.0,
                    )
                ]
            )

            #
            # Terminal loop 89:
            #
            # reverted       -> -0.10
            # terminal fail  -> -3.00
            #
            final_reward = (
                online_records[
                    -1
                ]
            )

            assert (
                final_reward.revert
                ==
                -0.10
            )

            assert (
                final_reward.terminal
                ==
                -3.0
            )

    finally:
        env.close()


def main():
    if not AUDIT_CSV.is_file():
        raise FileNotFoundError(
            f"Offline scored CSV not found: "
            f"{AUDIT_CSV}"
        )

    offline_df = pd.read_csv(
        AUDIT_CSV
    )

    required_columns = {
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
        "r_step",
        "r_tet",
        "r_revert",
        "r_convergence",
        "r_terminal",
        "reward_v1",
    }

    missing = (
        required_columns
        -
        set(
            offline_df.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Offline reward CSV is missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    for (
        case_name,
        spec,
    ) in CASES.items():
        run_case(
            case_name=case_name,
            spec=spec,
            offline_df=offline_df,
        )

    print()
    print(
        "=" * 70
    )
    print(
        "PASS: online C++ transitions -> "
        "TransitionMetrics -> Reward V1 "
        "match offline scored trajectories "
        "component-by-component."
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
