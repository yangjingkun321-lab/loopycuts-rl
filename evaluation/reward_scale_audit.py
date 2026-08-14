from __future__ import annotations

import argparse
import csv
import gc
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(
    "/home/yjk/codes/loopycuts_rl"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from envs.loopycuts_env import LoopyCutsEnv
from rewards.transition_metrics import extract_transition_metrics


EXECUTABLE = Path(
    "/home/yjk/codes/LoopyCuts/"
    "volumetric_cutter/"
    "volumetric_cutter"
)

DEFAULT_OUTPUT_DIR = Path(
    "/home/yjk/loopycuts_test/"
    "reward_scale_audit"
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    mesh_file: Path
    policy: str
    replay: tuple[int, ...] | None = None


def derive_loop_file(
    mesh_file: Path,
) -> Path:
    suffix = "_splitted.obj"

    if not mesh_file.name.endswith(
        suffix
    ):
        raise ValueError(
            "Cannot derive _loop.txt from mesh name: "
            f"{mesh_file}"
        )

    prefix = mesh_file.name[
        : -len(suffix)
    ]

    return mesh_file.with_name(
        prefix
        +
        "_loop.txt"
    )


CYLINDER_MESH = Path(
    "/home/yjk/loopycuts_inputs/"
    "cylinder_plate_clean/"
    "cylinder_plate_rem_splitted.obj"
)


CASES = {
    "cylinder_original": CaseSpec(
        name="cylinder_original",
        mesh_file=CYLINDER_MESH,
        policy="min_legal",
    ),

    "cylinder_seed3": CaseSpec(
        name="cylinder_seed3",
        mesh_file=CYLINDER_MESH,
        policy="replay_seed3",
        replay=(
            0,
            1,
            31,
            3,
            7,
            23,
            22,
            51,
            29,
            43,
            34,
            8,
            19,
            57,
            9,
            33,
        ),
    ),

    "bracket_original": CaseSpec(
        name="bracket_original",
        mesh_file=Path(
            "/home/yjk/codes/LoopyCuts/"
            "test_data/BracketInches/"
            "BracketInches_rem_rem_splitted.obj"
        ),
        policy="min_legal",
    ),

    "deckel_original": CaseSpec(
        name="deckel_original",
        mesh_file=Path(
            "/home/yjk/codes/LoopyCuts/"
            "test_data/deckel/"
            "deckel_rem_splitted.obj"
        ),
        policy="min_legal",
    ),

    "eraser_ball_original": CaseSpec(
        name="eraser_ball_original",
        mesh_file=Path(
            "/home/yjk/codes/LoopyCuts/"
            "test_data/eraser_ball/"
            "eraser_ball_rem_rem_splitted.obj"
        ),
        policy="min_legal",
    ),

    "bimba_original": CaseSpec(
        name="bimba_original",
        mesh_file=Path(
            "/home/yjk/codes/LoopyCuts/"
            "test_data/bimba/"
            "bimba_rem_splitted.obj"
        ),
        policy="min_legal",
    ),
}


def validate_case_files(
    spec: CaseSpec,
) -> Path:
    if not EXECUTABLE.is_file():
        raise FileNotFoundError(
            f"Executable not found: {EXECUTABLE}"
        )

    if not spec.mesh_file.is_file():
        raise FileNotFoundError(
            f"Mesh not found for {spec.name}: "
            f"{spec.mesh_file}"
        )

    loop_file = derive_loop_file(
        spec.mesh_file
    )

    if not loop_file.is_file():
        raise FileNotFoundError(
            f"Derived loop file not found for "
            f"{spec.name}: {loop_file}"
        )

    return loop_file


def run_case(
    spec: CaseSpec,
):
    loop_file = validate_case_files(
        spec
    )

    print()
    print("=" * 70)
    print(
        f"CASE: {spec.name}"
    )
    print(
        f"POLICY: {spec.policy}"
    )
    print(
        f"MESH: {spec.mesh_file}"
    )
    print(
        f"LOOPS: {loop_file}"
    )
    print("=" * 70)

    env = LoopyCutsEnv(
        executable=EXECUTABLE,
        mesh_file=spec.mesh_file,
        loop_file=loop_file,
        echo_logs=False,
    )

    rows = []

    try:
        observation, info = env.reset(
            seed=0
        )

        initial_state = dict(
            env.current_state
        )

        initial_actionable = len(
            env.legal_actions
        )

        initial_tets = int(
            initial_state[
                "tets"
            ]
        )

        initial_verts = int(
            initial_state[
                "verts"
            ]
        )

        max_tets = initial_tets
        max_verts = initial_verts
        max_mm_polys = int(
            initial_state[
                "mm_polys"
            ]
        )

        max_nonmanifold = 0
        max_high_genus = 0
        max_buggy_chains = 0

        replay_cursor = 0

        while not int(
            env.current_state[
                "terminal"
            ]
        ):
            # --------------------------------------------------------
            # Action policy used only for scale auditing.
            # --------------------------------------------------------

            if spec.replay is None:
                if not env.legal_actions:
                    raise RuntimeError(
                        f"{spec.name}: non-terminal "
                        "state has no legal actions"
                    )

                action = min(
                    env.legal_actions
                )

            else:
                if (
                    replay_cursor
                    >=
                    len(
                        spec.replay
                    )
                ):
                    raise RuntimeError(
                        f"{spec.name}: replay sequence "
                        "ended before terminal"
                    )

                action = spec.replay[
                    replay_cursor
                ]

                replay_cursor += 1

                if action not in (
                    env.legal_actions
                ):
                    raise RuntimeError(
                        f"{spec.name}: replay action "
                        f"{action} is not legal at "
                        f"step "
                        f"{env.current_state['step']}"
                    )

            state_before = dict(
                env.current_state
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            if not np.isfinite(
                float(
                    reward
                )
            ):
                raise RuntimeError(
                    f"{spec.name}: environment returned "
                    "a non-finite reward"
                )

            if truncated:
                raise RuntimeError(
                    f"{spec.name}: unexpected truncation"
                )

            state_after = dict(
                env.current_state
            )

            step_result = dict(
                info[
                    "step_result"
                ]
            )

            metrics = (
                extract_transition_metrics(
                    state_before=state_before,
                    step_result=step_result,
                    state_after=state_after,
                )
            )

            row = {
                "case":
                    spec.name,

                "model":
                    spec.mesh_file.stem,

                "policy":
                    spec.policy,

                "loop_type":
                    step_result[
                        "loop_type"
                    ],

                **metrics.to_dict(),

                # ----------------------------------------------------
                # Raw surrounding state for later interpretation.
                # ----------------------------------------------------

                "verts_before_raw":
                    int(
                        state_before[
                            "verts"
                        ]
                    ),

                "verts_after_raw":
                    int(
                        state_after[
                            "verts"
                        ]
                    ),

                "tets_before_raw":
                    int(
                        state_before[
                            "tets"
                        ]
                    ),

                "tets_after_raw":
                    int(
                        state_after[
                            "tets"
                        ]
                    ),

                "mm_polys_before_raw":
                    int(
                        state_before[
                            "mm_polys"
                        ]
                    ),

                "mm_polys_after_raw":
                    int(
                        state_after[
                            "mm_polys"
                        ]
                    ),

                "nonmanifold_before_raw":
                    int(
                        state_before[
                            "nonmanifold_polys"
                        ]
                    ),

                "nonmanifold_after_raw":
                    int(
                        state_after[
                            "nonmanifold_polys"
                        ]
                    ),

                "high_genus_before_raw":
                    int(
                        state_before[
                            "high_genus_polys"
                        ]
                    ),

                "high_genus_after_raw":
                    int(
                        state_after[
                            "high_genus_polys"
                        ]
                    ),

                "buggy_chains_before_raw":
                    int(
                        state_before[
                            "buggy_chains"
                        ]
                    ),

                "buggy_chains_after_raw":
                    int(
                        state_after[
                            "buggy_chains"
                        ]
                    ),

                "abs_delta_log_nonmanifold":
                    abs(
                        metrics
                        .delta_log_nonmanifold
                    ),

                "abs_delta_log_high_genus":
                    abs(
                        metrics
                        .delta_log_high_genus
                    ),

                "abs_delta_log_buggy_chains":
                    abs(
                        metrics
                        .delta_log_buggy_chains
                    ),
            }

            rows.append(
                row
            )

            max_tets = max(
                max_tets,
                int(
                    state_after[
                        "tets"
                    ]
                ),
            )

            max_verts = max(
                max_verts,
                int(
                    state_after[
                        "verts"
                    ]
                ),
            )

            max_mm_polys = max(
                max_mm_polys,
                int(
                    state_after[
                        "mm_polys"
                    ]
                ),
            )

            max_nonmanifold = max(
                max_nonmanifold,
                int(
                    state_after[
                        "nonmanifold_polys"
                    ]
                ),
            )

            max_high_genus = max(
                max_high_genus,
                int(
                    state_after[
                        "high_genus_polys"
                    ]
                ),
            )

            max_buggy_chains = max(
                max_buggy_chains,
                int(
                    state_after[
                        "buggy_chains"
                    ]
                ),
            )

            print(
                f"step={metrics.step:3d} "
                f"id={metrics.loop_id:3d} "
                f"type={step_result['loop_type']:8s} "
                f"status={metrics.status:9s} "
                f"conv={metrics.convergence_delta:+d} "
                f"tet_log={metrics.log_tet_growth:.5f} "
                f"buggy="
                f"{state_after['buggy_chains']:3d} "
                f"available="
                f"{state_after['available']:3d}"
            )

        # ------------------------------------------------------------
        # A replay trajectory must terminate exactly at its final item.
        # ------------------------------------------------------------

        if spec.replay is not None:
            if replay_cursor != len(
                spec.replay
            ):
                raise RuntimeError(
                    f"{spec.name}: environment became terminal "
                    f"after {replay_cursor} replay actions, "
                    f"but replay contains "
                    f"{len(spec.replay)}"
                )

        final_state = dict(
            env.current_state
        )

        steps = len(
            rows
        )

        committed = sum(
            int(
                row[
                    "committed"
                ]
            )
            for row in rows
        )

        reverted = sum(
            int(
                row[
                    "reverted"
                ]
            )
            for row in rows
        )

        convergence_up = sum(
            int(
                row[
                    "convergence_delta"
                ]
                ==
                1
            )
            for row in rows
        )

        convergence_down = sum(
            int(
                row[
                    "convergence_delta"
                ]
                ==
                -1
            )
            for row in rows
        )

        first_convergence_steps = [
            int(
                row[
                    "step"
                ]
            )
            for row in rows
            if int(
                row[
                    "first_convergence"
                ]
            )
        ]

        phase_close_steps = [
            int(
                row[
                    "step"
                ]
            )
            for row in rows
            if int(
                row[
                    "phase_closed_this_step"
                ]
            )
        ]

        total_step_time = sum(
            float(
                row[
                    "step_time"
                ]
            )
            for row in rows
        )

        final_tets = int(
            final_state[
                "tets"
            ]
        )

        final_verts = int(
            final_state[
                "verts"
            ]
        )

        summary = {
            "case":
                spec.name,

            "model":
                spec.mesh_file.stem,

            "policy":
                spec.policy,

            "num_loops":
                int(
                    final_state[
                        "loops"
                    ]
                ),

            "initial_actionable":
                initial_actionable,

            "steps":
                steps,

            "committed":
                committed,

            "reverted":
                reverted,

            "other_status":
                steps
                -
                committed
                -
                reverted,

            "convergence_up_events":
                convergence_up,

            "convergence_down_events":
                convergence_down,

            "first_convergence_step":
                (
                    first_convergence_steps[0]
                    if first_convergence_steps
                    else ""
                ),

            "phase_close_step":
                (
                    phase_close_steps[0]
                    if phase_close_steps
                    else ""
                ),

            "terminal":
                int(
                    final_state[
                        "terminal"
                    ]
                ),

            "final_converged":
                int(
                    final_state[
                        "converged"
                    ]
                ),

            "selection_success":
                int(
                    final_state[
                        "selection_success"
                    ]
                ),

            "initial_tets":
                initial_tets,

            "final_tets":
                final_tets,

            "max_tets":
                max_tets,

            "tet_ratio":
                (
                    final_tets
                    /
                    initial_tets
                ),

            "sum_log_tet_growth":
                sum(
                    float(
                        row[
                            "log_tet_growth"
                        ]
                    )
                    for row in rows
                ),

            "initial_verts":
                initial_verts,

            "final_verts":
                final_verts,

            "max_verts":
                max_verts,

            "vert_ratio":
                (
                    final_verts
                    /
                    initial_verts
                ),

            "sum_log_vert_growth":
                sum(
                    float(
                        row[
                            "log_vert_growth"
                        ]
                    )
                    for row in rows
                ),

            "max_mm_polys":
                max_mm_polys,

            "max_nonmanifold_polys":
                max_nonmanifold,

            "max_high_genus_polys":
                max_high_genus,

            "max_buggy_chains":
                max_buggy_chains,

            "final_nonmanifold_polys":
                int(
                    final_state[
                        "nonmanifold_polys"
                    ]
                ),

            "final_high_genus_polys":
                int(
                    final_state[
                        "high_genus_polys"
                    ]
                ),

            "final_buggy_chains":
                int(
                    final_state[
                        "buggy_chains"
                    ]
                ),

            "total_step_time":
                total_step_time,

            "mean_step_time":
                (
                    total_step_time
                    /
                    steps
                    if steps
                    else 0.0
                ),

            "max_step_time":
                (
                    max(
                        float(
                            row[
                                "step_time"
                            ]
                        )
                        for row in rows
                    )
                    if rows
                    else 0.0
                ),

            "max_available_drop":
                (
                    max(
                        int(
                            row[
                                "available_drop"
                            ]
                        )
                        for row in rows
                    )
                    if rows
                    else 0
                ),
        }

        print()
        print(
            "SUMMARY:",
            summary,
        )

        return (
            rows,
            summary,
        )

    finally:
        env.close()


def write_csv(
    path: Path,
    rows: list[dict],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def make_scale_stats(
    rows: list[dict],
) -> list[dict]:
    metrics = (
        "log_tet_growth",
        "log_vert_growth",
        "step_time",
        "abs_delta_log_nonmanifold",
        "abs_delta_log_high_genus",
        "abs_delta_log_buggy_chains",
        "post_log_nonmanifold",
        "post_log_high_genus",
        "post_log_buggy_chains",
        "delta_log_mm_polys",
        "available_drop",
    )

    result = []

    for name in metrics:
        values = np.asarray(
            [
                float(
                    row[
                        name
                    ]
                )
                for row in rows
            ],
            dtype=np.float64,
        )

        if values.size == 0:
            continue

        result.append(
            {
                "metric":
                    name,

                "count":
                    int(
                        values.size
                    ),

                "mean":
                    float(
                        values.mean()
                    ),

                "std":
                    float(
                        values.std()
                    ),

                "min":
                    float(
                        values.min()
                    ),

                "p50":
                    float(
                        np.percentile(
                            values,
                            50,
                        )
                    ),

                "p90":
                    float(
                        np.percentile(
                            values,
                            90,
                        )
                    ),

                "p95":
                    float(
                        np.percentile(
                            values,
                            95,
                        )
                    ),

                "max":
                    float(
                        values.max()
                    ),
            }
        )

    return result


def write_outputs(
    output_dir: Path,
    transition_rows: list[dict],
    summaries: list[dict],
):
    write_csv(
        output_dir
        /
        "transition_metrics.csv",
        transition_rows,
    )

    write_csv(
        output_dir
        /
        "episode_summary.csv",
        summaries,
    )

    write_csv(
        output_dir
        /
        "scale_stats.csv",
        make_scale_stats(
            transition_rows
        ),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        choices=[
            "all",
            *CASES.keys(),
        ],
        default="all",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if args.case == "all":
        selected = list(
            CASES.values()
        )
    else:
        selected = [
            CASES[
                args.case
            ]
        ]

    transition_rows = []
    summaries = []

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for spec in selected:
        rows, summary = (
            run_case(
                spec
            )
        )

        transition_rows.extend(
            rows
        )

        summaries.append(
            summary
        )

        #
        # Save after every model so already completed expensive
        # episodes are not lost if a later model fails.
        #
        write_outputs(
            args.output_dir,
            transition_rows,
            summaries,
        )

        gc.collect()

        print()
        print(
            "Saved completed audit data to:",
            args.output_dir,
        )

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(
        "transition CSV:",
        args.output_dir
        /
        "transition_metrics.csv",
    )

    print(
        "episode summary:",
        args.output_dir
        /
        "episode_summary.csv",
    )

    print(
        "scale statistics:",
        args.output_dir
        /
        "scale_stats.csv",
    )


if __name__ == "__main__":
    main()
