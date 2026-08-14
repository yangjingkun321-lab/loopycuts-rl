from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from bridge.cpp_client import LoopyCutsClient
from evaluation.baseline_audit import (
    EXECUTABLE,
    FROZEN_MANIFEST,
    validate_initial_action_space,
)
from policies.simple import RandomPolicy
from runtime.resource_monitor import ResourceMonitor


DEFAULT_OUTPUT_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "train_resource_feasibility_v1/"
    "resource_stress_pilots"
)


def load_model_row(
    *,
    model: str,
    split: str,
):
    with FROZEN_MANIFEST.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    matches = [
        row
        for row in rows
        if (
            row["model"] == model
            and
            row["split"] == split
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one manifest row for "
            f"model={model}, split={split}; "
            f"found {len(matches)}"
        )

    return matches[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--split",
        default="train",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--process-swap-stop-mb",
        type=float,
        default=0.0,
        help=(
            "Stop before starting another STEP once "
            "completed-step process swap exceeds this value."
        ),
    )

    parser.add_argument(
        "--min-mem-stop-mb",
        type=float,
        default=250.0,
        help=(
            "Stop before starting another STEP once "
            "MemAvailable falls below this value."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    args = parser.parse_args()

    row = load_model_row(
        model=args.model,
        split=args.split,
    )

    output_root = args.output_root
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_root
        /
        (
            f"{args.model}_"
            f"random_seed{args.seed}.json"
        )
    )

    policy = RandomPolicy(
        seed=args.seed
    )

    policy.reset()

    trajectory = []

    stop_reason = None

    with LoopyCutsClient(
        executable=EXECUTABLE,
        mesh_file=Path(
            row["mesh_file"]
        ),
        loop_file=Path(
            row["loop_file"]
        ),
        echo_logs=False,
    ) as client:

        initial_actions = (
            validate_initial_action_space(
                model=args.model,
                row=row,
                loop_file=Path(
                    row["loop_file"]
                ),
                client=client,
            )
        )

        initial_state = dict(
            client.state
        )

        monitor = ResourceMonitor(
            pid=client.process.pid,
            sample_interval_s=1.0,
        )

        monitor.start()

        try:
            while not int(
                client.state[
                    "terminal"
                ]
            ):
                if (
                    len(trajectory)
                    >=
                    args.max_steps
                ):
                    stop_reason = (
                        "MAX_STEPS"
                    )
                    break

                actions = list(
                    client.actions
                )

                if not actions:
                    raise RuntimeError(
                        "Non-terminal state has "
                        "no legal actions"
                    )

                action = policy.select(
                    client.state,
                    actions,
                )

                before = dict(
                    client.state
                )

                step_index = (
                    len(trajectory)
                    +
                    1
                )

                print(
                    f"BEGIN "
                    f"step={step_index} "
                    f"action={action} "
                    f"available={before['available']} "
                    f"tets={before['tets']} "
                    f"mm_polys={before['mm_polys']}",
                    flush=True,
                )

                t0 = (
                    time.perf_counter()
                )

                (
                    step_result,
                    state,
                    next_actions,
                ) = client.step(
                    action
                )

                wall_s = (
                    time.perf_counter()
                    -
                    t0
                )

                stats = (
                    monitor.snapshot()
                )

                item = {
                    "step":
                        step_index,

                    "action":
                        int(action),

                    "status":
                        step_result[
                            "status"
                        ],

                    "wall_s":
                        wall_s,

                    "cpp_step_s":
                        step_result.get(
                            "step_time"
                        ),

                    "available":
                        int(
                            state[
                                "available"
                            ]
                        ),

                    "verts":
                        int(
                            state[
                                "verts"
                            ]
                        ),

                    "tets":
                        int(
                            state[
                                "tets"
                            ]
                        ),

                    "mm_polys":
                        int(
                            state[
                                "mm_polys"
                            ]
                        ),

                    "converged":
                        int(
                            state[
                                "converged"
                            ]
                        ),

                    "regular_phase_closed":
                        int(
                            state[
                                "regular_phase_closed"
                            ]
                        ),

                    "peak_rss_mb":
                        stats.peak_rss_mb,

                    "peak_process_swap_mb":
                        stats.peak_process_swap_mb,

                    "min_mem_available_mb":
                        stats.min_mem_available_mb,
                }

                trajectory.append(
                    item
                )

                print(
                    f"END   "
                    f"step={step_index} "
                    f"status={item['status']} "
                    f"tets={item['tets']} "
                    f"rss_mb={item['peak_rss_mb']:.1f} "
                    f"swap_mb="
                    f"{item['peak_process_swap_mb']:.1f} "
                    f"min_mem_mb="
                    f"{item['min_mem_available_mb']:.1f}",
                    flush=True,
                )

                if (
                    stats.peak_process_swap_mb
                    >
                    args.process_swap_stop_mb
                ):
                    stop_reason = (
                        "PROCESS_SWAP_THRESHOLD"
                    )
                    break

                if (
                    stats.min_mem_available_mb
                    <
                    args.min_mem_stop_mb
                ):
                    stop_reason = (
                        "MIN_MEM_THRESHOLD"
                    )
                    break

        finally:
            final_stats = (
                monitor.stop()
            )

        final_state = dict(
            client.state
        )

    completed_terminal = bool(
        int(
            final_state[
                "terminal"
            ]
        )
    )

    if (
        completed_terminal
        and
        stop_reason is None
    ):
        status = (
            "SELECTION_TERMINAL"
        )

    else:
        status = (
            "RESOURCE_PILOT_STOPPED"
        )

    record = {
        "model":
            args.model,

        "split":
            args.split,

        "policy":
            "random",

        "seed":
            args.seed,

        "pilot_type":
            "selection_resource_feasibility",

        "status":
            status,

        "stop_reason":
            stop_reason,

        "terminal":
            completed_terminal,

        "finalization_attempted":
            False,

        "initial_actionable":
            len(
                initial_actions
            ),

        "initial_verts":
            int(
                initial_state[
                    "verts"
                ]
            ),

        "initial_tets":
            int(
                initial_state[
                    "tets"
                ]
            ),

        "completed_steps":
            len(
                trajectory
            ),

        "final_selection_state":
            final_state,

        "partial_tet_ratio":
            (
                int(
                    final_state[
                        "tets"
                    ]
                )
                /
                int(
                    initial_state[
                        "tets"
                    ]
                )
            ),

        "peak_rss_mb":
            final_stats.peak_rss_mb,

        "peak_process_swap_mb":
            final_stats.peak_process_swap_mb,

        "min_mem_available_mb":
            final_stats.min_mem_available_mb,

        "max_system_swap_used_mb":
            final_stats.max_system_swap_used_mb,

        "trajectory":
            trajectory,
    }

    output_file.write_text(
        json.dumps(
            record,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("RANDOM RESOURCE PILOT RESULT")
    print("=" * 88)
    print("model:", args.model)
    print("status:", status)
    print("stop_reason:", stop_reason)
    print(
        "completed_steps:",
        len(trajectory),
    )
    print(
        "partial_tet_ratio:",
        record[
            "partial_tet_ratio"
        ],
    )
    print(
        "peak_rss_mb:",
        final_stats.peak_rss_mb,
    )
    print(
        "peak_process_swap_mb:",
        final_stats.peak_process_swap_mb,
    )
    print(
        "min_mem_available_mb:",
        final_stats.min_mem_available_mb,
    )
    print(
        "output:",
        output_file,
    )


if __name__ == "__main__":
    main()
