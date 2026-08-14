from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(
    "/home/yjk/codes/loopycuts_rl"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from bridge.cpp_client import (
    LoopyCutsClient,
)

from evaluation.episode_runner import (
    run_episode,
)


EXECUTABLE = (
    "/home/yjk/codes/LoopyCuts/"
    "volumetric_cutter/"
    "volumetric_cutter"
)


# ============================================================
# Policies used only for this deterministic audit.
#
# Keep them local so the audit does not depend on any future
# changes to RL/training policy classes.
# ============================================================

class MinLegalPolicy:
    """
    Reproduce the frozen original-forward Stage-2 traversal:

        select the minimum currently authoritative legal loop ID.

    Legality remains entirely controlled by C++ ACTIONS.
    """

    def reset(
        self,
    ):
        pass

    def select(
        self,
        state,
        actions,
    ):
        if not actions:
            raise RuntimeError(
                "MinLegalPolicy received no legal actions"
            )

        return min(
            actions
        )


class ReplaySequencePolicy:
    """
    Replay an already verified exact action sequence.

    Every replayed ID must still be legal in the current C++ state.
    """

    def __init__(
        self,
        sequence,
    ):
        self.sequence = tuple(
            int(x)
            for x in sequence
        )

        self.index = 0

    def reset(
        self,
    ):
        self.index = 0

    def select(
        self,
        state,
        actions,
    ):
        if self.index >= len(
            self.sequence
        ):
            raise RuntimeError(
                "Replay sequence exhausted before "
                "Stage-2 terminal"
            )

        action = self.sequence[
            self.index
        ]

        if action not in actions:
            raise RuntimeError(
                f"Replay action {action} is illegal at "
                f"sequence index {self.index}. "
                f"Legal actions: {actions}"
            )

        self.index += 1

        return action


# ============================================================
# Audit cases
# ============================================================

CYLINDER_MESH = (
    "/home/yjk/loopycuts_inputs/"
    "cylinder_plate_clean/"
    "cylinder_plate_rem_splitted.obj"
)

CYLINDER_LOOPS = (
    "/home/yjk/loopycuts_inputs/"
    "cylinder_plate_clean/"
    "cylinder_plate_rem_loop.txt"
)


CYLINDER_SEED3 = (
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
)


@dataclass(frozen=True)
class AuditCase:
    name: str
    mesh: str
    loops: str
    policy_kind: str
    expected_steps: int
    expected_selection_success: int
    replay_sequence: tuple[int, ...] = ()


CASES = {
    "cylinder_original":
        AuditCase(
            name="cylinder_original",
            mesh=CYLINDER_MESH,
            loops=CYLINDER_LOOPS,
            policy_kind="min_legal",
            expected_steps=4,
            expected_selection_success=1,
        ),

    "cylinder_seed3":
        AuditCase(
            name="cylinder_seed3",
            mesh=CYLINDER_MESH,
            loops=CYLINDER_LOOPS,
            policy_kind="replay",
            replay_sequence=CYLINDER_SEED3,
            expected_steps=16,
            expected_selection_success=1,
        ),

    "bracket_original":
        AuditCase(
            name="bracket_original",
            mesh=(
                "/home/yjk/codes/LoopyCuts/"
                "test_data/BracketInches/"
                "BracketInches_rem_rem_splitted.obj"
            ),
            loops=(
                "/home/yjk/codes/LoopyCuts/"
                "test_data/BracketInches/"
                "BracketInches_rem_rem_loop.txt"
            ),
            policy_kind="min_legal",
            expected_steps=38,
            expected_selection_success=0,
        ),

    "deckel_original":
        AuditCase(
            name="deckel_original",
            mesh=(
                "/home/yjk/codes/LoopyCuts/"
                "test_data/deckel/"
                "deckel_rem_splitted.obj"
            ),
            loops=(
                "/home/yjk/codes/LoopyCuts/"
                "test_data/deckel/"
                "deckel_rem_loop.txt"
            ),
            policy_kind="min_legal",
            expected_steps=23,
            expected_selection_success=1,
        ),

    "eraser_ball_original":
        AuditCase(
            name="eraser_ball_original",
            mesh=(
                "/home/yjk/codes/LoopyCuts/"
                "test_data/eraser_ball/"
                "eraser_ball_rem_rem_splitted.obj"
            ),
            loops=(
                "/home/yjk/codes/LoopyCuts/"
                "test_data/eraser_ball/"
                "eraser_ball_rem_rem_loop.txt"
            ),
            policy_kind="min_legal",
            expected_steps=39,
            expected_selection_success=1,
        ),

    "bimba_original":
        AuditCase(
            name="bimba_original",
            mesh=(
                "/home/yjk/codes/LoopyCuts/"
                "test_data/bimba/"
                "bimba_rem_splitted.obj"
            ),
            loops=(
                "/home/yjk/codes/LoopyCuts/"
                "test_data/bimba/"
                "bimba_rem_loop.txt"
            ),
            policy_kind="min_legal",
            expected_steps=24,
            expected_selection_success=1,
        ),
}


SUMMARY_FIELDS = [
    "case",
    "policy",
    "outcome",

    "num_steps",
    "actions",

    "selection_terminal",
    "selection_converged",
    "selection_success",

    "selection_verts",
    "selection_tets",
    "selection_mm_polys",

    "finalization_attempted",
    "finalization_completed",
    "finalization_process_terminated",
    "finalization_crashed",

    "returncode",
    "signal",
    "signal_name",

    "final_hex",
    "final_total_polys",
    "full_hex",

    "finalized_state_flag",
    "final_mm_polys",

    "wall_time",

    "output_file_count",
    "output_bytes",
]


def make_policy(
    case: AuditCase,
):
    if (
        case.policy_kind
        ==
        "min_legal"
    ):
        return MinLegalPolicy()

    if (
        case.policy_kind
        ==
        "replay"
    ):
        return ReplaySequencePolicy(
            case.replay_sequence
        )

    raise ValueError(
        f"Unknown policy kind: "
        f"{case.policy_kind}"
    )


def classify_outcome(
    result,
):
    if not result[
        "finalization_attempted"
    ]:
        raise RuntimeError(
            "Finalization was not attempted"
        )

    if result[
        "finalization_completed"
    ]:
        final_result = result[
            "final_result"
        ]

        if final_result is None:
            raise RuntimeError(
                "FINALIZE completed but final_result is None"
            )

        if (
            "full_hex"
            not in final_result
        ):
            raise RuntimeError(
                "FINAL_RESULT is missing full_hex"
            )

        full_hex = int(
            final_result[
                "full_hex"
            ]
        )

        if full_hex not in (
            0,
            1,
        ):
            raise RuntimeError(
                f"Invalid full_hex value: "
                f"{full_hex}"
            )

        if full_hex == 1:
            return "FULL_HEX"

        return "NON_FULL_HEX"

    if result[
        "finalization_process_terminated"
    ]:
        #
        # A genuine non-zero process crash is a geometric/runtime
        # finalization outcome that we want to measure.
        #
        if result[
            "finalization_crashed"
        ]:
            return "FINALIZATION_CRASH"

        #
        # A clean/unknown early process termination is not a valid
        # mesh-quality label. Treat it as infrastructure failure.
        #
        raise RuntimeError(
            "RL server terminated during FINALIZE "
            "without a non-zero crash return code. "
            f"returncode="
            f"{result['finalization_returncode']}"
        )

    raise RuntimeError(
        "FINALIZE was attempted but neither completed "
        "nor reported process termination"
    )


def output_stats(
    output_dir: Path,
):
    files = [
        path
        for path in output_dir.rglob("*")
        if path.is_file()
    ]

    total_bytes = sum(
        path.stat().st_size
        for path in files
    )

    return (
        len(files),
        total_bytes,
    )


def save_rows(
    root: Path,
    rows,
):
    csv_path = (
        root
        /
        "finalization_outcomes.csv"
    )

    json_path = (
        root
        /
        "finalization_outcomes.json"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_FIELDS,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    json_path.write_text(
        json.dumps(
            rows,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_case(
    case: AuditCase,
    root: Path,
):
    output_dir = (
        root
        /
        "outputs"
        /
        case.name
    )

    if output_dir.exists():
        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mesh = Path(
        case.mesh
    )

    loops = Path(
        case.loops
    )

    if not mesh.is_file():
        raise FileNotFoundError(
            mesh
        )

    if not loops.is_file():
        raise FileNotFoundError(
            loops
        )

    policy = make_policy(
        case
    )

    print()
    print(
        "=" * 78
    )
    print(
        "CASE:",
        case.name
    )
    print(
        "POLICY:",
        case.policy_kind
    )
    print(
        "MESH:",
        mesh
    )
    print(
        "LOOPS:",
        loops
    )
    print(
        "OUTPUT:",
        output_dir
    )
    print(
        "=" * 78
    )

    start = time.perf_counter()

    with LoopyCutsClient(
        executable=EXECUTABLE,
        mesh_file=mesh,
        loop_file=loops,
        echo_logs=False,
    ) as client:

        result = run_episode(
            client=client,
            policy=policy,
            output_dir=output_dir,
            finalize=True,
        )

    wall_time = (
        time.perf_counter()
        -
        start
    )

    actions = [
        int(
            item[
                "action"
            ]
        )
        for item
        in result[
            "trajectory"
        ]
    ]

    # ============================================================
    # Selection regression against the already verified D2-B
    # trajectories. Finalization must not hide a selection drift.
    # ============================================================

    if (
        result[
            "num_steps"
        ]
        !=
        case.expected_steps
    ):
        raise RuntimeError(
            f"{case.name}: selection step regression: "
            f"expected {case.expected_steps}, "
            f"got {result['num_steps']}"
        )

    if (
        int(
            result[
                "selection_success"
            ]
        )
        !=
        case.expected_selection_success
    ):
        raise RuntimeError(
            f"{case.name}: selection_success regression: "
            f"expected "
            f"{case.expected_selection_success}, "
            f"got "
            f"{result['selection_success']}"
        )

    if isinstance(
        policy,
        ReplaySequencePolicy,
    ):
        if (
            policy.index
            !=
            len(
                policy.sequence
            )
        ):
            raise RuntimeError(
                f"{case.name}: terminal reached after consuming "
                f"{policy.index}/{len(policy.sequence)} "
                "replay actions"
            )

    outcome = classify_outcome(
        result
    )

    selection_state = result[
        "selection_state"
    ]

    final_result = (
        result[
            "final_result"
        ]
        or
        {}
    )

    final_state = (
        result[
            "final_state"
        ]
        or
        {}
    )

    file_count, total_bytes = (
        output_stats(
            output_dir
        )
    )

    if result[
        "finalization_log_tail"
    ]:
        (
            output_dir
            /
            "finalization_log_tail.txt"
        ).write_text(
            "\n".join(
                result[
                    "finalization_log_tail"
                ]
            )
            +
            "\n",
            encoding="utf-8",
        )

        #
        # Recompute after writing the diagnostic file.
        #
        file_count, total_bytes = (
            output_stats(
                output_dir
            )
        )

    row = {
        "case":
            case.name,

        "policy":
            case.policy_kind,

        "outcome":
            outcome,

        "num_steps":
            result[
                "num_steps"
            ],

        "actions":
            json.dumps(
                actions
            ),

        "selection_terminal":
            int(
                result[
                    "terminal"
                ]
            ),

        "selection_converged":
            int(
                result[
                    "converged"
                ]
            ),

        "selection_success":
            int(
                result[
                    "selection_success"
                ]
            ),

        "selection_verts":
            selection_state.get(
                "verts"
            ),

        "selection_tets":
            selection_state.get(
                "tets"
            ),

        "selection_mm_polys":
            selection_state.get(
                "mm_polys"
            ),

        "finalization_attempted":
            int(
                result[
                    "finalization_attempted"
                ]
            ),

        "finalization_completed":
            int(
                result[
                    "finalization_completed"
                ]
            ),

        "finalization_process_terminated":
            int(
                result[
                    "finalization_process_terminated"
                ]
            ),

        "finalization_crashed":
            int(
                result[
                    "finalization_crashed"
                ]
            ),

        "returncode":
            result[
                "finalization_returncode"
            ],

        "signal":
            result[
                "finalization_signal"
            ],

        "signal_name":
            result[
                "finalization_signal_name"
            ],

        "final_hex":
            final_result.get(
                "hex"
            ),

        "final_total_polys":
            final_result.get(
                "total_polys"
            ),

        "full_hex":
            final_result.get(
                "full_hex"
            ),

        "finalized_state_flag":
            final_state.get(
                "finalized"
            ),

        "final_mm_polys":
            final_state.get(
                "mm_polys"
            ),

        "wall_time":
            wall_time,

        "output_file_count":
            file_count,

        "output_bytes":
            total_bytes,
    }

    print()
    print(
        "SELECTION:"
    )

    print(
        "  steps:",
        row[
            "num_steps"
        ]
    )

    print(
        "  converged:",
        row[
            "selection_converged"
        ]
    )

    print(
        "  success:",
        row[
            "selection_success"
        ]
    )

    print(
        "  tets:",
        row[
            "selection_tets"
        ]
    )

    print()

    print(
        "FINALIZATION:"
    )

    print(
        "  outcome:",
        outcome
    )

    print(
        "  completed:",
        row[
            "finalization_completed"
        ]
    )

    print(
        "  crashed:",
        row[
            "finalization_crashed"
        ]
    )

    print(
        "  returncode:",
        row[
            "returncode"
        ]
    )

    print(
        "  signal:",
        row[
            "signal_name"
        ]
    )

    print(
        "  hex:",
        row[
            "final_hex"
        ]
    )

    print(
        "  total_polys:",
        row[
            "final_total_polys"
        ]
    )

    print(
        "  full_hex:",
        row[
            "full_hex"
        ]
    )

    print(
        "  wall_time:",
        f"{wall_time:.3f}s"
    )

    print(
        "  output files:",
        file_count
    )

    print(
        "  output bytes:",
        total_bytes
    )

    # ============================================================
    # Grounded harness sanity:
    # Cylinder original is already independently verified to
    # finalize to exactly 88/88 full hex.
    # ============================================================

    if (
        case.name
        ==
        "cylinder_original"
    ):
        if not (
            outcome == "FULL_HEX"
            and int(
                row[
                    "final_hex"
                ]
            ) == 88
            and int(
                row[
                    "final_total_polys"
                ]
            ) == 88
            and int(
                row[
                    "full_hex"
                ]
            ) == 1
        ):
            raise RuntimeError(
                "Cylinder FINALIZE sanity regression failed"
            )

    return row


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
        default=Path(
            "/home/yjk/loopycuts_test/"
            "finalization_outcome_audit_v1"
        ),
    )

    args = parser.parse_args()

    root = args.output_dir

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        " "
        in
        str(
            root
        )
    ):
        raise ValueError(
            "Audit output path must not contain spaces"
        )

    if (
        args.case
        ==
        "all"
    ):
        selected = list(
            CASES.values()
        )
    else:
        selected = [
            CASES[
                args.case
            ]
        ]

    #
    # Preserve prior completed rows when cases are run in separate
    # invocations.
    #
    summary_json = (
        root
        /
        "finalization_outcomes.json"
    )

    rows_by_case = {}

    if summary_json.is_file():
        old_rows = json.loads(
            summary_json.read_text(
                encoding="utf-8"
            )
        )

        for row in old_rows:
            rows_by_case[
                row[
                    "case"
                ]
            ] = row

    for case in selected:
        row = run_case(
            case,
            root,
        )

        rows_by_case[
            case.name
        ] = row

        #
        # Save after every completed case.
        #
        ordered_rows = [
            rows_by_case[
                name
            ]
            for name in CASES
            if name in rows_by_case
        ]

        save_rows(
            root,
            ordered_rows,
        )

        print()
        print(
            "Saved completed audit rows to:",
            root
        )

    print()
    print(
        "=" * 78
    )

    print(
        "FINALIZATION OUTCOME AUDIT COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        root
        /
        "finalization_outcomes.csv"
    )


if __name__ == "__main__":
    main()
