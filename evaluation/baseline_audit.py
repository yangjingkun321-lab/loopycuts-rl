from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
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

from finalization.outcome import (
    evaluate_terminal_finalization,
)

from policies.simple import (
    OriginalOrderPolicy,
    RandomPolicy,
)

from rewards.transition_metrics import (
    extract_transition_metrics,
)

from runtime.resource_monitor import (
    ResourceMonitor,
)


EXECUTABLE = Path(
    "/home/yjk/codes/LoopyCuts/"
    "volumetric_cutter/"
    "volumetric_cutter"
)


FROZEN_MANIFEST = (
    PROJECT_ROOT
    /
    "data/manifests/"
    "dataset_split_v2.csv"
)

FROZEN_MANIFEST_JSON = (
    PROJECT_ROOT
    /
    "data/manifests/"
    "dataset_split_v2.json"
)

FROZEN_HASH_FILE = (
    PROJECT_ROOT
    /
    "data/manifests/"
    "dataset_split_v2.sha256"
)


DEFAULT_OUTPUT_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "baseline_audit_v1"
)


ALLOWED_SPLITS = (
    "engineering_calibration",
    "train",
    "dev",
    "test",
)


CSV_FIELDS = [
    "model",
    "split",

    "policy",
    "seed",

    "mesh_file",
    "loop_file",

    "initial_actionable",
    "initial_verts",
    "initial_tets",
    "initial_mm_polys",

    "num_steps",
    "actions",
    "status_counts",

    "committed_steps",
    "reverted_steps",

    "selection_terminal",
    "selection_converged",
    "selection_success",

    "selection_verts",
    "selection_tets",
    "selection_mm_polys",

    "max_selection_verts",
    "max_selection_tets",
    "max_selection_mm_polys",

    "vert_ratio",
    "tet_ratio",

    "sum_cpp_step_time",

    "outcome",

    "finalization_completed",
    "finalization_crashed",

    "return_code",
    "signal_number",
    "signal_name",

    "final_hex",
    "final_total_polys",
    "full_hex",

    "client_initialization_wall_time",
    "selection_wall_time",
    "finalization_wall_time",
    "total_wall_time",

    # Selection-only passive resource measurements.
    "selection_resource_samples",
    "selection_peak_rss_mb",
    "selection_peak_process_swap_mb",
    "selection_min_mem_available_mb",

    # Whole episode: selection + FINALIZE_EVAL.
    "resource_samples",
    "peak_rss_mb",
    "peak_process_swap_mb",
    "min_mem_available_mb",
    "max_system_swap_used_mb",
    "resource_monitor_elapsed_s",

    "finalization_log_tail",
]


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def verify_frozen_manifest():
    """
    Refuse to run baseline evaluation if the frozen Dataset Split V2
    files have changed since their recorded hashes were created.
    """

    for path in (
        FROZEN_MANIFEST,
        FROZEN_MANIFEST_JSON,
        FROZEN_HASH_FILE,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    expected = {}

    for raw_line in (
        FROZEN_HASH_FILE
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    ):
        line = raw_line.strip()

        if not line:
            continue

        parts = line.split(
            maxsplit=1
        )

        if len(parts) != 2:
            raise RuntimeError(
                "Malformed frozen manifest "
                f"hash line: {raw_line!r}"
            )

        digest, relative_path = (
            parts
        )

        expected[
            Path(
                relative_path
            ).name
        ] = digest

    for path in (
        FROZEN_MANIFEST,
        FROZEN_MANIFEST_JSON,
    ):
        expected_digest = (
            expected.get(
                path.name
            )
        )

        if expected_digest is None:
            raise RuntimeError(
                "No recorded SHA256 for "
                f"{path.name}"
            )

        actual_digest = (
            sha256_file(
                path
            )
        )

        if (
            actual_digest
            !=
            expected_digest
        ):
            raise RuntimeError(
                "Frozen dataset manifest hash mismatch.\n"
                f"File:     {path}\n"
                f"Expected: {expected_digest}\n"
                f"Actual:   {actual_digest}"
            )


def load_manifest():
    if not EXECUTABLE.is_file():
        raise FileNotFoundError(
            EXECUTABLE
        )

    rows = []

    with FROZEN_MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(
            f
        )

        required = {
            "model",
            "split",
            "mesh_file",
            "loop_file",
            "parsed_loops",
            "actionable_nonconvex",
        }

        missing = (
            required
            -
            set(
                reader.fieldnames
                or []
            )
        )

        if missing:
            raise RuntimeError(
                "Frozen manifest is missing fields: "
                +
                ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        for row in reader:
            rows.append(
                dict(
                    row
                )
            )

    if len(
        rows
    ) != 74:
        raise RuntimeError(
            "Frozen Dataset Split V2 must "
            f"contain 74 models; got {len(rows)}"
        )

    models = [
        row[
            "model"
        ]
        for row in rows
    ]

    if len(
        models
    ) != len(
        set(
            models
        )
    ):
        raise RuntimeError(
            "Frozen manifest contains "
            "duplicate model names"
        )

    counts = Counter(
        row[
            "split"
        ]
        for row in rows
    )

    expected_counts = {
        "engineering_calibration":
            5,

        "train":
            49,

        "dev":
            10,

        "test":
            10,
    }

    if dict(
        counts
    ) != expected_counts:
        raise RuntimeError(
            "Frozen Dataset Split V2 count regression.\n"
            f"Expected: {expected_counts}\n"
            f"Actual:   {dict(counts)}"
        )

    return rows


def make_policy(
    policy_name: str,
    seed: int | None,
):
    if policy_name == "original":
        if seed is not None:
            raise ValueError(
                "--seed must not be supplied "
                "for policy=original"
            )

        return (
            OriginalOrderPolicy()
        )

    if policy_name == "random":
        if seed is None:
            raise ValueError(
                "--seed is required "
                "for policy=random"
            )

        return RandomPolicy(
            seed=seed
        )

    raise ValueError(
        f"Unknown policy: {policy_name}"
    )


def select_models(
    manifest,
    *,
    split,
    requested_models,
    allow_held_out_test,
):
    if (
        split
        ==
        "test"
        and
        not allow_held_out_test
    ):
        raise RuntimeError(
            "Held-out test is sealed. "
            "Do not evaluate it during development. "
            "Use --allow-held-out-test only in "
            "the final Phase-4 evaluation after "
            "the full experimental protocol "
            "is frozen."
        )

    selected = [
        row
        for row in manifest
        if (
            row[
                "split"
            ]
            ==
            split
        )
    ]

    if requested_models:
        requested = set(
            requested_models
        )

        available = {
            row[
                "model"
            ]
            for row in selected
        }

        missing = (
            requested
            -
            available
        )

        if missing:
            raise RuntimeError(
                "Requested model(s) are not "
                f"in split={split}: "
                +
                ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        selected = [
            row
            for row in selected
            if (
                row[
                    "model"
                ]
                in
                requested
            )
        ]

    selected.sort(
        key=lambda row: (
            int(
                row.get(
                    "v1_complexity_stratum",
                    -1,
                )
            ),
            int(
                row[
                    "actionable_nonconvex"
                ]
            ),
            row[
                "model"
            ],
        )
    )

    if not selected:
        raise RuntimeError(
            "No models selected"
        )

    return selected


def validate_model_files(
    row,
):
    mesh = Path(
        row[
            "mesh_file"
        ]
    )

    loops = Path(
        row[
            "loop_file"
        ]
    )

    if not mesh.is_file():
        raise FileNotFoundError(
            f"{row['model']}: mesh not found: "
            f"{mesh}"
        )

    if not loops.is_file():
        raise FileNotFoundError(
            f"{row['model']}: loop file not found: "
            f"{loops}"
        )

    return (
        mesh,
        loops,
    )


def aggregate_selection_metrics(
    *,
    initial_state,
    trajectory,
):
    previous_state = dict(
        initial_state
    )

    committed_steps = 0
    reverted_steps = 0

    sum_cpp_step_time = 0.0

    max_verts = int(
        initial_state[
            "verts"
        ]
    )

    max_tets = int(
        initial_state[
            "tets"
        ]
    )

    max_mm_polys = int(
        initial_state[
            "mm_polys"
        ]
    )

    status_counts = Counter()

    for item in trajectory:
        step_result = dict(
            item[
                "step_result"
            ]
        )

        state_after = dict(
            item[
                "state"
            ]
        )

        metrics = (
            extract_transition_metrics(
                state_before=previous_state,
                step_result=step_result,
                state_after=state_after,
            )
        )

        committed_steps += int(
            metrics.committed
        )

        reverted_steps += int(
            metrics.reverted
        )

        sum_cpp_step_time += float(
            metrics.step_time
        )

        status_counts[
            str(
                step_result.get(
                    "status",
                    "UNKNOWN",
                )
            )
        ] += 1

        max_verts = max(
            max_verts,
            int(
                state_after[
                    "verts"
                ]
            ),
        )

        max_tets = max(
            max_tets,
            int(
                state_after[
                    "tets"
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

        previous_state = (
            state_after
        )

    return {
        "committed_steps":
            committed_steps,

        "reverted_steps":
            reverted_steps,

        "sum_cpp_step_time":
            sum_cpp_step_time,

        "max_selection_verts":
            max_verts,

        "max_selection_tets":
            max_tets,

        "max_selection_mm_polys":
            max_mm_polys,

        "status_counts":
            dict(
                sorted(
                    status_counts.items()
                )
            ),
    }


def make_output_dir(
    *,
    output_root,
    split,
    policy,
    seed,
):
    if policy == "original":
        run_name = "original"

    else:
        run_name = (
            f"random_seed_{seed:04d}"
        )

    return (
        output_root
        /
        split
        /
        run_name
    )


def csv_safe_row(
    row,
):
    result = {}

    for field in (
        CSV_FIELDS
    ):
        value = row.get(
            field
        )

        if isinstance(
            value,
            (
                list,
                tuple,
                dict,
            ),
        ):
            result[
                field
            ] = json.dumps(
                value,
                separators=(
                    ",",
                    ":",
                ),
            )

        elif value is None:
            result[
                field
            ] = ""

        else:
            result[
                field
            ] = value

    return result


def save_results(
    output_dir: Path,
    rows,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_dir
        /
        "results.json"
    )

    csv_path = (
        output_dir
        /
        "results.csv"
    )

    json_path.write_text(
        json.dumps(
            rows,
            indent=2,
        ),
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                csv_safe_row(
                    row
                )
            )


def load_existing_results(
    output_dir: Path,
):
    json_path = (
        output_dir
        /
        "results.json"
    )

    if not json_path.is_file():
        return []

    data = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        list,
    ):
        raise RuntimeError(
            f"Existing results are not a list: "
            f"{json_path}"
        )

    return data


def run_model(
    *,
    row,
    policy_name,
    seed,
    max_steps,
    echo_logs,
):
    model = row[
        "model"
    ]

    mesh, loops = (
        validate_model_files(
            row
        )
    )

    policy = make_policy(
        policy_name,
        seed,
    )

    print()
    print(
        "=" * 88
    )

    print(
        "MODEL:",
        model
    )

    print(
        "SPLIT:",
        row[
            "split"
        ]
    )

    print(
        "POLICY:",
        policy_name
    )

    print(
        "SEED:",
        (
            seed
            if seed is not None
            else "-"
        )
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
        "=" * 88
    )

    total_start = (
        time.perf_counter()
    )

    client_start = (
        time.perf_counter()
    )

    with LoopyCutsClient(
        executable=EXECUTABLE,
        mesh_file=mesh,
        loop_file=loops,
        echo_logs=echo_logs,
    ) as client:

        client_initialization_wall_time = (
            time.perf_counter()
            -
            client_start
        )

        #
        # Passive resource monitoring only.
        #
        # IMPORTANT:
        #     ResourceMonitor NEVER terminates or modifies
        #     the C++ process. It has no effect on policy,
        #     legality, termination, finalization, or reward.
        #
        resource_monitor = ResourceMonitor(
            pid=client.process.pid,
            sample_interval_s=1.0,
        )

        resource_monitor.start()

        resource_stats = None

        if client.state is None:
            raise RuntimeError(
                f"{model}: client initialized "
                "without state"
            )

        initial_state = dict(
            client.state
        )

        initial_actions = list(
            client.actions
        )

        initial_actionable = len(
            initial_actions
        )

        if (
            initial_actionable
            !=
            int(
                row[
                    "actionable_nonconvex"
                ]
            )
        ):
            raise RuntimeError(
                f"{model}: manifest/action-space "
                "regression. "
                f"Manifest actionable="
                f"{row['actionable_nonconvex']}, "
                f"C++ ACTIONS="
                f"{initial_actionable}"
            )

        selection_start = (
            time.perf_counter()
        )

        selection = run_episode(
            client=client,
            policy=policy,
            finalize=False,
            max_steps=max_steps,
        )

        selection_wall_time = (
            time.perf_counter()
            -
            selection_start
        )

        selection_state = dict(
            selection[
                "selection_state"
            ]
        )

        if not int(
            selection_state[
                "terminal"
            ]
        ):
            raise RuntimeError(
                f"{model}: run_episode returned "
                "before selection terminal"
            )

        aggregate = (
            aggregate_selection_metrics(
                initial_state=initial_state,
                trajectory=selection[
                    "trajectory"
                ],
            )
        )

        actions = [
            int(
                item[
                    "action"
                ]
            )
            for item
            in selection[
                "trajectory"
            ]
        ]

        #
        # Passive snapshot at the exact boundary between
        # Stage-2 selection and FINALIZE_EVAL.
        #
        # This does not stop or modify the C++ process.
        #
        selection_resource_stats = (
            resource_monitor.snapshot()
        )

        finalization_start = (
            time.perf_counter()
        )

        outcome = (
            evaluate_terminal_finalization(
                client
            )
        )

        finalization_wall_time = (
            time.perf_counter()
            -
            finalization_start
        )

        outcome_dict = (
            outcome.to_dict()
        )

        resource_stats = (
            resource_monitor.stop()
        )

    total_wall_time = (
        time.perf_counter()
        -
        total_start
    )

    initial_verts = int(
        initial_state[
            "verts"
        ]
    )

    initial_tets = int(
        initial_state[
            "tets"
        ]
    )

    selection_verts = int(
        selection_state[
            "verts"
        ]
    )

    selection_tets = int(
        selection_state[
            "tets"
        ]
    )

    if (
        initial_verts <= 0
        or
        initial_tets <= 0
    ):
        raise RuntimeError(
            f"{model}: invalid initial "
            "mesh size"
        )

    vert_ratio = (
        selection_verts
        /
        initial_verts
    )

    tet_ratio = (
        selection_tets
        /
        initial_tets
    )

    for value, name in (
        (
            vert_ratio,
            "vert_ratio",
        ),
        (
            tet_ratio,
            "tet_ratio",
        ),
    ):
        if not math.isfinite(
            value
        ):
            raise RuntimeError(
                f"{model}: non-finite "
                f"{name}"
            )

    result = {
        "model":
            model,

        "split":
            row[
                "split"
            ],

        "policy":
            policy_name,

        "seed":
            seed,

        "mesh_file":
            str(
                mesh
            ),

        "loop_file":
            str(
                loops
            ),

        "initial_actionable":
            initial_actionable,

        "initial_verts":
            initial_verts,

        "initial_tets":
            initial_tets,

        "initial_mm_polys":
            int(
                initial_state[
                    "mm_polys"
                ]
            ),

        "num_steps":
            int(
                selection[
                    "num_steps"
                ]
            ),

        "actions":
            actions,

        "status_counts":
            aggregate[
                "status_counts"
            ],

        "committed_steps":
            int(
                aggregate[
                    "committed_steps"
                ]
            ),

        "reverted_steps":
            int(
                aggregate[
                    "reverted_steps"
                ]
            ),

        "selection_terminal":
            int(
                selection_state[
                    "terminal"
                ]
            ),

        "selection_converged":
            int(
                selection_state[
                    "converged"
                ]
            ),

        "selection_success":
            int(
                selection_state[
                    "selection_success"
                ]
            ),

        "selection_verts":
            selection_verts,

        "selection_tets":
            selection_tets,

        "selection_mm_polys":
            int(
                selection_state[
                    "mm_polys"
                ]
            ),

        "max_selection_verts":
            int(
                aggregate[
                    "max_selection_verts"
                ]
            ),

        "max_selection_tets":
            int(
                aggregate[
                    "max_selection_tets"
                ]
            ),

        "max_selection_mm_polys":
            int(
                aggregate[
                    "max_selection_mm_polys"
                ]
            ),

        "vert_ratio":
            float(
                vert_ratio
            ),

        "tet_ratio":
            float(
                tet_ratio
            ),

        "sum_cpp_step_time":
            float(
                aggregate[
                    "sum_cpp_step_time"
                ]
            ),

        "outcome":
            outcome_dict[
                "outcome"
            ],

        "finalization_completed":
            bool(
                outcome_dict[
                    "completed"
                ]
            ),

        "finalization_crashed":
            bool(
                outcome_dict[
                    "crashed"
                ]
            ),

        "return_code":
            outcome_dict[
                "return_code"
            ],

        "signal_number":
            outcome_dict[
                "signal_number"
            ],

        "signal_name":
            outcome_dict[
                "signal_name"
            ],

        "final_hex":
            outcome_dict[
                "final_hex"
            ],

        "final_total_polys":
            outcome_dict[
                "final_total_polys"
            ],

        "full_hex":
            outcome_dict[
                "full_hex"
            ],

        "client_initialization_wall_time":
            float(
                client_initialization_wall_time
            ),

        "selection_wall_time":
            float(
                selection_wall_time
            ),

        "finalization_wall_time":
            float(
                finalization_wall_time
            ),

        "total_wall_time":
            float(
                total_wall_time
            ),

        "selection_resource_samples":
            int(
                selection_resource_stats.samples
            ),

        "selection_peak_rss_mb":
            float(
                selection_resource_stats.peak_rss_mb
            ),

        "selection_peak_process_swap_mb":
            float(
                selection_resource_stats.peak_process_swap_mb
            ),

        "selection_min_mem_available_mb":
            float(
                selection_resource_stats.min_mem_available_mb
            ),

        "resource_samples":
            int(
                resource_stats.samples
            ),

        "peak_rss_mb":
            float(
                resource_stats.peak_rss_mb
            ),

        "peak_process_swap_mb":
            float(
                resource_stats.peak_process_swap_mb
            ),

        "min_mem_available_mb":
            float(
                resource_stats.min_mem_available_mb
            ),

        "max_system_swap_used_mb":
            float(
                resource_stats.max_system_swap_used_mb
            ),

        "resource_monitor_elapsed_s":
            float(
                resource_stats.monitor_elapsed_s
            ),

        "finalization_log_tail":
            list(
                outcome_dict[
                    "log_tail"
                ]
            ),
    }

    print()
    print(
        "RESULT:"
    )

    print(
        "  steps:",
        result[
            "num_steps"
        ]
    )

    print(
        "  committed:",
        result[
            "committed_steps"
        ]
    )

    print(
        "  reverted:",
        result[
            "reverted_steps"
        ]
    )

    print(
        "  selection_success:",
        result[
            "selection_success"
        ]
    )

    print(
        "  tet_ratio:",
        f"{result['tet_ratio']:.6f}"
    )

    print(
        "  outcome:",
        result[
            "outcome"
        ]
    )

    print(
        "  final_hex:",
        result[
            "final_hex"
        ]
    )

    print(
        "  final_total_polys:",
        result[
            "final_total_polys"
        ]
    )

    print(
        "  selection_wall_time:",
        f"{selection_wall_time:.3f}s"
    )

    print(
        "  finalization_wall_time:",
        f"{finalization_wall_time:.3f}s"
    )

    print(
        "  total_wall_time:",
        f"{total_wall_time:.3f}s"
    )

    print(
        "  selection_peak_rss_mb:",
        f"{result['selection_peak_rss_mb']:.1f}"
    )

    print(
        "  selection_peak_process_swap_mb:",
        f"{result['selection_peak_process_swap_mb']:.1f}"
    )

    print(
        "  selection_min_mem_available_mb:",
        f"{result['selection_min_mem_available_mb']:.1f}"
    )

    print(
        "  peak_rss_mb:",
        f"{result['peak_rss_mb']:.1f}"
    )

    print(
        "  peak_process_swap_mb:",
        f"{result['peak_process_swap_mb']:.1f}"
    )

    print(
        "  min_mem_available_mb:",
        f"{result['min_mem_available_mb']:.1f}"
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=ALLOWED_SPLITS,
        required=True,
    )

    parser.add_argument(
        "--policy",
        choices=[
            "original",
            "random",
        ],
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--allow-held-out-test",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--echo-logs",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.resume
        and
        args.overwrite
    ):
        raise ValueError(
            "--resume and --overwrite "
            "are mutually exclusive"
        )

    if (
        args.policy
        ==
        "original"
        and
        args.seed is not None
    ):
        raise ValueError(
            "--seed is only valid for "
            "policy=random"
        )

    if (
        args.policy
        ==
        "random"
        and
        args.seed is None
    ):
        raise ValueError(
            "--seed is required for "
            "policy=random"
        )

    verify_frozen_manifest()

    manifest = load_manifest()

    selected = select_models(
        manifest,
        split=args.split,
        requested_models=args.models,
        allow_held_out_test=(
            args.allow_held_out_test
        ),
    )

    output_dir = make_output_dir(
        output_root=args.output_root,
        split=args.split,
        policy=args.policy,
        seed=args.seed,
    )

    results_json = (
        output_dir
        /
        "results.json"
    )

    if (
        results_json.exists()
        and
        not args.resume
        and
        not args.overwrite
    ):
        raise RuntimeError(
            "Output already exists:\n"
            f"{results_json}\n"
            "Use --resume to continue or "
            "--overwrite to replace it."
        )

    if args.overwrite:
        rows = []

    elif args.resume:
        rows = load_existing_results(
            output_dir
        )

    else:
        rows = []

    completed_models = {
        row[
            "model"
        ]
        for row in rows
    }

    print()
    print(
        "=" * 88
    )

    print(
        "BASELINE AUDIT V1"
    )

    print(
        "=" * 88
    )

    print(
        "Frozen manifest:",
        FROZEN_MANIFEST
    )

    print(
        "Split:",
        args.split
    )

    print(
        "Policy:",
        args.policy
    )

    print(
        "Seed:",
        (
            args.seed
            if args.seed is not None
            else "-"
        )
    )

    print(
        "Selected models:",
        len(
            selected
        )
    )

    print(
        "Already completed:",
        len(
            completed_models
        )
    )

    print(
        "Output:",
        output_dir
    )

    print(
        "=" * 88
    )

    for row in selected:
        model = row[
            "model"
        ]

        if (
            model
            in
            completed_models
        ):
            print(
                f"SKIP completed: {model}"
            )

            continue

        result = run_model(
            row=row,
            policy_name=args.policy,
            seed=args.seed,
            max_steps=args.max_steps,
            echo_logs=args.echo_logs,
        )

        rows.append(
            result
        )

        #
        # Save immediately after every expensive model.
        #
        save_results(
            output_dir,
            rows,
        )

        completed_models.add(
            model
        )

        print()
        print(
            "Saved cumulative results:",
            output_dir
        )

    print()
    print(
        "=" * 88
    )

    print(
        "BASELINE AUDIT COMPLETE"
    )

    print(
        "=" * 88
    )

    print(
        "Completed rows:",
        len(
            rows
        )
    )

    print(
        "CSV:",
        output_dir
        /
        "results.csv"
    )

    print(
        "JSON:",
        output_dir
        /
        "results.json"
    )


if __name__ == "__main__":
    main()
