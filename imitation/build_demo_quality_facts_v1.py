from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


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


from imitation.demo_v1 import (
    load_episode,
    sha256_file,
)


DATASET_MANIFEST = (
    PROJECT_ROOT
    /
    "data/manifests/"
    "dataset_split_v2.csv"
)

RESOURCE_PROFILE = (
    PROJECT_ROOT
    /
    "data/manifests/"
    "train_resource_profile_v1.csv"
)

BASELINE_JSON = Path(
    "/home/yjk/loopycuts_test/"
    "train_resource_feasibility_v1/"
    "train/original/results.json"
)

RAW_DEMO_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "demonstrations_v1/raw/train"
)

OUTPUT = (
    PROJECT_ROOT
    /
    "data/manifests/"
    "demo_quality_facts_v1.csv"
)


FIELDS = [
    "model",
    "split",

    "actionable_nonconvex",
    "v1_complexity_stratum",

    # ----------------------------------------------------------
    # Raw Demonstration V1 facts.
    # ----------------------------------------------------------

    "raw_demo_status",
    "demo_schema_version",
    "demo_observation_version",
    "demo_reward_version",
    "demo_teacher_version",

    "demo_num_steps",
    "demo_total_return",
    "demo_outcome",
    "demo_quality_class",

    "demo_npz_file",
    "demo_metadata_file",
    "demo_npz_sha256",

    # ----------------------------------------------------------
    # Independent Original complete-baseline facts.
    # ----------------------------------------------------------

    "original_profile_status",

    "original_num_steps",
    "original_committed_steps",
    "original_reverted_steps",

    "original_selection_success",
    "original_tet_ratio",

    "original_outcome",
    "original_final_hex",
    "original_final_total_polys",

    "original_selection_peak_rss_mb",
    "original_selection_peak_process_swap_mb",
    "original_selection_min_mem_available_mb",

    "original_peak_rss_mb",
    "original_peak_process_swap_mb",
    "original_min_mem_available_mb",

    # ----------------------------------------------------------
    # Original selection-only pilot facts.
    # ----------------------------------------------------------

    "original_selection_pilot_status",
    "original_selection_pilot_stop_reason",
    "original_selection_pilot_terminal",
    "original_selection_pilot_completed_steps",
    "original_selection_pilot_tet_ratio",
    "original_selection_pilot_peak_rss_mb",
    "original_selection_pilot_peak_process_swap_mb",
    "original_selection_pilot_min_mem_available_mb",

    # ----------------------------------------------------------
    # Cross-source reproducibility.
    # ----------------------------------------------------------

    "demo_baseline_trajectory_match",

    # ----------------------------------------------------------
    # Deliberately NOT frozen yet.
    # ----------------------------------------------------------

    "quality_role",
    "quality_reason",
]


def load_csv(
    path: Path,
):
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return [
            dict(row)
            for row in csv.DictReader(f)
        ]


def load_json_list(
    path: Path,
):
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        list,
    ):
        raise RuntimeError(
            f"Expected JSON list: {path}"
        )

    return data


def copy_profile_field(
    out,
    profile,
    field,
):
    out[field] = (
        profile.get(
            field,
            ""
        )
    )


def main():
    manifest_rows = [
        row
        for row in load_csv(
            DATASET_MANIFEST
        )
        if row[
            "split"
        ] == "train"
    ]

    if len(
        manifest_rows
    ) != 49:
        raise RuntimeError(
            "Expected 49 train models; "
            f"got {len(manifest_rows)}"
        )

    profile_rows = (
        load_csv(
            RESOURCE_PROFILE
        )
    )

    profile_by_model = {
        row[
            "model"
        ]:
            row
        for row
        in profile_rows
    }

    if len(
        profile_by_model
    ) != 49:
        raise RuntimeError(
            "Train resource profile must "
            "contain exactly 49 models"
        )

    baseline_rows = (
        load_json_list(
            BASELINE_JSON
        )
    )

    baseline_by_model = {
        row[
            "model"
        ]:
            row
        for row
        in baseline_rows
    }

    rows = []

    raw_demo_count = 0
    trajectory_match_count = 0

    for manifest in manifest_rows:
        model = manifest[
            "model"
        ]

        if (
            model
            not in
            profile_by_model
        ):
            raise RuntimeError(
                "Resource profile missing "
                f"model={model}"
            )

        profile = (
            profile_by_model[
                model
            ]
        )

        out = {
            field:
                ""
            for field
            in FIELDS
        }

        out[
            "model"
        ] = model

        out[
            "split"
        ] = "train"

        out[
            "actionable_nonconvex"
        ] = manifest.get(
            "actionable_nonconvex",
            "",
        )

        out[
            "v1_complexity_stratum"
        ] = manifest.get(
            "v1_complexity_stratum",
            "",
        )

        # ======================================================
        # Resource-profile facts.
        # ======================================================

        profile_fields = [
            "original_profile_status",

            "original_num_steps",
            "original_committed_steps",
            "original_reverted_steps",

            "original_selection_success",
            "original_tet_ratio",

            "original_outcome",
            "original_final_hex",
            "original_final_total_polys",

            "original_selection_peak_rss_mb",
            "original_selection_peak_process_swap_mb",
            "original_selection_min_mem_available_mb",

            "original_peak_rss_mb",
            "original_peak_process_swap_mb",
            "original_min_mem_available_mb",

            "original_selection_pilot_status",
            "original_selection_pilot_stop_reason",
            "original_selection_pilot_terminal",
            "original_selection_pilot_completed_steps",
            "original_selection_pilot_tet_ratio",
            "original_selection_pilot_peak_rss_mb",
            "original_selection_pilot_peak_process_swap_mb",
            "original_selection_pilot_min_mem_available_mb",
        ]

        for field in profile_fields:
            copy_profile_field(
                out,
                profile,
                field,
            )

        # ======================================================
        # Raw Demonstration V1.
        # ======================================================

        demo_dir = (
            RAW_DEMO_ROOT
            /
            model
        )

        stem = (
            f"{model}_"
            "original_demo_v1"
        )

        npz_path = (
            demo_dir
            /
            f"{stem}.npz"
        )

        metadata_path = (
            demo_dir
            /
            f"{stem}.json"
        )

        if (
            npz_path.exists()
            !=
            metadata_path.exists()
        ):
            raise RuntimeError(
                f"{model}: partial raw "
                "demonstration artifact"
            )

        if not npz_path.is_file():
            out[
                "raw_demo_status"
            ] = "NOT_COLLECTED"

            out[
                "quality_role"
            ] = "UNFROZEN"

            out[
                "quality_reason"
            ] = "UNFROZEN"

            rows.append(
                out
            )

            continue

        raw_demo_count += 1

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            metadata.get(
                "model"
            )
            !=
            model
        ):
            raise RuntimeError(
                f"{model}: demo metadata "
                "model mismatch"
            )

        if (
            metadata.get(
                "split"
            )
            !=
            "train"
        ):
            raise RuntimeError(
                f"{model}: demo split "
                "must be train"
            )

        expected_sha = (
            metadata.get(
                "npz_sha256"
            )
        )

        actual_sha = (
            sha256_file(
                npz_path
            )
        )

        if (
            expected_sha
            !=
            actual_sha
        ):
            raise RuntimeError(
                f"{model}: raw demo "
                "checksum mismatch"
            )

        data = load_episode(
            npz_path
        )

        demo_steps = int(
            data[
                "actions"
            ].shape[
                0
            ]
        )

        if (
            demo_steps
            !=
            int(
                metadata[
                    "num_steps"
                ]
            )
        ):
            raise RuntimeError(
                f"{model}: demo metadata/"
                "NPZ step mismatch"
            )

        out[
            "raw_demo_status"
        ] = "COLLECTED"

        out[
            "demo_schema_version"
        ] = metadata[
            "schema_version"
        ]

        out[
            "demo_observation_version"
        ] = metadata[
            "observation_version"
        ]

        out[
            "demo_reward_version"
        ] = metadata[
            "reward_version"
        ]

        out[
            "demo_teacher_version"
        ] = metadata[
            "teacher_version"
        ]

        out[
            "demo_num_steps"
        ] = demo_steps

        out[
            "demo_total_return"
        ] = metadata[
            "total_return"
        ]

        out[
            "demo_outcome"
        ] = metadata[
            "finalization_outcome"
        ][
            "outcome"
        ]

        out[
            "demo_quality_class"
        ] = metadata[
            "quality_class"
        ]

        out[
            "demo_npz_file"
        ] = str(
            npz_path
        )

        out[
            "demo_metadata_file"
        ] = str(
            metadata_path
        )

        out[
            "demo_npz_sha256"
        ] = actual_sha

        # ======================================================
        # Every formal raw demo currently requires an independent
        # complete Original baseline.
        # ======================================================

        if (
            model
            not in
            baseline_by_model
        ):
            raise RuntimeError(
                f"{model}: raw demo exists "
                "without complete Original "
                "baseline"
            )

        baseline = (
            baseline_by_model[
                model
            ]
        )

        if (
            profile[
                "original_profile_status"
            ]
            !=
            "COMPLETE"
        ):
            raise RuntimeError(
                f"{model}: raw demo baseline "
                "is not COMPLETE in resource "
                "profile"
            )

        demo_actions = (
            data[
                "actions"
            ]
            .astype(
                np.int64
            )
            .tolist()
        )

        baseline_actions = [
            int(action)
            for action
            in baseline[
                "actions"
            ]
        ]

        if (
            demo_actions
            !=
            baseline_actions
        ):
            raise RuntimeError(
                f"{model}: raw demo action "
                "trajectory differs from "
                "Original baseline"
            )

        if (
            demo_steps
            !=
            int(
                baseline[
                    "num_steps"
                ]
            )
        ):
            raise RuntimeError(
                f"{model}: raw demo/baseline "
                "step mismatch"
            )

        demo_outcome = (
            metadata[
                "finalization_outcome"
            ][
                "outcome"
            ]
        )

        if (
            demo_outcome
            !=
            baseline[
                "outcome"
            ]
        ):
            raise RuntimeError(
                f"{model}: raw demo/baseline "
                "outcome mismatch"
            )

        out[
            "demo_baseline_trajectory_match"
        ] = "MATCH"

        trajectory_match_count += 1

        # ======================================================
        # No quality thresholds are frozen yet.
        # ======================================================

        out[
            "quality_role"
        ] = "UNFROZEN"

        out[
            "quality_reason"
        ] = "UNFROZEN"

        rows.append(
            out
        )

    if len(
        rows
    ) != 49:
        raise RuntimeError(
            "Demo quality facts must "
            "contain 49 train models"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=
                FIELDS,
            lineterminator=
                "\n",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    complete_baselines = sum(
        row[
            "original_profile_status"
        ]
        ==
        "COMPLETE"
        for row
        in rows
    )

    selection_pilots = sum(
        bool(
            row[
                "original_selection_pilot_status"
            ]
        )
        for row
        in rows
    )

    print(
        "OUTPUT:",
        OUTPUT,
    )

    print(
        "train rows:",
        len(
            rows
        ),
    )

    print(
        "raw demos:",
        raw_demo_count,
    )

    print(
        "complete baselines:",
        complete_baselines,
    )

    print(
        "selection pilots:",
        selection_pilots,
    )

    print(
        "demo/baseline matches:",
        trajectory_match_count,
    )

    print(
        "quality decisions frozen:",
        0,
    )


if __name__ == "__main__":
    main()
