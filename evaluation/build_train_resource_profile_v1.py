from __future__ import annotations

import csv
import json
import sys
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


MANIFEST = (
    PROJECT_ROOT
    /
    "data/manifests/dataset_split_v2.csv"
)

EXPERIMENT_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "train_resource_feasibility_v1"
)

ORIGINAL_JSON = (
    EXPERIMENT_ROOT
    /
    "train/original/results.json"
)

RANDOM_JSON = (
    EXPERIMENT_ROOT
    /
    "train/random_seed_0000/results.json"
)

PILOT_ROOT = (
    EXPERIMENT_ROOT
    /
    "resource_stress_pilots"
)

OUTPUT = (
    PROJECT_ROOT
    /
    "data/manifests/"
    "train_resource_profile_v1.csv"
)


FIELDS = [
    "model",
    "split",

    "header_loops",
    "actionable_nonconvex",
    "concave",
    "regular",
    "convex",
    "v1_complexity_stratum",

    # ----------------------------------------------------------
    # Original complete baseline facts.
    # ----------------------------------------------------------

    "original_profile_status",

    "original_initial_actionable",
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
    # Random seed0 complete baseline facts, where available.
    # ----------------------------------------------------------

    "random_seed0_complete_status",

    "random_seed0_num_steps",
    "random_seed0_committed_steps",
    "random_seed0_reverted_steps",

    "random_seed0_selection_success",
    "random_seed0_tet_ratio",

    "random_seed0_outcome",
    "random_seed0_final_hex",
    "random_seed0_final_total_polys",

    "random_seed0_selection_peak_rss_mb",
    "random_seed0_selection_peak_process_swap_mb",
    "random_seed0_selection_min_mem_available_mb",

    "random_seed0_peak_rss_mb",
    "random_seed0_peak_process_swap_mb",
    "random_seed0_min_mem_available_mb",

    # ----------------------------------------------------------
    # Selection resource pilot facts.
    # ----------------------------------------------------------

    "random_seed0_pilot_status",
    "random_seed0_pilot_stop_reason",

    "random_seed0_pilot_terminal",
    "random_seed0_pilot_completed_steps",
    "random_seed0_pilot_tet_ratio",

    "random_seed0_pilot_peak_rss_mb",
    "random_seed0_pilot_peak_process_swap_mb",
    "random_seed0_pilot_min_mem_available_mb",

    # ----------------------------------------------------------
    # Incomplete terminal finalization incidents.
    # ----------------------------------------------------------

    "random_seed0_finalization_status",

    # ----------------------------------------------------------
    # Decisions deliberately NOT frozen yet.
    # ----------------------------------------------------------

    "resource_role",
    "exploration_risk",
]


def load_json_list(
    path: Path,
):
    if not path.is_file():
        return []

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


def load_json_object(
    path: Path,
):
    if not path.is_file():
        return None

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            f"Expected JSON object: {path}"
        )

    return data


def assign(
    out,
    prefix,
    source,
    mapping,
):
    for dst, src in mapping.items():
        value = source.get(
            src,
            ""
        )

        out[
            f"{prefix}{dst}"
        ] = value


def main():
    with MANIFEST.open(
        newline="",
        encoding="utf-8",
    ) as f:
        manifest = [
            row
            for row in csv.DictReader(f)
            if row["split"] == "train"
        ]

    if len(manifest) != 49:
        raise RuntimeError(
            "Expected 49 train models, "
            f"found {len(manifest)}"
        )

    original = {
        row["model"]: row
        for row in load_json_list(
            ORIGINAL_JSON
        )
    }

    random_complete = {
        row["model"]: row
        for row in load_json_list(
            RANDOM_JSON
        )
    }

    rows = []

    for manifest_row in manifest:
        model = manifest_row["model"]

        out = {
            field: ""
            for field in FIELDS
        }

        for field in [
            "model",
            "split",
            "header_loops",
            "actionable_nonconvex",
            "concave",
            "regular",
            "convex",
            "v1_complexity_stratum",
        ]:
            out[field] = (
                manifest_row.get(
                    field,
                    ""
                )
            )

        # ======================================================
        # Original complete baseline.
        # ======================================================

        if model in original:
            source = original[model]

            out[
                "original_profile_status"
            ] = "COMPLETE"

            assign(
                out,
                "original_",
                source,
                {
                    "initial_actionable":
                        "initial_actionable",

                    "num_steps":
                        "num_steps",

                    "committed_steps":
                        "committed_steps",

                    "reverted_steps":
                        "reverted_steps",

                    "selection_success":
                        "selection_success",

                    "tet_ratio":
                        "tet_ratio",

                    "outcome":
                        "outcome",

                    "final_hex":
                        "final_hex",

                    "final_total_polys":
                        "final_total_polys",

                    "selection_peak_rss_mb":
                        "selection_peak_rss_mb",

                    "selection_peak_process_swap_mb":
                        "selection_peak_process_swap_mb",

                    "selection_min_mem_available_mb":
                        "selection_min_mem_available_mb",

                    "peak_rss_mb":
                        "peak_rss_mb",

                    "peak_process_swap_mb":
                        "peak_process_swap_mb",

                    "min_mem_available_mb":
                        "min_mem_available_mb",
                },
            )

        else:
            out[
                "original_profile_status"
            ] = "UNPROFILED"

        # ======================================================
        # Random complete baseline.
        # ======================================================

        if model in random_complete:
            source = (
                random_complete[
                    model
                ]
            )

            out[
                "random_seed0_complete_status"
            ] = "COMPLETE"

            assign(
                out,
                "random_seed0_",
                source,
                {
                    "num_steps":
                        "num_steps",

                    "committed_steps":
                        "committed_steps",

                    "reverted_steps":
                        "reverted_steps",

                    "selection_success":
                        "selection_success",

                    "tet_ratio":
                        "tet_ratio",

                    "outcome":
                        "outcome",

                    "final_hex":
                        "final_hex",

                    "final_total_polys":
                        "final_total_polys",

                    "selection_peak_rss_mb":
                        "selection_peak_rss_mb",

                    "selection_peak_process_swap_mb":
                        "selection_peak_process_swap_mb",

                    "selection_min_mem_available_mb":
                        "selection_min_mem_available_mb",

                    "peak_rss_mb":
                        "peak_rss_mb",

                    "peak_process_swap_mb":
                        "peak_process_swap_mb",

                    "min_mem_available_mb":
                        "min_mem_available_mb",
                },
            )

        else:
            out[
                "random_seed0_complete_status"
            ] = "NOT_RUN_OR_INCOMPLETE"

        # ======================================================
        # Selection pilot.
        #
        # Both generated pilot records and the manually preserved
        # motor_tail record are supported.
        # ======================================================

        pilot_file = (
            PILOT_ROOT
            /
            f"{model}_random_seed0.json"
        )

        pilot = load_json_object(
            pilot_file
        )

        if pilot is not None:
            out[
                "random_seed0_pilot_status"
            ] = pilot.get(
                "status",
                "",
            )

            out[
                "random_seed0_pilot_stop_reason"
            ] = pilot.get(
                "stop_reason",
                "",
            )

            out[
                "random_seed0_pilot_terminal"
            ] = pilot.get(
                "terminal",
                "",
            )

            out[
                "random_seed0_pilot_completed_steps"
            ] = pilot.get(
                "completed_steps",
                pilot.get(
                    "last_completed_step",
                    "",
                ),
            )

            out[
                "random_seed0_pilot_tet_ratio"
            ] = pilot.get(
                "partial_tet_ratio",
                "",
            )

            out[
                "random_seed0_pilot_peak_rss_mb"
            ] = pilot.get(
                "peak_rss_mb",
                pilot.get(
                    "selection_peak_rss_mb",
                    "",
                ),
            )

            out[
                "random_seed0_pilot_peak_process_swap_mb"
            ] = pilot.get(
                "peak_process_swap_mb",
                pilot.get(
                    "selection_peak_process_swap_mb",
                    "",
                ),
            )

            out[
                "random_seed0_pilot_min_mem_available_mb"
            ] = pilot.get(
                "min_mem_available_mb",
                pilot.get(
                    "selection_min_mem_available_mb",
                    "",
                ),
            )

        # ======================================================
        # Finalization incident.
        # ======================================================

        incident_file = (
            PILOT_ROOT
            /
            (
                f"{model}_random_seed0_"
                "finalization_interrupt.json"
            )
        )

        incident = load_json_object(
            incident_file
        )

        if incident is not None:
            out[
                "random_seed0_finalization_status"
            ] = incident.get(
                "status",
                "",
            )

        # ======================================================
        # Deliberately unresolved until eligibility freeze.
        # ======================================================

        out["resource_role"] = (
            "UNFROZEN"
        )

        out["exploration_risk"] = (
            "UNFROZEN"
        )

        rows.append(
            out
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
            fieldnames=FIELDS,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    original_complete = sum(
        row[
            "original_profile_status"
        ]
        ==
        "COMPLETE"
        for row in rows
    )

    random_complete_n = sum(
        row[
            "random_seed0_complete_status"
        ]
        ==
        "COMPLETE"
        for row in rows
    )

    random_pilot_n = sum(
        bool(
            row[
                "random_seed0_pilot_status"
            ]
        )
        for row in rows
    )

    print("OUTPUT:", OUTPUT)
    print("train rows:", len(rows))
    print(
        "original complete:",
        original_complete,
    )
    print(
        "original unprofiled:",
        len(rows)
        -
        original_complete,
    )
    print(
        "random complete:",
        random_complete_n,
    )
    print(
        "random pilots:",
        random_pilot_n,
    )


if __name__ == "__main__":
    main()
