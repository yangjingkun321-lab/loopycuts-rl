from __future__ import annotations

import csv
import hashlib
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


QUALITY_VERSION = (
    "demo_quality_v1"
)


FACTS = (
    PROJECT_ROOT
    /
    "data/manifests/"
    "demo_quality_facts_v1.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    /
    "data/manifests/"
    "demo_quality_v1.csv"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    /
    "data/manifests/"
    "demo_quality_v1.json"
)


ROLE_BC_CORE = (
    "BC_CORE"
)

ROLE_RL_AUXILIARY = (
    "RL_AUXILIARY"
)

ROLE_EXCLUDE = (
    "EXCLUDE"
)

ROLE_NOT_AVAILABLE = (
    "NOT_AVAILABLE"
)


FIELDS = [
    "quality_version",

    "model",
    "split",

    "raw_demo_status",
    "integrity_status",

    "demo_num_steps",
    "demo_outcome",

    "original_profile_status",
    "original_selection_success",
    "original_outcome",

    "demo_baseline_trajectory_match",

    "quality_role",
    "quality_reason",

    # Main training eligibility.
    "main_demo_replay_eligible",
    "strong_bc_eligible",

    # Kept outside the main D_demo but may be used by a later
    # explicitly designed auxiliary off-policy experiment.
    "auxiliary_rl_eligible",

    "demo_npz_file",
    "demo_metadata_file",
]


def sha256_file(
    path: Path,
):
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        while True:
            block = f.read(
                1024
                *
                1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def read_facts():
    if not FACTS.is_file():
        raise FileNotFoundError(
            FACTS
        )

    with FACTS.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = [
            dict(row)
            for row
            in csv.DictReader(f)
        ]

    if len(
        rows
    ) != 49:
        raise RuntimeError(
            "Demonstration Quality Facts V1 "
            "must contain 49 train rows; "
            f"got {len(rows)}"
        )

    models = [
        row["model"]
        for row
        in rows
    ]

    if len(
        models
    ) != len(
        set(models)
    ):
        raise RuntimeError(
            "Duplicate models in "
            "Demonstration Quality Facts V1"
        )

    return rows


def classify_row(
    row,
):
    """
    Demonstration Quality V1 deliberately separates:

      1. demonstration quality/integrity, and
      2. environment resource risk.

    RSS, swap, tet-ratio and runtime are NOT quality thresholds.

    Quality is determined only after a raw demo exists and is
    independently reproduced by the frozen Original baseline.
    """

    model = row[
        "model"
    ]

    raw_status = row[
        "raw_demo_status"
    ]

    result = {
        field:
            ""
        for field
        in FIELDS
    }

    result[
        "quality_version"
    ] = QUALITY_VERSION

    for field in [
        "model",
        "split",
        "raw_demo_status",
        "demo_num_steps",
        "demo_outcome",
        "original_profile_status",
        "original_selection_success",
        "original_outcome",
        "demo_baseline_trajectory_match",
        "demo_npz_file",
        "demo_metadata_file",
    ]:
        result[
            field
        ] = row.get(
            field,
            ""
        )

    if (
        raw_status
        ==
        "NOT_COLLECTED"
    ):
        result[
            "integrity_status"
        ] = "NOT_APPLICABLE"

        result[
            "quality_role"
        ] = ROLE_NOT_AVAILABLE

        result[
            "quality_reason"
        ] = "RAW_DEMO_NOT_COLLECTED"

        result[
            "main_demo_replay_eligible"
        ] = "0"

        result[
            "strong_bc_eligible"
        ] = "0"

        result[
            "auxiliary_rl_eligible"
        ] = "0"

        return result

    if (
        raw_status
        !=
        "COLLECTED"
    ):
        raise RuntimeError(
            f"{model}: unknown "
            f"raw_demo_status={raw_status!r}"
        )

    # ==========================================================
    # Integrity gate.
    # ==========================================================

    if (
        row[
            "original_profile_status"
        ]
        !=
        "COMPLETE"
    ):
        raise RuntimeError(
            f"{model}: collected raw demo "
            "does not have a COMPLETE "
            "Original baseline"
        )

    if (
        row[
            "demo_baseline_trajectory_match"
        ]
        !=
        "MATCH"
    ):
        raise RuntimeError(
            f"{model}: collected raw demo "
            "does not reproduce its "
            "Original baseline trajectory"
        )

    if (
        row[
            "demo_outcome"
        ]
        !=
        row[
            "original_outcome"
        ]
    ):
        raise RuntimeError(
            f"{model}: demo/baseline "
            "outcome mismatch"
        )

    if (
        row[
            "demo_quality_class"
        ]
        !=
        "UNCLASSIFIED"
    ):
        raise RuntimeError(
            f"{model}: raw Demonstration V1 "
            "metadata must remain immutable "
            "and UNCLASSIFIED; quality is "
            "stored externally"
        )

    result[
        "integrity_status"
    ] = "VERIFIED"

    outcome = row[
        "demo_outcome"
    ]

    # ==========================================================
    # Demonstration Quality V1 policy.
    #
    # No resource threshold is used here.
    # ==========================================================

    if (
        outcome
        ==
        "FULL_HEX"
    ):
        result[
            "quality_role"
        ] = ROLE_BC_CORE

        result[
            "quality_reason"
        ] = (
            "VERIFIED_ORIGINAL_TRAJECTORY_"
            "ENDS_FULL_HEX"
        )

        result[
            "main_demo_replay_eligible"
        ] = "1"

        result[
            "strong_bc_eligible"
        ] = "1"

        result[
            "auxiliary_rl_eligible"
        ] = "0"

    elif (
        outcome
        ==
        "NON_FULL_HEX"
    ):
        result[
            "quality_role"
        ] = ROLE_RL_AUXILIARY

        result[
            "quality_reason"
        ] = (
            "VERIFIED_ORIGINAL_TRAJECTORY_"
            "ENDS_NON_FULL_HEX"
        )

        # Do not clone a failed global outcome in the main BC
        # training path.
        result[
            "main_demo_replay_eligible"
        ] = "0"

        result[
            "strong_bc_eligible"
        ] = "0"

        # Preserve it for a later explicitly controlled
        # off-policy / negative-experience experiment.
        result[
            "auxiliary_rl_eligible"
        ] = "1"

    elif (
        outcome
        ==
        "FINALIZATION_CRASH"
    ):
        result[
            "quality_role"
        ] = ROLE_EXCLUDE

        result[
            "quality_reason"
        ] = (
            "VERIFIED_TRAJECTORY_ENDS_"
            "FINALIZATION_CRASH"
        )

        result[
            "main_demo_replay_eligible"
        ] = "0"

        result[
            "strong_bc_eligible"
        ] = "0"

        result[
            "auxiliary_rl_eligible"
        ] = "0"

    else:
        raise RuntimeError(
            f"{model}: unknown "
            f"demo outcome={outcome!r}"
        )

    return result


def main():
    facts = read_facts()

    rows = [
        classify_row(
            row
        )
        for row
        in facts
    ]

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
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

    counts = {
        role:
            sum(
                row[
                    "quality_role"
                ]
                ==
                role
                for row
                in rows
            )
        for role
        in [
            ROLE_BC_CORE,
            ROLE_RL_AUXILIARY,
            ROLE_EXCLUDE,
            ROLE_NOT_AVAILABLE,
        ]
    }

    main_demo_transitions = sum(
        int(
            row[
                "demo_num_steps"
            ]
        )
        for row
        in rows
        if (
            row[
                "main_demo_replay_eligible"
            ]
            ==
            "1"
        )
    )

    auxiliary_transitions = sum(
        int(
            row[
                "demo_num_steps"
            ]
        )
        for row
        in rows
        if (
            row[
                "auxiliary_rl_eligible"
            ]
            ==
            "1"
        )
    )

    summary = {
        "quality_version":
            QUALITY_VERSION,

        "source_facts_file":
            FACTS.name,

        "source_facts_sha256":
            sha256_file(
                FACTS
            ),

        "policy": {
            "BC_CORE":
                (
                    "Collected and integrity-verified "
                    "Original trajectory ending FULL_HEX"
                ),

            "RL_AUXILIARY":
                (
                    "Collected and integrity-verified "
                    "Original trajectory ending "
                    "NON_FULL_HEX; excluded from "
                    "strong BC/main demo replay"
                ),

            "EXCLUDE":
                (
                    "Collected trajectory ending "
                    "FINALIZATION_CRASH"
                ),

            "NOT_AVAILABLE":
                "Raw demo not collected",

            "resource_policy":
                (
                    "Resource measurements are used "
                    "for environment-run feasibility, "
                    "not Demonstration Quality V1 "
                    "classification thresholds."
                ),
        },

        "counts":
            counts,

        "main_demo_transitions":
            main_demo_transitions,

        "auxiliary_transitions":
            auxiliary_transitions,
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print(
        "OUTPUT CSV:",
        OUTPUT_CSV,
    )

    print(
        "OUTPUT JSON:",
        OUTPUT_JSON,
    )

    print(
        "train rows:",
        len(
            rows
        ),
    )

    print(
        "BC_CORE:",
        counts[
            ROLE_BC_CORE
        ],
    )

    print(
        "RL_AUXILIARY:",
        counts[
            ROLE_RL_AUXILIARY
        ],
    )

    print(
        "EXCLUDE:",
        counts[
            ROLE_EXCLUDE
        ],
    )

    print(
        "NOT_AVAILABLE:",
        counts[
            ROLE_NOT_AVAILABLE
        ],
    )

    print(
        "main demo transitions:",
        main_demo_transitions,
    )

    print(
        "auxiliary transitions:",
        auxiliary_transitions,
    )


if __name__ == "__main__":
    main()
