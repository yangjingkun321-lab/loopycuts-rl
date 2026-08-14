from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


V1_CSV = Path(
    "/home/yjk/loopycuts_test/"
    "dataset_split_v1/"
    "dataset_split_v1.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "/home/yjk/loopycuts_test/"
    "dataset_split_v2"
)


#
# Author-corpus models that have already participated in
# engineering / reward / finalization development.
#
ENGINEERING_AUTHOR_MODELS = {
    "BracketInches",
    "bimba",
    "deckel",
    "eraser_ball",
    "cylinder_plate",
}


#
# Geometry audit result:
#
# tris_open / tris_closed have identical normalized bbox
# dimensions and very small normalized surface distances.
#
# They are treated as one near-duplicate geometry family.
#
TRIS_FAMILY = {
    "tris_open",
    "tris_closed",
}


#
# Independent static quantities used ONLY to choose a replacement
# for a vacated test slot from the SAME complexity stratum.
#
# regular_fraction is deliberately excluded because:
#
#   concave_fraction
# + regular_fraction
# + convex_fraction
# = 1
#
# and would therefore be redundant.
#
MATCH_FEATURES = [
    "parsed_loops",
    "actionable_nonconvex",
    "concave_fraction",
    "convex_fraction",
]


def validate_v1(
    df: pd.DataFrame,
):
    required = {
        "model",
        "mesh_file",
        "loop_file",
        "parsed_loops",
        "actionable_nonconvex",
        "concave",
        "regular",
        "convex",
        "concave_fraction",
        "regular_fraction",
        "convex_fraction",
        "complexity_stratum",
        "split",
    }

    missing = (
        required
        -
        set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Dataset Split V1 is missing columns: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )

    if len(
        df
    ) != 74:
        raise RuntimeError(
            f"Expected 74 models, got {len(df)}"
        )

    if not df[
        "model"
    ].is_unique:
        raise RuntimeError(
            "Dataset Split V1 contains duplicate models"
        )

    expected_counts = {
        "engineering_calibration":
            4,

        "train":
            50,

        "dev":
            10,

        "test":
            10,
    }

    actual_counts = (
        df[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    if (
        actual_counts
        !=
        expected_counts
    ):
        raise RuntimeError(
            "Unexpected Dataset Split V1 counts.\n"
            f"Expected: {expected_counts}\n"
            f"Actual:   {actual_counts}"
        )

    for model in (
        ENGINEERING_AUTHOR_MODELS
    ):
        if model not in set(
            df[
                "model"
            ]
        ):
            raise RuntimeError(
                f"Missing expected model: {model}"
            )

    for model in (
        TRIS_FAMILY
    ):
        if model not in set(
            df[
                "model"
            ]
        ):
            raise RuntimeError(
                f"Missing tris family model: {model}"
            )


def choose_replacement(
    *,
    df: pd.DataFrame,
    target_row: pd.Series,
    target_stratum: int,
):
    """
    Fill one vacated test slot.

    Constraints:
        - candidate currently belongs to train;
        - candidate belongs to exactly the same original
          complexity stratum;
        - candidate is not a known near-duplicate family member.

    Among valid candidates, choose the one whose static corpus
    feature vector is closest to the removed test model.

    Features are standardized using the original non-engineering
    V1 population, so quantities with different units do not
    dominate merely because of numerical scale.
    """

    candidates = df[
        (
            df[
                "split"
            ]
            ==
            "train"
        )
        &
        (
            df[
                "v1_complexity_stratum"
            ]
            ==
            target_stratum
        )
        &
        (
            ~df[
                "model"
            ].isin(
                TRIS_FAMILY
            )
        )
    ].copy()

    if candidates.empty:
        raise RuntimeError(
            f"No train replacement available in "
            f"stratum {target_stratum}"
        )

    reference = df[
        df[
            "v1_split"
        ]
        !=
        "engineering_calibration"
    ]

    scales = {}

    for feature in (
        MATCH_FEATURES
    ):
        scale = float(
            reference[
                feature
            ].std(
                ddof=0
            )
        )

        if (
            not math.isfinite(
                scale
            )
            or
            scale
            <=
            0.0
        ):
            raise RuntimeError(
                f"Invalid standard deviation "
                f"for feature {feature}: "
                f"{scale}"
            )

        scales[
            feature
        ] = scale

    scored = []

    for index, row in (
        candidates.iterrows()
    ):
        squared = 0.0
        components = {}

        for feature in (
            MATCH_FEATURES
        ):
            delta = (
                float(
                    row[
                        feature
                    ]
                )
                -
                float(
                    target_row[
                        feature
                    ]
                )
            ) / scales[
                feature
            ]

            components[
                feature
            ] = float(
                delta
            )

            squared += (
                delta
                *
                delta
            )

        score = float(
            math.sqrt(
                squared
            )
        )

        scored.append(
            (
                score,
                str(
                    row[
                        "model"
                    ]
                ),
                index,
                components,
            )
        )

    #
    # Deterministic tie break by model name.
    #
    scored.sort(
        key=lambda item: (
            item[
                0
            ],
            item[
                1
            ],
        )
    )

    (
        score,
        model,
        index,
        components,
    ) = scored[
        0
    ]

    return (
        index,
        model,
        score,
        components,
    )


def make_summary(
    df: pd.DataFrame,
):
    rows = []

    for split in [
        "engineering_calibration",
        "train",
        "dev",
        "test",
    ]:
        x = df[
            df[
                "split"
            ]
            ==
            split
        ]

        rows.append(
            {
                "split":
                    split,

                "num_models":
                    int(
                        len(
                            x
                        )
                    ),

                "loops_min":
                    int(
                        x[
                            "parsed_loops"
                        ].min()
                    ),

                "loops_median":
                    float(
                        x[
                            "parsed_loops"
                        ].median()
                    ),

                "loops_mean":
                    float(
                        x[
                            "parsed_loops"
                        ].mean()
                    ),

                "loops_max":
                    int(
                        x[
                            "parsed_loops"
                        ].max()
                    ),

                "actionable_min":
                    int(
                        x[
                            "actionable_nonconvex"
                        ].min()
                    ),

                "actionable_median":
                    float(
                        x[
                            "actionable_nonconvex"
                        ].median()
                    ),

                "actionable_mean":
                    float(
                        x[
                            "actionable_nonconvex"
                        ].mean()
                    ),

                "actionable_max":
                    int(
                        x[
                            "actionable_nonconvex"
                        ].max()
                    ),

                "concave_total":
                    int(
                        x[
                            "concave"
                        ].sum()
                    ),

                "regular_total":
                    int(
                        x[
                            "regular"
                        ].sum()
                    ),

                "convex_total":
                    int(
                        x[
                            "convex"
                        ].sum()
                    ),

                "mean_concave_fraction":
                    float(
                        x[
                            "concave_fraction"
                        ].mean()
                    ),

                "mean_regular_fraction":
                    float(
                        x[
                            "regular_fraction"
                        ].mean()
                    ),

                "mean_convex_fraction":
                    float(
                        x[
                            "convex_fraction"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--v1-csv",
        type=Path,
        default=V1_CSV,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if not args.v1_csv.is_file():
        raise FileNotFoundError(
            args.v1_csv
        )

    df = pd.read_csv(
        args.v1_csv
    )

    validate_v1(
        df
    )

    #
    # Preserve V1 provenance explicitly.
    #
    df[
        "v1_split"
    ] = df[
        "split"
    ]

    df[
        "v1_complexity_stratum"
    ] = df[
        "complexity_stratum"
    ]

    changes = []

    # ============================================================
    # Repair 1:
    #
    # Author cylinder is the same / near-identical geometry as the
    # external Cylinder used extensively during engineering.
    #
    # test -> engineering_calibration
    # ============================================================

    cylinder_mask = (
        df[
            "model"
        ]
        ==
        "cylinder_plate"
    )

    if int(
        cylinder_mask.sum()
    ) != 1:
        raise RuntimeError(
            "Expected exactly one cylinder_plate"
        )

    cylinder_index = df.index[
        cylinder_mask
    ][
        0
    ]

    cylinder_row = df.loc[
        cylinder_index
    ].copy()

    if (
        cylinder_row[
            "v1_split"
        ]
        !=
        "test"
    ):
        raise RuntimeError(
            "Expected V1 cylinder_plate "
            "to belong to test"
        )

    cylinder_stratum = int(
        cylinder_row[
            "v1_complexity_stratum"
        ]
    )

    df.at[
        cylinder_index,
        "split",
    ] = "engineering_calibration"

    df.at[
        cylinder_index,
        "complexity_stratum",
    ] = -1

    changes.append(
        {
            "model":
                "cylinder_plate",

            "v1_split":
                "test",

            "v2_split":
                "engineering_calibration",

            "reason":
                (
                    "near-duplicate of external "
                    "engineering Cylinder"
                ),

            "source_stratum":
                cylinder_stratum,

            "replacement_for":
                "",
        }
    )

    # ============================================================
    # Repair 2:
    #
    # tris_open / tris_closed are the same geometric family.
    # Keep both in train.
    # ============================================================

    tris_open_mask = (
        df[
            "model"
        ]
        ==
        "tris_open"
    )

    tris_closed_mask = (
        df[
            "model"
        ]
        ==
        "tris_closed"
    )

    if (
        int(
            tris_open_mask.sum()
        )
        !=
        1
        or
        int(
            tris_closed_mask.sum()
        )
        !=
        1
    ):
        raise RuntimeError(
            "Could not uniquely identify tris family"
        )

    tris_open_index = df.index[
        tris_open_mask
    ][
        0
    ]

    tris_open_row = df.loc[
        tris_open_index
    ].copy()

    tris_closed_row = df.loc[
        tris_closed_mask
    ].iloc[
        0
    ]

    if (
        tris_open_row[
            "v1_split"
        ]
        !=
        "test"
    ):
        raise RuntimeError(
            "Expected V1 tris_open to belong to test"
        )

    if (
        tris_closed_row[
            "v1_split"
        ]
        !=
        "train"
    ):
        raise RuntimeError(
            "Expected V1 tris_closed to belong to train"
        )

    tris_stratum = int(
        tris_open_row[
            "v1_complexity_stratum"
        ]
    )

    df.at[
        tris_open_index,
        "split",
    ] = "train"

    changes.append(
        {
            "model":
                "tris_open",

            "v1_split":
                "test",

            "v2_split":
                "train",

            "reason":
                (
                    "lock near-duplicate "
                    "tris_open/tris_closed family"
                ),

            "source_stratum":
                tris_stratum,

            "replacement_for":
                "",
        }
    )

    # ============================================================
    # Fill the two vacated test strata.
    #
    # We do NOT globally reshuffle the split.
    # ============================================================

    vacancies = [
        (
            "cylinder_plate",
            cylinder_row,
            cylinder_stratum,
        ),

        (
            "tris_open",
            tris_open_row,
            tris_stratum,
        ),
    ]

    for (
        removed_model,
        removed_row,
        stratum,
    ) in vacancies:
        (
            replacement_index,
            replacement_model,
            score,
            components,
        ) = choose_replacement(
            df=df,
            target_row=removed_row,
            target_stratum=stratum,
        )

        old_split = str(
            df.at[
                replacement_index,
                "split",
            ]
        )

        if old_split != "train":
            raise RuntimeError(
                "Replacement was not train"
            )

        df.at[
            replacement_index,
            "split",
        ] = "test"

        changes.append(
            {
                "model":
                    replacement_model,

                "v1_split":
                    "train",

                "v2_split":
                    "test",

                "reason":
                    (
                        "same-stratum static-feature "
                        "replacement"
                    ),

                "source_stratum":
                    int(
                        stratum
                    ),

                "replacement_for":
                    removed_model,

                "replacement_score":
                    float(
                        score
                    ),

                "delta_parsed_loops_z":
                    components[
                        "parsed_loops"
                    ],

                "delta_actionable_z":
                    components[
                        "actionable_nonconvex"
                    ],

                "delta_concave_fraction_z":
                    components[
                        "concave_fraction"
                    ],

                "delta_convex_fraction_z":
                    components[
                        "convex_fraction"
                    ],
            }
        )

    # ============================================================
    # Final invariants
    # ============================================================

    counts = (
        df[
            "split"
        ]
        .value_counts()
        .to_dict()
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

    if (
        counts
        !=
        expected_counts
    ):
        raise RuntimeError(
            "V2 split counts mismatch.\n"
            f"Expected: {expected_counts}\n"
            f"Actual:   {counts}"
        )

    actual_engineering = set(
        df.loc[
            df[
                "split"
            ]
            ==
            "engineering_calibration",
            "model",
        ]
    )

    if (
        actual_engineering
        !=
        ENGINEERING_AUTHOR_MODELS
    ):
        raise RuntimeError(
            "Engineering model set mismatch.\n"
            f"Expected: "
            f"{sorted(ENGINEERING_AUTHOR_MODELS)}\n"
            f"Actual:   "
            f"{sorted(actual_engineering)}"
        )

    tris_splits = set(
        df.loc[
            df[
                "model"
            ].isin(
                TRIS_FAMILY
            ),
            "split",
        ]
    )

    if tris_splits != {
        "train",
    }:
        raise RuntimeError(
            "tris family was not completely "
            "locked to train"
        )

    #
    # V1 design property is intentionally retained:
    #
    # exactly one dev and one test sample from each of the original
    # ten actionable-complexity strata.
    #
    for split in (
        "dev",
        "test",
    ):
        x = df[
            df[
                "split"
            ]
            ==
            split
        ]

        counts_by_stratum = (
            x[
                "v1_complexity_stratum"
            ]
            .astype(
                int
            )
            .value_counts()
            .to_dict()
        )

        expected_strata = {
            stratum:
                1
            for stratum
            in range(
                10
            )
        }

        if (
            counts_by_stratum
            !=
            expected_strata
        ):
            raise RuntimeError(
                f"{split}: expected exactly one "
                "model from each original "
                "complexity stratum.\n"
                f"Actual: "
                f"{counts_by_stratum}"
            )

    if (
        "cylinder_plate"
        in
        set(
            df.loc[
                df[
                    "split"
                ]
                ==
                "test",
                "model",
            ]
        )
    ):
        raise RuntimeError(
            "Author cylinder leaked into test"
        )

    # ============================================================
    # Outputs
    # ============================================================

    summary = make_summary(
        df
    )

    changes_df = pd.DataFrame(
        changes
    )

    split_order = {
        "engineering_calibration":
            0,

        "train":
            1,

        "dev":
            2,

        "test":
            3,
    }

    df[
        "_split_order"
    ] = df[
        "split"
    ].map(
        split_order
    )

    df = (
        df
        .sort_values(
            [
                "_split_order",
                "v1_complexity_stratum",
                "actionable_nonconvex",
                "parsed_loops",
                "model",
            ]
        )
        .drop(
            columns=[
                "_split_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        args.output_dir
        /
        "dataset_split_v2.csv"
    )

    summary_path = (
        args.output_dir
        /
        "dataset_split_v2_summary.csv"
    )

    changes_path = (
        args.output_dir
        /
        "dataset_split_v2_changes.csv"
    )

    json_path = (
        args.output_dir
        /
        "dataset_split_v2.json"
    )

    df.to_csv(
        manifest_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    changes_df.to_csv(
        changes_path,
        index=False,
    )

    payload = {
        "version":
            "dataset_split_v2",

        "source_manifest":
            str(
                args.v1_csv
            ),

        "strategy":
            (
                "minimal repair of Dataset Split V1"
            ),

        "counts":
            expected_counts,

        "geometry_audit_decisions":
            {
                "cylinder_plate":
                    (
                        "engineering_calibration; "
                        "near-duplicate of external "
                        "engineering Cylinder"
                    ),

                "tris_open/tris_closed":
                    (
                        "locked together in train"
                    ),

                "bimba/busto_bimba":
                    (
                        "not near-duplicate by current "
                        "geometry audit"
                    ),

                "Plate1-4":
                    (
                        "not near-duplicate by current "
                        "geometry audit"
                    ),

                "bone1/bone_femur":
                    (
                        "not near-duplicate by current "
                        "geometry audit"
                    ),
            },

        "models":
            df.to_dict(
                orient="records"
            ),

        "changes":
            changes,
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 92
    )

    print(
        "DATASET SPLIT V2"
    )

    print(
        "=" * 92
    )

    print()

    print(
        "Split counts:"
    )

    print(
        df[
            "split"
        ]
        .value_counts()
        .reindex(
            [
                "engineering_calibration",
                "train",
                "dev",
                "test",
            ]
        )
        .to_string()
    )

    print()

    print(
        "V1 -> V2 changes:"
    )

    print(
        changes_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Distribution summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Dev:"
    )

    print(
        df.loc[
            df[
                "split"
            ]
            ==
            "dev",
            [
                "v1_complexity_stratum",
                "model",
                "parsed_loops",
                "actionable_nonconvex",
                "concave",
                "regular",
                "convex",
            ],
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Held-out test:"
    )

    print(
        df.loc[
            df[
                "split"
            ]
            ==
            "test",
            [
                "v1_complexity_stratum",
                "model",
                "parsed_loops",
                "actionable_nonconvex",
                "concave",
                "regular",
                "convex",
            ],
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Near-duplicate lock:"
    )

    print(
        df.loc[
            df[
                "model"
            ].isin(
                TRIS_FAMILY
            ),
            [
                "model",
                "v1_split",
                "split",
            ],
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Saved:"
    )

    print(
        manifest_path
    )

    print(
        summary_path
    )

    print(
        changes_path
    )

    print(
        json_path
    )

    print()

    print(
        "PASS: Dataset Split V2 minimally repairs "
        "the identified geometry leakage while "
        "preserving one dev/test sample per original "
        "complexity stratum."
    )


if __name__ == "__main__":
    main()
