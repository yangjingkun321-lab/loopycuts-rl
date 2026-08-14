from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


CORPUS_CSV = Path(
    "/home/yjk/loopycuts_test/"
    "rl_corpus/"
    "loop_corpus.csv"
)

TEST_DATA_ROOT = Path(
    "/home/yjk/codes/LoopyCuts/"
    "test_data"
)

DEFAULT_OUTPUT_DIR = Path(
    "/home/yjk/loopycuts_test/"
    "dataset_split_v1"
)

SEED = 20260813


# ============================================================
# These author-corpus models have already participated in
# environment/reward/finalization engineering.
#
# They MUST NOT become blind test data.
#
# Match by parent directory of mesh_file rather than assuming
# exact capitalization of the CSV "model" field.
# ============================================================

ENGINEERING_MODEL_DIRS = {
    "bracketinches",
    "deckel",
    "eraser_ball",
    "bimba",
}


# Cylinder is external to the 74-model author corpus.
EXTERNAL_ENGINEERING_CASES = [
    {
        "name":
            "cylinder_plate",

        "mesh_file":
            (
                "/home/yjk/loopycuts_inputs/"
                "cylinder_plate_clean/"
                "cylinder_plate_rem_splitted.obj"
            ),

        "loop_file":
            (
                "/home/yjk/loopycuts_inputs/"
                "cylinder_plate_clean/"
                "cylinder_plate_rem_loop.txt"
            ),

        "role":
            "engineering_calibration_external",
    }
]


def validate_corpus(
    df: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "model",
        "relative_loop_file",
        "mesh_file",
        "mesh_exists",
        "header_loops",
        "parsed_loops",
        "count_matches",
        "concave",
        "regular",
        "convex",
        "singular",
        "actionable_nonconvex",
        "type_pattern",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            "Corpus CSV is missing fields: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if len(df) != 74:
        raise RuntimeError(
            f"Expected exactly 74 author models, "
            f"got {len(df)}"
        )

    if not df[
        "model"
    ].is_unique:
        duplicates = (
            df[
                df[
                    "model"
                ].duplicated(
                    keep=False
                )
            ][
                "model"
            ]
            .tolist()
        )

        raise RuntimeError(
            f"Duplicate model names: {duplicates}"
        )

    if not bool(
        df[
            "mesh_exists"
        ].astype(
            bool
        ).all()
    ):
        bad = (
            df.loc[
                ~df[
                    "mesh_exists"
                ].astype(
                    bool
                ),
                "model",
            ]
            .tolist()
        )

        raise RuntimeError(
            f"Missing meshes: {bad}"
        )

    if not bool(
        df[
            "count_matches"
        ].astype(
            bool
        ).all()
    ):
        bad = (
            df.loc[
                ~df[
                    "count_matches"
                ].astype(
                    bool
                ),
                "model",
            ]
            .tolist()
        )

        raise RuntimeError(
            f"Loop count mismatches: {bad}"
        )

    if not bool(
        (
            df[
                "singular"
            ].astype(
                int
            )
            ==
            0
        ).all()
    ):
        bad = (
            df.loc[
                df[
                    "singular"
                ].astype(
                    int
                )
                !=
                0,
                "model",
            ]
            .tolist()
        )

        raise RuntimeError(
            f"Unexpected SINGULAR loops: {bad}"
        )

    if not bool(
        (
            df[
                "header_loops"
            ].astype(
                int
            )
            ==
            df[
                "parsed_loops"
            ].astype(
                int
            )
        ).all()
    ):
        raise RuntimeError(
            "header_loops != parsed_loops"
        )

    df = df.copy()

    # ------------------------------------------------------------
    # Construct and verify absolute loop paths.
    # ------------------------------------------------------------

    df[
        "loop_file"
    ] = [
        str(
            TEST_DATA_ROOT
            /
            relative
        )
        for relative
        in df[
            "relative_loop_file"
        ]
    ]

    missing_loop_files = [
        path
        for path in df[
            "loop_file"
        ]
        if not Path(
            path
        ).is_file()
    ]

    if missing_loop_files:
        raise RuntimeError(
            "Derived loop files are missing:\n"
            +
            "\n".join(
                missing_loop_files
            )
        )

    missing_mesh_files = [
        path
        for path in df[
            "mesh_file"
        ]
        if not Path(
            path
        ).is_file()
    ]

    if missing_mesh_files:
        raise RuntimeError(
            "Mesh files are missing:\n"
            +
            "\n".join(
                missing_mesh_files
            )
        )

    # ------------------------------------------------------------
    # Useful normalized composition features.
    # ------------------------------------------------------------

    parsed = (
        df[
            "parsed_loops"
        ]
        .astype(
            float
        )
    )

    actionable = (
        df[
            "actionable_nonconvex"
        ]
        .astype(
            float
        )
    )

    if bool(
        (
            parsed
            <=
            0
        ).any()
    ):
        raise RuntimeError(
            "parsed_loops must be positive"
        )

    if bool(
        (
            actionable
            <=
            0
        ).any()
    ):
        raise RuntimeError(
            "actionable_nonconvex must be positive"
        )

    df[
        "concave_fraction"
    ] = (
        df[
            "concave"
        ].astype(
            float
        )
        /
        parsed
    )

    df[
        "regular_fraction"
    ] = (
        df[
            "regular"
        ].astype(
            float
        )
        /
        parsed
    )

    df[
        "convex_fraction"
    ] = (
        df[
            "convex"
        ].astype(
            float
        )
        /
        parsed
    )

    return df


def mark_engineering_models(
    df: pd.DataFrame,
):
    df = df.copy()

    df[
        "mesh_parent"
    ] = [
        Path(
            path
        ).parent.name.lower()
        for path
        in df[
            "mesh_file"
        ]
    ]

    found = set(
        df.loc[
            df[
                "mesh_parent"
            ].isin(
                ENGINEERING_MODEL_DIRS
            ),
            "mesh_parent",
        ]
    )

    if found != ENGINEERING_MODEL_DIRS:
        raise RuntimeError(
            "Engineering model detection mismatch.\n"
            f"Expected: "
            f"{sorted(ENGINEERING_MODEL_DIRS)}\n"
            f"Found: "
            f"{sorted(found)}"
        )

    engineering = (
        df[
            df[
                "mesh_parent"
            ].isin(
                ENGINEERING_MODEL_DIRS
            )
        ]
        .copy()
    )

    if len(
        engineering
    ) != 4:
        raise RuntimeError(
            "Expected exactly 4 author-corpus "
            "engineering models, got "
            f"{len(engineering)}"
        )

    remaining = (
        df[
            ~df[
                "mesh_parent"
            ].isin(
                ENGINEERING_MODEL_DIRS
            )
        ]
        .copy()
    )

    if len(
        remaining
    ) != 70:
        raise RuntimeError(
            f"Expected 70 remaining models, "
            f"got {len(remaining)}"
        )

    return (
        engineering,
        remaining,
    )


def split_remaining(
    remaining: pd.DataFrame,
):
    """
    Deterministic stratification.

    70 remaining models are ordered by the variable that most
    directly controls RL candidate-set/horizon complexity:

        actionable_nonconvex

    with parsed_loops and model used only as deterministic
    tie-breakers.

    The sorted models are partitioned into 10 consecutive strata
    of exactly 7 models.

    Each stratum contributes:

        5 train
        1 dev
        1 test

    The within-stratum choice is reproducibly shuffled with a
    fixed seed.
    """

    ordered = (
        remaining
        .sort_values(
            [
                "actionable_nonconvex",
                "parsed_loops",
                "model",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        ordered
    ) != 70:
        raise RuntimeError(
            "split_remaining requires exactly "
            "70 models"
        )

    ordered[
        "complexity_stratum"
    ] = (
        np.arange(
            len(
                ordered
            )
        )
        //
        7
    )

    assignments = {}

    for stratum in range(
        10
    ):
        rows = (
            ordered[
                ordered[
                    "complexity_stratum"
                ]
                ==
                stratum
            ]
        )

        if len(
            rows
        ) != 7:
            raise RuntimeError(
                f"Stratum {stratum} does not "
                "contain exactly 7 models"
            )

        models = (
            rows[
                "model"
            ]
            .tolist()
        )

        rng = random.Random(
            SEED
            +
            stratum
        )

        rng.shuffle(
            models
        )

        #
        # One dev and one test from every complexity stratum.
        #
        assignments[
            models[
                0
            ]
        ] = "dev"

        assignments[
            models[
                1
            ]
        ] = "test"

        for model in models[
            2:
        ]:
            assignments[
                model
            ] = "train"

    ordered[
        "split"
    ] = [
        assignments[
            model
        ]
        for model
        in ordered[
            "model"
        ]
    ]

    counts = (
        ordered[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    expected = {
        "train":
            50,

        "dev":
            10,

        "test":
            10,
    }

    if counts != expected:
        raise RuntimeError(
            "Unexpected split counts.\n"
            f"Expected: {expected}\n"
            f"Got: {counts}"
        )

    #
    # Every stratum must contain exactly 5/1/1.
    #
    check = (
        ordered
        .groupby(
            [
                "complexity_stratum",
                "split",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    for stratum in range(
        10
    ):
        row = check.loc[
            stratum
        ]

        if (
            int(
                row.get(
                    "train",
                    0,
                )
            )
            !=
            5
            or
            int(
                row.get(
                    "dev",
                    0,
                )
            )
            !=
            1
            or
            int(
                row.get(
                    "test",
                    0,
                )
            )
            !=
            1
        ):
            raise RuntimeError(
                f"Stratum {stratum} does not "
                "have 5/1/1 allocation"
            )

    return ordered


def make_summary(
    manifest: pd.DataFrame,
):
    rows = []

    for split in (
        "engineering_calibration",
        "train",
        "dev",
        "test",
    ):
        df = (
            manifest[
                manifest[
                    "split"
                ]
                ==
                split
            ]
        )

        if df.empty:
            continue

        rows.append(
            {
                "split":
                    split,

                "num_models":
                    len(
                        df
                    ),

                "loops_min":
                    int(
                        df[
                            "parsed_loops"
                        ].min()
                    ),

                "loops_median":
                    float(
                        df[
                            "parsed_loops"
                        ].median()
                    ),

                "loops_mean":
                    float(
                        df[
                            "parsed_loops"
                        ].mean()
                    ),

                "loops_max":
                    int(
                        df[
                            "parsed_loops"
                        ].max()
                    ),

                "actionable_min":
                    int(
                        df[
                            "actionable_nonconvex"
                        ].min()
                    ),

                "actionable_median":
                    float(
                        df[
                            "actionable_nonconvex"
                        ].median()
                    ),

                "actionable_mean":
                    float(
                        df[
                            "actionable_nonconvex"
                        ].mean()
                    ),

                "actionable_max":
                    int(
                        df[
                            "actionable_nonconvex"
                        ].max()
                    ),

                "concave_total":
                    int(
                        df[
                            "concave"
                        ].sum()
                    ),

                "regular_total":
                    int(
                        df[
                            "regular"
                        ].sum()
                    ),

                "convex_total":
                    int(
                        df[
                            "convex"
                        ].sum()
                    ),

                "mean_concave_fraction":
                    float(
                        df[
                            "concave_fraction"
                        ].mean()
                    ),

                "mean_regular_fraction":
                    float(
                        df[
                            "regular_fraction"
                        ].mean()
                    ),

                "mean_convex_fraction":
                    float(
                        df[
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
        "--corpus-csv",
        type=Path,
        default=CORPUS_CSV,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if not args.corpus_csv.is_file():
        raise FileNotFoundError(
            args.corpus_csv
        )

    df = pd.read_csv(
        args.corpus_csv
    )

    df = validate_corpus(
        df
    )

    (
        engineering,
        remaining,
    ) = mark_engineering_models(
        df
    )

    split_df = split_remaining(
        remaining
    )

    engineering = engineering.copy()

    engineering[
        "split"
    ] = (
        "engineering_calibration"
    )

    engineering[
        "complexity_stratum"
    ] = -1

    manifest = pd.concat(
        [
            engineering,
            split_df,
        ],
        ignore_index=True,
    )

    if len(
        manifest
    ) != 74:
        raise RuntimeError(
            "Final manifest does not "
            "contain 74 author models"
        )

    if not manifest[
        "model"
    ].is_unique:
        raise RuntimeError(
            "Final manifest contains "
            "duplicate models"
        )

    split_counts = (
        manifest[
            "split"
        ]
        .value_counts()
        .to_dict()
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

    if split_counts != expected_counts:
        raise RuntimeError(
            "Final split counts mismatch.\n"
            f"Expected: {expected_counts}\n"
            f"Got: {split_counts}"
        )

    #
    # Stable output ordering:
    #
    # engineering first, then train/dev/test,
    # complexity from small to large.
    #
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

    manifest[
        "_split_order"
    ] = manifest[
        "split"
    ].map(
        split_order
    )

    manifest = (
        manifest
        .sort_values(
            [
                "_split_order",
                "complexity_stratum",
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

    summary = make_summary(
        manifest
    )

    strata = (
        manifest[
            manifest[
                "split"
            ]
            !=
            "engineering_calibration"
        ]
        .groupby(
            [
                "complexity_stratum",
                "split",
            ]
        )
        .agg(
            num_models=(
                "model",
                "size",
            ),

            actionable_min=(
                "actionable_nonconvex",
                "min",
            ),

            actionable_max=(
                "actionable_nonconvex",
                "max",
            ),

            loops_min=(
                "parsed_loops",
                "min",
            ),

            loops_max=(
                "parsed_loops",
                "max",
            ),
        )
        .reset_index()
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        args.output_dir
        /
        "dataset_split_v1.csv"
    )

    summary_path = (
        args.output_dir
        /
        "dataset_split_v1_summary.csv"
    )

    strata_path = (
        args.output_dir
        /
        "dataset_split_v1_strata.csv"
    )

    json_path = (
        args.output_dir
        /
        "dataset_split_v1.json"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    strata.to_csv(
        strata_path,
        index=False,
    )

    payload = {
        "version":
            "dataset_split_v1",

        "seed":
            SEED,

        "source":
            {
                "author_corpus":
                    str(
                        args.corpus_csv
                    ),

                "author_num_models":
                    74,
            },

        "policy":
            {
                "engineering_author_models":
                    4,

                "train":
                    50,

                "dev":
                    10,

                "test":
                    10,

                "stratification_variable":
                    "actionable_nonconvex",

                "num_complexity_strata":
                    10,

                "models_per_stratum":
                    7,

                "per_stratum":
                    {
                        "train":
                            5,

                        "dev":
                            1,

                        "test":
                            1,
                    },

                "engineering_parent_dirs":
                    sorted(
                        ENGINEERING_MODEL_DIRS
                    ),
            },

        "external_engineering_cases":
            EXTERNAL_ENGINEERING_CASES,

        "models":
            manifest.to_dict(
                orient="records"
            ),
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
        "=" * 84
    )

    print(
        "DATASET SPLIT V1"
    )

    print(
        "=" * 84
    )

    print()

    print(
        "Split counts:"
    )

    print(
        manifest[
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
        "Engineering/calibration author models:"
    )

    print(
        manifest.loc[
            manifest[
                "split"
            ]
            ==
            "engineering_calibration",
            [
                "model",
                "mesh_file",
                "parsed_loops",
                "actionable_nonconvex",
            ],
        ].to_string(
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
        "Dev models:"
    )

    print(
        manifest.loc[
            manifest[
                "split"
            ]
            ==
            "dev",
            [
                "complexity_stratum",
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
        "Held-out test models:"
    )

    print(
        manifest.loc[
            manifest[
                "split"
            ]
            ==
            "test",
            [
                "complexity_stratum",
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
        "Saved:"
    )

    print(
        manifest_path
    )

    print(
        summary_path
    )

    print(
        strata_path
    )

    print(
        json_path
    )

    print()

    print(
        "PASS: deterministic 4 / 50 / 10 / 10 "
        "engineering/train/dev/test split generated."
    )


if __name__ == "__main__":
    main()
