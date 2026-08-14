import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median


LOOPYCUTS_ROOT = Path(
    "/home/yjk/codes/LoopyCuts"
)

TEST_DATA_ROOT = (
    LOOPYCUTS_ROOT /
    "test_data"
)

OUTPUT_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "rl_corpus"
)

CSV_FILE = (
    OUTPUT_ROOT /
    "loop_corpus.csv"
)

JSON_FILE = (
    OUTPUT_ROOT /
    "loop_corpus.json"
)


TYPE_NAMES = {
    "CONCAVE",
    "REGULAR",
    "CONVEX",
    "SINGULAR",
}


TYPE_SHORT = {
    "CONCAVE": "C",
    "REGULAR": "R",
    "CONVEX": "V",
    "SINGULAR": "S",
}


def percentile(values, p):
    """
    Linear-interpolated percentile.

    p is in [0, 100].
    """

    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return float(values[0])

    pos = (
        (len(values) - 1)
        * p
        / 100.0
    )

    lower = math.floor(pos)
    upper = math.ceil(pos)

    if lower == upper:
        return float(
            values[lower]
        )

    weight = pos - lower

    return (
        values[lower] * (1.0 - weight)
        +
        values[upper] * weight
    )


def compress_type_pattern(types):
    """
    Example:

        C C R R R C V V

    becomes:

        C2 R3 C1 V2
    """

    if not types:
        return ""

    parts = []

    current = types[0]
    count = 1

    for t in types[1:]:
        if t == current:
            count += 1
        else:
            parts.append(
                f"{TYPE_SHORT.get(current, current)}"
                f"{count}"
            )

            current = t
            count = 1

    parts.append(
        f"{TYPE_SHORT.get(current, current)}"
        f"{count}"
    )

    return " ".join(parts)


def parse_loop_file(path):
    lines = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    ]

    nonempty = [
        line
        for line in lines
        if line
    ]

    if not nonempty:
        raise ValueError(
            "empty loop file"
        )

    try:
        header_count = int(
            nonempty[0]
        )

    except ValueError as exc:
        raise ValueError(
            "first non-empty line "
            "is not an integer loop count"
        ) from exc

    #
    # Every loop block contains exactly one
    # loop-type line.
    #
    types = [
        line.upper()
        for line in nonempty[1:]
        if line.upper()
        in TYPE_NAMES
    ]

    counts = Counter(types)

    parsed_count = len(types)

    #
    # Matching Stage-2 mesh produced together
    # with this loop file.
    #
    filename = path.name

    if filename.endswith(
        "_loop.txt"
    ):
        mesh_name = (
            filename[
                :-len("_loop.txt")
            ]
            +
            "_splitted.obj"
        )

        mesh_path = (
            path.parent /
            mesh_name
        )

    else:
        mesh_path = None

    relative_path = (
        path.relative_to(
            TEST_DATA_ROOT
        )
    )

    return {
        "model":
            path.parent.name,

        "loop_file":
            str(path),

        "relative_loop_file":
            str(relative_path),

        "mesh_file":
            (
                str(mesh_path)
                if mesh_path
                else ""
            ),

        "mesh_exists":
            bool(
                mesh_path
                and mesh_path.is_file()
            ),

        "header_loops":
            header_count,

        "parsed_loops":
            parsed_count,

        "count_matches":
            (
                header_count
                == parsed_count
            ),

        "concave":
            counts[
                "CONCAVE"
            ],

        "regular":
            counts[
                "REGULAR"
            ],

        "convex":
            counts[
                "CONVEX"
            ],

        "singular":
            counts[
                "SINGULAR"
            ],

        "actionable_nonconvex":
            (
                counts["CONCAVE"]
                +
                counts["REGULAR"]
                +
                counts["SINGULAR"]
            ),

        "type_pattern":
            compress_type_pattern(
                types
            ),

        "types":
            types,
    }


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    loop_files = sorted(
        TEST_DATA_ROOT.rglob(
            "*_loop.txt"
        )
    )

    if not loop_files:
        raise RuntimeError(
            "No *_loop.txt files found under "
            f"{TEST_DATA_ROOT}"
        )

    rows = []
    failures = []

    for path in loop_files:
        try:
            row = parse_loop_file(
                path
            )

            rows.append(row)

        except Exception as exc:
            failures.append(
                {
                    "loop_file":
                        str(path),

                    "error":
                        str(exc),
                }
            )

    #
    # Sort largest models first.
    #
    rows.sort(
        key=lambda x: (
            x["header_loops"],
            x["model"],
        ),
        reverse=True,
    )

    totals = [
        row["header_loops"]
        for row in rows
    ]

    actionable = [
        row[
            "actionable_nonconvex"
        ]
        for row in rows
    ]

    concaves = [
        row["concave"]
        for row in rows
    ]

    regulars = [
        row["regular"]
        for row in rows
    ]

    convexes = [
        row["convex"]
        for row in rows
    ]

    mismatches = [
        row
        for row in rows
        if not row["count_matches"]
    ]

    missing_meshes = [
        row
        for row in rows
        if not row["mesh_exists"]
    ]

    singular_files = [
        row
        for row in rows
        if row["singular"] > 0
    ]

    summary = {
        "test_data_root":
            str(TEST_DATA_ROOT),

        "num_loop_files":
            len(loop_files),

        "num_parsed":
            len(rows),

        "num_parse_failures":
            len(failures),

        "num_header_mismatches":
            len(mismatches),

        "num_missing_splitted_mesh":
            len(missing_meshes),

        "num_files_with_singular":
            len(singular_files),

        "total_loops": {
            "min":
                min(totals)
                if totals
                else None,

            "max":
                max(totals)
                if totals
                else None,

            "mean":
                mean(totals)
                if totals
                else None,

            "median":
                median(totals)
                if totals
                else None,

            "p90":
                percentile(
                    totals,
                    90,
                ),

            "p95":
                percentile(
                    totals,
                    95,
                ),

            "p99":
                percentile(
                    totals,
                    99,
                ),
        },

        "actionable_nonconvex": {
            "min":
                min(actionable)
                if actionable
                else None,

            "max":
                max(actionable)
                if actionable
                else None,

            "median":
                median(actionable)
                if actionable
                else None,

            "p95":
                percentile(
                    actionable,
                    95,
                ),

            "p99":
                percentile(
                    actionable,
                    99,
                ),
        },

        "type_totals": {
            "concave":
                sum(concaves),

            "regular":
                sum(regulars),

            "convex":
                sum(convexes),

            "singular":
                sum(
                    row["singular"]
                    for row in rows
                ),
        },
    }

    #
    # CSV: one row per author loop file.
    #
    csv_fields = [
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
    ]

    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row[key]
                    for key in csv_fields
                }
            )

    #
    # JSON contains full type sequence as well.
    #
    with JSON_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "summary":
                    summary,

                "models":
                    rows,

                "parse_failures":
                    failures,

                "header_mismatches":
                    mismatches,

                "missing_meshes":
                    missing_meshes,

                "files_with_singular":
                    singular_files,
            },
            f,
            indent=2,
        )

    print()
    print(
        "============================================"
    )

    print(
        "LOOPYCUTS RL CORPUS SCAN"
    )

    print(
        "============================================"
    )

    print(
        "loop files found:",
        summary[
            "num_loop_files"
        ],
    )

    print(
        "parsed:",
        summary[
            "num_parsed"
        ],
    )

    print(
        "parse failures:",
        summary[
            "num_parse_failures"
        ],
    )

    print(
        "header mismatches:",
        summary[
            "num_header_mismatches"
        ],
    )

    print(
        "missing *_splitted.obj:",
        summary[
            "num_missing_splitted_mesh"
        ],
    )

    print(
        "files containing SINGULAR:",
        summary[
            "num_files_with_singular"
        ],
    )

    print()

    print(
        "===== TOTAL LOOP COUNT ====="
    )

    for key, value in (
        summary[
            "total_loops"
        ].items()
    ):
        print(
            f"{key:>8}: {value}"
        )

    print()

    print(
        "===== ACTIONABLE NON-CONVEX ====="
    )

    for key, value in (
        summary[
            "actionable_nonconvex"
        ].items()
    ):
        print(
            f"{key:>8}: {value}"
        )

    print()

    print(
        "===== GLOBAL TYPE TOTALS ====="
    )

    for key, value in (
        summary[
            "type_totals"
        ].items()
    ):
        print(
            f"{key:>8}: {value}"
        )

    print()

    print(
        "===== 20 LARGEST LOOP FILES ====="
    )

    print(
        f"{'model':<28}"
        f"{'loops':>7}"
        f"{'C':>6}"
        f"{'R':>6}"
        f"{'V':>6}"
        f"{'act':>7}  "
        f"pattern"
    )

    for row in rows[:20]:
        print(
            f"{row['model']:<28}"
            f"{row['header_loops']:>7}"
            f"{row['concave']:>6}"
            f"{row['regular']:>6}"
            f"{row['convex']:>6}"
            f"{row['actionable_nonconvex']:>7}  "
            f"{row['type_pattern']}"
        )

    print()

    if mismatches:
        print(
            "===== HEADER MISMATCHES ====="
        )

        for row in mismatches:
            print(
                row[
                    "relative_loop_file"
                ],
                "header=",
                row[
                    "header_loops"
                ],
                "parsed=",
                row[
                    "parsed_loops"
                ],
            )

        print()

    if failures:
        print(
            "===== PARSE FAILURES ====="
        )

        for item in failures:
            print(
                item[
                    "loop_file"
                ],
                ":",
                item[
                    "error"
                ],
            )

        print()

    print(
        "CSV :",
        CSV_FILE,
    )

    print(
        "JSON:",
        JSON_FILE,
    )


if __name__ == "__main__":
    main()