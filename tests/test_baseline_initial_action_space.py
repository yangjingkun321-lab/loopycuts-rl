import csv
import sys
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

from evaluation.baseline_audit import (
    EXECUTABLE,
    validate_initial_action_space,
)


MANIFEST = (
    PROJECT_ROOT
    /
    "data/manifests/dataset_split_v2.csv"
)


def load_manifest():
    with MANIFEST.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return {
            row["model"]: row
            for row in csv.DictReader(f)
        }


def main():
    manifest = load_manifest()

    cases = [
        {
            "model": "mechanical02",
            "manifest_actionable": 80,
            "runtime_actions": 80,
        },
        {
            "model": "des6",
            "manifest_actionable": 96,
            "runtime_actions": 95,
        },
    ]

    for case in cases:
        model = case["model"]
        row = manifest[model]

        print()
        print("=" * 80)
        print("MODEL:", model)
        print("=" * 80)

        assert (
            int(
                row["actionable_nonconvex"]
            )
            ==
            case["manifest_actionable"]
        )

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

            actions = (
                validate_initial_action_space(
                    model=model,
                    row=row,
                    loop_file=Path(
                        row["loop_file"]
                    ),
                    client=client,
                )
            )

            assert (
                len(actions)
                ==
                case["runtime_actions"]
            )

            assert (
                int(
                    client.state[
                        "available"
                    ]
                )
                ==
                len(actions)
            )

            if model == "mechanical02":
                assert (
                    len(actions)
                    ==
                    int(
                        row[
                            "actionable_nonconvex"
                        ]
                    )
                )

            elif model == "des6":
                # Stage-1 static metadata contains
                # 96 non-convex loops, but Stage-2
                # initialization consumes loop 74.
                assert 74 in client.used
                assert 74 in client.top_relevant

                assert 74 not in client.reverted
                assert 74 not in client.nico_bug
                assert 74 not in actions

                assert (
                    int(
                        row[
                            "actionable_nonconvex"
                        ]
                    )
                    -
                    len(actions)
                    ==
                    1
                )

    print()
    print("=" * 80)
    print(
        "PASS: baseline initial action-space validation"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
