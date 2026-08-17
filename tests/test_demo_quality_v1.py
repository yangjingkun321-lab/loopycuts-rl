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


from imitation.build_demo_quality_v1 import (
    ROLE_BC_CORE,
    ROLE_EXCLUDE,
    ROLE_NOT_AVAILABLE,
    ROLE_RL_AUXILIARY,
    classify_row,
)


def make_row(
    *,
    raw_status,
    outcome="",
    match="",
):
    return {
        "model":
            "synthetic",

        "split":
            "train",

        "raw_demo_status":
            raw_status,

        "demo_num_steps":
            (
                "4"
                if raw_status == "COLLECTED"
                else ""
            ),

        "demo_outcome":
            outcome,

        "demo_quality_class":
            (
                "UNCLASSIFIED"
                if raw_status == "COLLECTED"
                else ""
            ),

        "demo_npz_file":
            "",

        "demo_metadata_file":
            "",

        "original_profile_status":
            (
                "COMPLETE"
                if raw_status == "COLLECTED"
                else "UNPROFILED"
            ),

        "original_selection_success":
            "",

        "original_outcome":
            outcome,

        "demo_baseline_trajectory_match":
            match,
    }


def main():
    full_hex = classify_row(
        make_row(
            raw_status=
                "COLLECTED",

            outcome=
                "FULL_HEX",

            match=
                "MATCH",
        )
    )

    assert (
        full_hex[
            "quality_role"
        ]
        ==
        ROLE_BC_CORE
    )

    assert (
        full_hex[
            "main_demo_replay_eligible"
        ]
        ==
        "1"
    )

    assert (
        full_hex[
            "strong_bc_eligible"
        ]
        ==
        "1"
    )


    non_full = classify_row(
        make_row(
            raw_status=
                "COLLECTED",

            outcome=
                "NON_FULL_HEX",

            match=
                "MATCH",
        )
    )

    assert (
        non_full[
            "quality_role"
        ]
        ==
        ROLE_RL_AUXILIARY
    )

    assert (
        non_full[
            "main_demo_replay_eligible"
        ]
        ==
        "0"
    )

    assert (
        non_full[
            "strong_bc_eligible"
        ]
        ==
        "0"
    )

    assert (
        non_full[
            "auxiliary_rl_eligible"
        ]
        ==
        "1"
    )


    crash = classify_row(
        make_row(
            raw_status=
                "COLLECTED",

            outcome=
                "FINALIZATION_CRASH",

            match=
                "MATCH",
        )
    )

    assert (
        crash[
            "quality_role"
        ]
        ==
        ROLE_EXCLUDE
    )

    assert (
        crash[
            "main_demo_replay_eligible"
        ]
        ==
        "0"
    )


    unavailable = classify_row(
        make_row(
            raw_status=
                "NOT_COLLECTED",
        )
    )

    assert (
        unavailable[
            "quality_role"
        ]
        ==
        ROLE_NOT_AVAILABLE
    )


    bad = make_row(
        raw_status=
            "COLLECTED",

        outcome=
            "FULL_HEX",

        match=
            "MISMATCH",
    )

    try:
        classify_row(
            bad
        )

    except RuntimeError:
        pass

    else:
        raise RuntimeError(
            "Integrity mismatch was "
            "incorrectly accepted"
        )


    print(
        "PASS: Demonstration "
        "Quality V1 classification"
    )


if __name__ == "__main__":
    main()
