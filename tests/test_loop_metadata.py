import math
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


from dataset_tools.loop_metadata import (
    LoopMetadataParseError,
    parse_loop_metadata,
)


CYLINDER_LOOP_FILE = Path(
    "/home/yjk/loopycuts_inputs/"
    "cylinder_plate_clean/"
    "cylinder_plate_rem_loop.txt"
)


def main():
    loops = parse_loop_metadata(
        CYLINDER_LOOP_FILE
    )

    print(
        "num_loops:",
        len(loops),
    )

    assert len(loops) == 91

    # ============================================================
    # IDs must exactly match serialization order.
    # ============================================================

    assert [
        loop.loop_id
        for loop in loops
    ] == list(
        range(91)
    )

    # ============================================================
    # Type partition confirmed by the real C++ Loops loader.
    # ============================================================

    assert all(
        loops[i].loop_type
        ==
        "CONCAVE"
        for i in range(
            0,
            2,
        )
    )

    assert all(
        loops[i].loop_type
        ==
        "REGULAR"
        for i in range(
            2,
            65,
        )
    )

    assert all(
        loops[i].loop_type
        ==
        "CONVEX"
        for i in range(
            65,
            91,
        )
    )

    type_counts = {
        loop_type: sum(
            loop.loop_type
            ==
            loop_type
            for loop in loops
        )
        for loop_type in (
            "CONCAVE",
            "REGULAR",
            "CONVEX",
        )
    }

    print(
        "type_counts:",
        type_counts,
    )

    assert type_counts == {
        "CONCAVE": 2,
        "REGULAR": 63,
        "CONVEX": 26,
    }

    # ============================================================
    # Segment counts / open-closed values cross-checked against
    # the C++ startup log from the same file.
    # ============================================================

    expected = {
        0: (
            "CONCAVE",
            True,
            72,
        ),

        1: (
            "CONCAVE",
            True,
            72,
        ),

        2: (
            "REGULAR",
            True,
            323,
        ),

        3: (
            "REGULAR",
            True,
            276,
        ),

        63: (
            "REGULAR",
            True,
            373,
        ),

        64: (
            "REGULAR",
            True,
            338,
        ),

        65: (
            "CONVEX",
            True,
            72,
        ),

        67: (
            "CONVEX",
            False,
            36,
        ),

        75: (
            "CONVEX",
            False,
            2,
        ),

        90: (
            "CONVEX",
            False,
            36,
        ),
    }

    for (
        loop_id,
        (
            expected_type,
            expected_closed,
            expected_segments,
        ),
    ) in expected.items():

        loop = loops[
            loop_id
        ]

        assert (
            loop.loop_type
            ==
            expected_type
        )

        assert (
            loop.closed
            is expected_closed
        )

        assert (
            loop.num_segments
            ==
            expected_segments
        )

    # ============================================================
    # Sharp metadata invariants.
    #
    # We deliberately do NOT hard-code exact sharp counts here yet;
    # they were not printed by the C++ startup log.
    # ============================================================

    for loop in loops:
        assert (
            0
            <=
            loop.num_sharp_segments
            <=
            loop.num_segments
        )

        assert (
            0.0
            <=
            loop.sharp_fraction
            <=
            1.0
        )

        if (
            loop.num_segments
            >
            0
        ):
            expected_fraction = (
                loop.num_sharp_segments
                /
                loop.num_segments
            )

            assert math.isclose(
                loop.sharp_fraction,
                expected_fraction,
                rel_tol=0.0,
                abs_tol=1e-15,
            )

    # ============================================================
    # Serialized position.
    # ============================================================

    assert math.isclose(
        loops[0]
        .serialized_position,
        0.0,
    )

    assert math.isclose(
        loops[45]
        .serialized_position,
        45.0 / 90.0,
    )

    assert math.isclose(
        loops[90]
        .serialized_position,
        1.0,
    )

    # ============================================================
    # Print a few records so the parser result is easy to inspect.
    # ============================================================

    print()

    print(
        "Selected loop metadata:"
    )

    for loop_id in (
        0,
        1,
        2,
        3,
        63,
        64,
        65,
        67,
        75,
        90,
    ):
        print(
            loops[
                loop_id
            ]
        )

    print()

    print(
        "PASS: _loop.txt static metadata "
        "matches the C++ Loops loader."
    )


if __name__ == "__main__":
    main()
