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


from bridge.cpp_client import LoopyCutsClient


EXECUTABLE = (
    "/home/yjk/codes/LoopyCuts_v5/"
    "volumetric_cutter/"
    "volumetric_cutter"
)

MESH = (
    "/home/yjk/loopycuts_inputs/"
    "cylinder_plate_clean/"
    "cylinder_plate_rem_splitted.obj"
)

LOOPS = (
    "/home/yjk/loopycuts_inputs/"
    "cylinder_plate_clean/"
    "cylinder_plate_rem_loop.txt"
)


EXPECTED_NICO_BUG = [
    66,
    68,
    70,
    72,
    74,
    76,
    78,
    80,
    82,
    84,
    86,
    88,
    90,
]


#
# This is the already verified random_seed3 prefix,
# stopped immediately after the first reverted action.
#
PREFIX_TO_REVERT = [
    0,
    1,
    31,
    3,
    7,
    23,
    22,
]


def main():
    with LoopyCutsClient(
        executable=EXECUTABLE,
        mesh_file=MESH,
        loop_file=LOOPS,
        echo_logs=False,
    ) as client:

        # ============================================================
        # Initial state
        # ============================================================

        print(
            "Initial dynamic loop status:"
        )

        print(
            "used:",
            client.used,
        )

        print(
            "reverted:",
            client.reverted,
        )

        print(
            "nico_bug:",
            client.nico_bug,
        )

        print(
            "top_relevant:",
            client.top_relevant,
        )

        print(
            "actions[:10]:",
            client.actions[:10],
        )

        assert client.used == []

        assert client.reverted == []

        assert (
            client.nico_bug
            ==
            EXPECTED_NICO_BUG
        )

        assert client.top_relevant == []

        assert client.actions == (
            list(
                range(65)
            )
        )

        # ============================================================
        # STATE command must refresh the status lists while preserving
        # the old get_state() API.
        # ============================================================

        state, actions = (
            client.get_state()
        )

        assert (
            state
            is client.state
        )

        assert (
            actions
            is client.actions
        )

        assert client.used == []

        assert client.reverted == []

        assert (
            client.nico_bug
            ==
            EXPECTED_NICO_BUG
        )

        assert client.top_relevant == []

        # ============================================================
        # Replay the verified random_seed3 prefix through loop 22.
        # ============================================================

        step_result = None

        for (
            step_index,
            loop_id,
        ) in enumerate(
            PREFIX_TO_REVERT,
            start=1,
        ):
            assert (
                loop_id
                in client.actions
            )

            (
                step_result,
                state,
                actions,
            ) = client.step(
                loop_id
            )

            print()

            print(
                f"STEP {step_index} "
                f"loop={loop_id} "
                f"status="
                f"{step_result['status']}"
            )

            print(
                "used:",
                client.used,
            )

            print(
                "reverted:",
                client.reverted,
            )

            print(
                "available:",
                state[
                    "available"
                ],
            )

            #
            # These statuses should not spontaneously change.
            #
            assert (
                client.nico_bug
                ==
                EXPECTED_NICO_BUG
            )

            assert (
                client.top_relevant
                ==
                []
            )

            #
            # Any executed/reverted loop must no longer be legal.
            #
            assert (
                loop_id
                not in actions
            )

            #
            # Cylinder loop 0 is known not to consume additional mates.
            #
            if loop_id == 0:
                assert (
                    client.used
                    ==
                    [0]
                )

                assert (
                    client.reverted
                    ==
                    []
                )

        # ============================================================
        # The verified random_seed3 trajectory reverts loop 22.
        # ============================================================

        assert step_result is not None

        assert (
            step_result[
                "loop_id"
            ]
            ==
            22
        )

        assert (
            step_result[
                "status"
            ]
            ==
            "REVERTED"
        )

        assert (
            step_result[
                "reverted"
            ]
            ==
            1
        )

        assert (
            22
            in client.reverted
        )

        assert (
            22
            not in client.actions
        )

        print()

        print(
            "Final checked "
            "dynamic loop status:"
        )

        print(
            "used:",
            client.used,
        )

        print(
            "reverted:",
            client.reverted,
        )

        print(
            "nico_bug:",
            client.nico_bug,
        )

        print(
            "top_relevant:",
            client.top_relevant,
        )

    print()

    print(
        "PASS: cpp_client parses "
        "USED / REVERTED / "
        "NICO_BUG / TOP_RELEVANT "
        "without changing existing "
        "STATE/STEP APIs."
    )


if __name__ == "__main__":
    main()
