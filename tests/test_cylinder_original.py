import sys
from pathlib import Path


PROJECT_ROOT = Path(
    "/home/yjk/codes/loopycuts_rl"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from bridge.cpp_client import LoopyCutsClient


EXECUTABLE = (
    "/home/yjk/codes/LoopyCuts/"
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

OUTPUT = (
    "/home/yjk/loopycuts_test/"
    "cylinder_plate/"
    "python_bridge_regression/"
    "original"
)


EXPECTED = {
    0: {
        "verts_after": 39604,
        "tets_after": 157160,
        "converged": 0,
    },
    1: {
        "verts_after": 44620,
        "tets_after": 183648,
        "converged": 0,
    },
    2: {
        "verts_after": 47825,
        "tets_after": 201148,
        "converged": 0,
    },
    3: {
        "verts_after": 50802,
        "tets_after": 217478,
        "converged": 1,
    },
}


def main():
    with LoopyCutsClient(
        executable=EXECUTABLE,
        mesh_file=MESH,
        loop_file=LOOPS,
        echo_logs=False,
    ) as client:

        print("Initial state:")
        print(client.state)

        assert client.state["verts"] == 36007
        assert client.state["tets"] == 138399
        assert client.state["terminal"] == 0

        for loop_id in [0, 1, 2, 3]:
            result, state, actions = (
                client.step(loop_id)
            )

            expected = EXPECTED[loop_id]

            print()
            print(
                f"STEP {loop_id}:",
                result,
            )

            assert (
                result["verts_after"]
                == expected["verts_after"]
            )

            assert (
                result["tets_after"]
                == expected["tets_after"]
            )

            assert (
                result["converged"]
                == expected["converged"]
            )

        print()
        print("Terminal state:")
        print(client.state)

        assert client.state["converged"] == 1
        assert client.state["terminal"] == 1
        assert client.state["selection_success"] == 1
        assert client.state["available"] == 0
        assert client.actions == []

        final_result, final_state = (
            client.finalize(OUTPUT)
        )

        print()
        print("Final result:")
        print(final_result)

        assert final_result["hex"] == 88
        assert final_result["total_polys"] == 88
        assert final_result["full_hex"] == 1

        assert final_state["finalized"] == 1

    print()
    print("PASS: Python bridge regression succeeded.")


if __name__ == "__main__":
    main()