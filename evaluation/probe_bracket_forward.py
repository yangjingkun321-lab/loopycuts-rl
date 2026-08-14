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
    "/home/yjk/codes/LoopyCuts/"
    "test_data/BracketInches/"
    "BracketInches_rem_rem_splitted.obj"
)

LOOPS = (
    "/home/yjk/codes/LoopyCuts/"
    "test_data/BracketInches/"
    "BracketInches_rem_rem_loop.txt"
)


def main():

    cursor = -1
    trajectory = []

    with LoopyCutsClient(
        executable=EXECUTABLE,
        mesh_file=MESH,
        loop_file=LOOPS,
        echo_logs=False,
    ) as client:

        while True:

            actions = list(client.actions)

            #
            # Emulate the original forward-only for-loop:
            # never return to an ID that the traversal
            # has already passed.
            #
            forward_actions = [
                a
                for a in actions
                if a > cursor
            ]

            if not forward_actions:
                break

            action = min(forward_actions)

            result, state, actions_after = (
                client.step(action)
            )

            cursor = action

            record = {
                "step": result["step"],
                "loop_id": action,
                "loop_type":
                    result["loop_type"],
                "status":
                    result["status"],
                "conv_before":
                    result["converged_before"],
                "conv_after":
                    result["converged"],
                "available_after":
                    state["available"],
                "tets":
                    state["tets"],
            }

            trajectory.append(record)

            #
            # Print only convergence-related events
            # and the rear CONCAVE block.
            #
            if (
                result["converged_before"]
                != result["converged"]
                or action >= 81
            ):
                print(
                    f"step={result['step']:3d} "
                    f"id={action:3d} "
                    f"type={result['loop_type']:8s} "
                    f"status={result['status']:9s} "
                    f"conv="
                    f"{result['converged_before']}"
                    f"->{result['converged']} "
                    f"available="
                    f"{state['available']} "
                    f"tets={state['tets']}"
                )

        print()
        print(
            "===================================="
        )
        print(
            "FORWARD-ONLY PROBE END"
        )
        print(
            "===================================="
        )

        print("cursor:", cursor)

        print(
            "server converged:",
            client.state["converged"],
        )

        print(
            "server terminal:",
            client.state["terminal"],
        )

        print(
            "server available:",
            client.state["available"],
        )

        print(
            "all available actions:",
            client.actions,
        )

        print()

        print(
            "Executed rear concaves:"
        )

        for x in trajectory:
            if x["loop_id"] >= 81:
                print(x)


if __name__ == "__main__":
    main()