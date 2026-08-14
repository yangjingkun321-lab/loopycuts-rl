import json
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
from policies.simple import OriginalOrderPolicy


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

OUTPUT = Path(
    "/home/yjk/loopycuts_test/"
    "semantics_original/BracketInches/"
    "dynamic_reopen"
)


def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy = OriginalOrderPolicy()

    trajectory = []

    first_convergence = None
    convergence_losses = []
    convergence_recoveries = []

    with LoopyCutsClient(
        executable=EXECUTABLE,
        mesh_file=MESH,
        loop_file=LOOPS,
        echo_logs=False,
    ) as client:

        policy.reset()

        initial_tets = client.state["tets"]

        max_tets = initial_tets
        max_verts = client.state["verts"]

        while not client.state["terminal"]:

            actions = list(client.actions)

            if not actions:
                raise RuntimeError(
                    "Non-terminal state has "
                    "no legal actions"
                )

            action = policy.select(
                client.state,
                actions,
            )

            result, state, next_actions = (
                client.step(action)
            )

            before = result[
                "converged_before"
            ]

            after = result[
                "converged"
            ]

            record = {
                "step": result["step"],
                "loop_id": action,
                "loop_type":
                    result["loop_type"],
                "status":
                    result["status"],
                "converged_before":
                    before,
                "converged_after":
                    after,
                "available_after":
                    state["available"],
                "verts":
                    state["verts"],
                "tets":
                    state["tets"],
            }

            trajectory.append(record)

            max_tets = max(
                max_tets,
                state["tets"],
            )

            max_verts = max(
                max_verts,
                state["verts"],
            )

            if (
                first_convergence is None
                and before == 0
                and after == 1
            ):
                first_convergence = record

                print(
                    "FIRST_CONVERGENCE:",
                    record,
                )

            if before == 1 and after == 0:
                convergence_losses.append(
                    record
                )

                print(
                    "CONVERGENCE_LOST:",
                    record,
                )

                print(
                    "ACTIONS_REOPENED:",
                    next_actions,
                )

            if (
                first_convergence is not None
                and before == 0
                and after == 1
                and record is not first_convergence
            ):
                convergence_recoveries.append(
                    record
                )

                print(
                    "CONVERGENCE_RECOVERED:",
                    record,
                )

            #
            # Print the interesting Bracket region.
            #
            if (
                action >= 81
                or convergence_losses
            ):
                print(
                    f"step={result['step']:3d} "
                    f"id={action:3d} "
                    f"type={result['loop_type']:8s} "
                    f"status={result['status']:9s} "
                    f"conv={before}->{after} "
                    f"available={state['available']} "
                    f"tets={state['tets']}"
                )

        summary = {
            "steps":
                len(trajectory),

            "actions": [
                x["loop_id"]
                for x in trajectory
            ],

            "committed":
                sum(
                    x["status"] == "COMMITTED"
                    for x in trajectory
                ),

            "reverted":
                sum(
                    x["status"] == "REVERTED"
                    for x in trajectory
                ),

            "first_convergence":
                first_convergence,

            "convergence_losses":
                convergence_losses,

            "convergence_recoveries":
                convergence_recoveries,

            "max_verts":
                max_verts,

            "max_tets":
                max_tets,

            "final_state":
                dict(client.state),

            "trajectory":
                trajectory,
        }

    path = (
        OUTPUT /
        "dynamic_reopen_summary.json"
    )

    path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "===================================="
    )
    print(
        "DYNAMIC REOPEN SUMMARY"
    )
    print(
        "===================================="
    )

    print(
        "steps:",
        summary["steps"],
    )

    print(
        "committed:",
        summary["committed"],
    )

    print(
        "reverted:",
        summary["reverted"],
    )

    print(
        "first convergence:",
        summary["first_convergence"],
    )

    print(
        "convergence losses:",
        len(
            summary["convergence_losses"]
        ),
    )

    print(
        "convergence recoveries:",
        len(
            summary["convergence_recoveries"]
        ),
    )

    print(
        "max tets:",
        summary["max_tets"],
    )

    print(
        "final state:",
        summary["final_state"],
    )

    print(
        "saved:",
        path,
    )


if __name__ == "__main__":
    main()