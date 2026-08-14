import argparse
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
from policies.simple import RandomPolicy
from evaluation.episode_runner import run_episode


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


def summarize(result):
    trajectory = result["trajectory"]

    actions = [
        x["action"]
        for x in trajectory
    ]

    committed = [
        x["action"]
        for x in trajectory
        if x["step_result"]["committed"] == 1
    ]

    reverted = [
        x["action"]
        for x in trajectory
        if x["step_result"]["reverted"] == 1
    ]

    statuses = [
        x["step_result"]["status"]
        for x in trajectory
    ]

    if trajectory:
        max_verts = max(
            x["state"]["verts"]
            for x in trajectory
        )

        max_tets = max(
            x["state"]["tets"]
            for x in trajectory
        )

        sum_step_time = sum(
            x["step_result"]["step_time"]
            for x in trajectory
        )
    else:
        max_verts = 0
        max_tets = 0
        sum_step_time = 0.0

    return {
        "actions": actions,
        "statuses": statuses,
        "steps": result["num_steps"],
        "committed": len(committed),
        "reverted": len(reverted),
        "reverted_actions": reverted,
        "max_verts": max_verts,
        "max_tets": max_tets,
        "sum_step_time": sum_step_time,
        "converged": result["converged"],
        "terminal": result["terminal"],
        "selection_success":
            result["selection_success"],
        "final_result":
            result["final_result"],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    seed = args.seed

    output_dir = Path(
        "/home/yjk/loopycuts_test/"
        "cylinder_plate/"
        "dynamic_random"
    ) / f"seed_{seed}"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy = RandomPolicy(
        seed=seed
    )

    with LoopyCutsClient(
        executable=EXECUTABLE,
        mesh_file=MESH,
        loop_file=LOOPS,
        echo_logs=False,
    ) as client:

        result = run_episode(
            client=client,
            policy=policy,
            output_dir=output_dir,
            finalize=True,
        )

    summary = summarize(result)

    print()
    print("====================================")
    print(f"Dynamic Random Policy -- seed {seed}")
    print("====================================")

    for key, value in summary.items():
        print(f"{key}: {value}")

    summary_file = (
        output_dir /
        "summary.json"
    )

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved summary to: {summary_file}"
    )


if __name__ == "__main__":
    main()