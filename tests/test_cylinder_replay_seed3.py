import filecmp
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
from policies.simple import ReplayPolicy
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

OUTPUT = Path(
    "/home/yjk/loopycuts_test/"
    "cylinder_plate/"
    "dynamic_policy_regression/"
    "random_seed3"
)

BASELINE_HEX = Path(
    "/home/yjk/loopycuts_test/"
    "cylinder_plate/"
    "runs/"
    "random_seed3/"
    "cylinder_plate_rem_splitted_hex.mesh"
)

OUTPUT_HEX = (
    OUTPUT /
    "cylinder_plate_rem_splitted_hex.mesh"
)


SEQUENCE = [
    0,
    1,
    31,
    3,
    7,
    23,
    22,
    51,
    29,
    43,
    34,
    8,
    19,
    57,
    9,
    33,
]


def main():
    policy = ReplayPolicy(
        SEQUENCE
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
            output_dir=OUTPUT,
            finalize=True,
        )

    trajectory = result["trajectory"]

    actions = [
        item["action"]
        for item in trajectory
    ]

    statuses = [
        item["step_result"]["status"]
        for item in trajectory
    ]

    committed_actions = [
        item["action"]
        for item in trajectory
        if item["step_result"]["committed"] == 1
    ]

    reverted_actions = [
        item["action"]
        for item in trajectory
        if item["step_result"]["reverted"] == 1
    ]

    max_tets = max(
        item["state"]["tets"]
        for item in trajectory
    )

    max_verts = max(
        item["state"]["verts"]
        for item in trajectory
    )

    print("Actions:")
    print(actions)

    print()
    print("Statuses:")
    print(statuses)

    print()
    print("Committed actions:")
    print(committed_actions)

    print()
    print("Reverted actions:")
    print(reverted_actions)

    print()
    print("Steps:")
    print(result["num_steps"])

    print()
    print("Max verts:")
    print(max_verts)

    print()
    print("Max tets:")
    print(max_tets)

    print()
    print("Selection success:")
    print(result["selection_success"])

    print()
    print("Final result:")
    print(result["final_result"])

    #
    # Regression checks against the previously verified
    # external-order random_seed3 trajectory.
    #
    assert actions == SEQUENCE

    assert result["num_steps"] == 16

    assert result["selection_success"] == 1
    assert result["converged"] == 1

    assert len(committed_actions) == 15

    assert reverted_actions == [22]

    assert max_verts == 79640
    assert max_tets == 372727

    assert result["final_result"]["hex"] == 472
    assert result["final_result"]["total_polys"] == 472
    assert result["final_result"]["full_hex"] == 1

    assert OUTPUT_HEX.exists()

    assert BASELINE_HEX.exists(), (
        f"Baseline file does not exist: "
        f"{BASELINE_HEX}"
    )

    assert filecmp.cmp(
        BASELINE_HEX,
        OUTPUT_HEX,
        shallow=False,
    )

    print()
    print(
        "PASS: ReplayPolicy random_seed3 "
        "matches the previous external-order "
        "result byte-for-byte."
    )


if __name__ == "__main__":
    main()