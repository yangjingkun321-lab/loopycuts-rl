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
from policies.simple import OriginalOrderPolicy
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
    "original"
)

BASELINE_HEX = Path(
    "/home/yjk/loopycuts_test/"
    "cylinder_plate/"
    "runs/"
    "batch_baseline/"
    "cylinder_plate_rem_splitted_hex.mesh"
)

OUTPUT_HEX = (
    OUTPUT /
    "cylinder_plate_rem_splitted_hex.mesh"
)


def main():
    policy = OriginalOrderPolicy()

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

    actions = [
        item["action"]
        for item in result["trajectory"]
    ]

    print("Actions:")
    print(actions)

    print()
    print("Steps:")
    print(result["num_steps"])

    print()
    print("Selection success:")
    print(result["selection_success"])

    print()
    print("Final result:")
    print(result["final_result"])

    assert actions == [0, 1, 2, 3]

    assert result["num_steps"] == 4
    assert result["selection_success"] == 1
    assert result["converged"] == 1

    assert result["final_result"]["hex"] == 88
    assert result["final_result"]["total_polys"] == 88
    assert result["final_result"]["full_hex"] == 1

    assert OUTPUT_HEX.exists()

    assert filecmp.cmp(
        BASELINE_HEX,
        OUTPUT_HEX,
        shallow=False,
    )

    print()
    print(
        "PASS: Dynamic original-order policy "
        "matches Batch byte-for-byte."
    )


if __name__ == "__main__":
    main()