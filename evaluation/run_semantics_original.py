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
from policies.simple import OriginalOrderPolicy


EXECUTABLE = (
    "/home/yjk/codes/LoopyCuts/"
    "volumetric_cutter/"
    "volumetric_cutter"
)


MODELS = {
    "BracketInches": {
        "mesh": (
            "/home/yjk/codes/LoopyCuts/"
            "test_data/BracketInches/"
            "BracketInches_rem_rem_splitted.obj"
        ),
        "loops": (
            "/home/yjk/codes/LoopyCuts/"
            "test_data/BracketInches/"
            "BracketInches_rem_rem_loop.txt"
        ),
    },

    "deckel": {
        "mesh": (
            "/home/yjk/codes/LoopyCuts/"
            "test_data/deckel/"
            "deckel_rem_splitted.obj"
        ),
        "loops": (
            "/home/yjk/codes/LoopyCuts/"
            "test_data/deckel/"
            "deckel_rem_loop.txt"
        ),
    },

    "eraser_ball": {
        "mesh": (
            "/home/yjk/codes/LoopyCuts/"
            "test_data/eraser_ball/"
            "eraser_ball_rem_rem_splitted.obj"
        ),
        "loops": (
            "/home/yjk/codes/LoopyCuts/"
            "test_data/eraser_ball/"
            "eraser_ball_rem_rem_loop.txt"
        ),
    },

    "bimba": {
        "mesh": (
            "/home/yjk/codes/LoopyCuts/"
            "test_data/bimba/"
            "bimba_rem_splitted.obj"
        ),
        "loops": (
            "/home/yjk/codes/LoopyCuts/"
            "test_data/bimba/"
            "bimba_rem_loop.txt"
        ),
    },
}


OUTPUT_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "semantics_original"
)


def run_probe(
    model_name,
    verbose=False,
    max_steps=10000,
):
    cfg = MODELS[model_name]

    output_dir = (
        OUTPUT_ROOT /
        model_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy = OriginalOrderPolicy()
    policy.reset()

    trajectory = []

    first_convergence = None

    post_convergence_concaves = []

    convergence_broken_by_concave = False

    stopped_on_semantic_event = False

    committed = 0
    reverted = 0

    with LoopyCutsClient(
        executable=EXECUTABLE,
        mesh_file=cfg["mesh"],
        loop_file=cfg["loops"],
        echo_logs=False,
    ) as client:

        initial_state = dict(
            client.state
        )

        max_verts = initial_state["verts"]
        max_tets = initial_state["tets"]

        while not client.state["terminal"]:

            if len(trajectory) >= max_steps:
                raise RuntimeError(
                    f"{model_name}: exceeded "
                    f"max_steps={max_steps}"
                )

            actions = list(
                client.actions
            )

            if not actions:
                raise RuntimeError(
                    f"{model_name}: "
                    "non-terminal state has "
                    "no legal actions"
                )

            #
            # Dynamic form of Stage1 original order:
            # choose the currently legal loop with
            # the smallest original loop ID.
            #
            action = policy.select(
                client.state,
                actions,
            )

            step_result, state, next_actions = (
                client.step(action)
            )

            loop_type = step_result[
                "loop_type"
            ]

            conv_before = step_result[
                "converged_before"
            ]

            conv_after = step_result[
                "converged"
            ]

            if step_result["committed"] == 1:
                committed += 1

            if step_result["reverted"] == 1:
                reverted += 1

            max_verts = max(
                max_verts,
                state["verts"],
            )

            max_tets = max(
                max_tets,
                state["tets"],
            )

            record = {
                "step": step_result["step"],
                "loop_id": action,
                "loop_type": loop_type,
                "status":
                    step_result["status"],
                "committed":
                    step_result["committed"],
                "reverted":
                    step_result["reverted"],
                "converged_before":
                    conv_before,
                "converged_after":
                    conv_after,
                "verts":
                    state["verts"],
                "tets":
                    state["tets"],
                "available_after":
                    state["available"],
            }

            trajectory.append(
                record
            )

            #
            # First 0 -> 1 convergence transition.
            #
            if (
                first_convergence is None
                and conv_before == 0
                and conv_after == 1
            ):
                first_convergence = {
                    "step":
                        step_result["step"],
                    "loop_id":
                        action,
                    "loop_type":
                        loop_type,
                    "verts":
                        state["verts"],
                    "tets":
                        state["tets"],
                }

                print(
                    f"[{model_name}] "
                    "FIRST_CONVERGENCE "
                    f"step={step_result['step']} "
                    f"loop={action} "
                    f"type={loop_type} "
                    f"tets={state['tets']}"
                )

            #
            # A CONCAVE executed after the mesh was
            # already converged.
            #
            if (
                loop_type == "CONCAVE"
                and conv_before == 1
            ):
                event = {
                    "step":
                        step_result["step"],
                    "loop_id":
                        action,
                    "status":
                        step_result["status"],
                    "converged_before":
                        conv_before,
                    "converged_after":
                        conv_after,
                    "verts":
                        state["verts"],
                    "tets":
                        state["tets"],
                }

                post_convergence_concaves.append(
                    event
                )

                print(
                    f"[{model_name}] "
                    "POST_CONVERGENCE_CONCAVE "
                    f"step={step_result['step']} "
                    f"loop={action} "
                    f"conv={conv_before}"
                    f"->{conv_after}"
                )

                #
                # This is the exact semantic event
                # we are currently testing.
                #
                # STOP immediately so that the current
                # RL server does not reopen old REGULAR
                # actions and silently diverge from the
                # original forward-only Batch traversal.
                #
                if conv_after == 0:
                    convergence_broken_by_concave = True
                    stopped_on_semantic_event = True

                    print(
                        f"[{model_name}] "
                        "CONVERGENCE_BROKEN_BY_CONCAVE "
                        f"step={step_result['step']} "
                        f"loop={action}"
                    )

                    break

            if verbose:
                print(
                    f"[{model_name}] "
                    f"step={step_result['step']} "
                    f"id={action} "
                    f"type={loop_type} "
                    f"status={step_result['status']} "
                    f"conv={conv_before}->{conv_after} "
                    f"tets={state['tets']}"
                )

        final_probe_state = dict(
            client.state
        )

        remaining_actions = list(
            client.actions
        )

    summary = {
        "model": model_name,

        "mesh": cfg["mesh"],
        "loops_file": cfg["loops"],

        "initial_state":
            initial_state,

        "steps":
            len(trajectory),

        "actions": [
            x["loop_id"]
            for x in trajectory
        ],

        "types": [
            x["loop_type"]
            for x in trajectory
        ],

        "statuses": [
            x["status"]
            for x in trajectory
        ],

        "committed":
            committed,

        "reverted":
            reverted,

        "max_verts":
            max_verts,

        "max_tets":
            max_tets,

        "first_convergence":
            first_convergence,

        "post_convergence_concaves":
            post_convergence_concaves,

        "num_post_convergence_concaves":
            len(
                post_convergence_concaves
            ),

        "convergence_broken_by_concave":
            convergence_broken_by_concave,

        "stopped_on_semantic_event":
            stopped_on_semantic_event,

        "terminal_at_stop":
            final_probe_state["terminal"],

        "converged_at_stop":
            final_probe_state["converged"],

        "selection_success_at_stop":
            final_probe_state[
                "selection_success"
            ],

        "available_at_stop":
            final_probe_state["available"],

        "remaining_actions_at_stop":
            remaining_actions,

        "trajectory":
            trajectory,
    }

    summary_file = (
        output_dir /
        "semantics_summary.json"
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
        "========================================"
    )
    print(
        f"SEMANTICS SUMMARY: {model_name}"
    )
    print(
        "========================================"
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
        "max_tets:",
        summary["max_tets"],
    )

    print(
        "first_convergence:",
        summary["first_convergence"],
    )

    print(
        "post_convergence_concaves:",
        summary[
            "num_post_convergence_concaves"
        ],
    )

    print(
        "convergence_broken_by_concave:",
        summary[
            "convergence_broken_by_concave"
        ],
    )

    print(
        "stopped_on_semantic_event:",
        summary[
            "stopped_on_semantic_event"
        ],
    )

    print(
        "terminal_at_stop:",
        summary[
            "terminal_at_stop"
        ],
    )

    print(
        "converged_at_stop:",
        summary[
            "converged_at_stop"
        ],
    )

    print(
        "available_at_stop:",
        summary[
            "available_at_stop"
        ],
    )

    print(
        "saved:",
        summary_file,
    )

    print()

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        required=True,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
    )

    args = parser.parse_args()

    run_probe(
        model_name=args.model,
        verbose=args.verbose,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()