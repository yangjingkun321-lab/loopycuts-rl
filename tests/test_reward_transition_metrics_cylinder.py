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


from envs.loopycuts_env import LoopyCutsEnv
from rewards.transition_metrics import (
    extract_transition_metrics,
)


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

LOOP_FILE = (
    "/home/yjk/loopycuts_inputs/"
    "cylinder_plate_clean/"
    "cylinder_plate_rem_loop.txt"
)


def main():
    env = LoopyCutsEnv(
        executable=EXECUTABLE,
        mesh_file=MESH,
        loop_file=LOOP_FILE,
        echo_logs=False,
    )

    metrics = []

    try:
        observation, info = env.reset(
            seed=123
        )

        initial_verts = (
            env.current_state[
                "verts"
            ]
        )

        initial_tets = (
            env.current_state[
                "tets"
            ]
        )

        for action in (
            0,
            1,
            2,
            3,
        ):
            state_before = dict(
                env.current_state
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            state_after = dict(
                env.current_state
            )

            record = (
                extract_transition_metrics(
                    state_before=state_before,
                    step_result=(
                        info[
                            "step_result"
                        ]
                    ),
                    state_after=state_after,
                )
            )

            metrics.append(
                record
            )

            print()

            print(
                record
            )

            assert math.isclose(
                float(
                    reward
                ),
                float(
                    info[
                        "reward_breakdown"
                    ][
                        "total"
                    ]
                ),
                rel_tol=0.0,
                abs_tol=1e-15,
            )

            assert truncated is False

        assert len(metrics) == 4

        # ============================================================
        # Action outcomes
        # ============================================================

        assert [
            m.loop_id
            for m in metrics
        ] == [
            0,
            1,
            2,
            3,
        ]

        assert [
            m.committed
            for m in metrics
        ] == [
            1,
            1,
            1,
            1,
        ]

        assert [
            m.reverted
            for m in metrics
        ] == [
            0,
            0,
            0,
            0,
        ]

        # ============================================================
        # Convergence / phase
        # ============================================================

        assert [
            m.convergence_delta
            for m in metrics
        ] == [
            0,
            0,
            0,
            1,
        ]

        assert [
            m.first_convergence
            for m in metrics
        ] == [
            0,
            0,
            0,
            1,
        ]

        assert [
            m.phase_closed_this_step
            for m in metrics
        ] == [
            0,
            0,
            0,
            1,
        ]

        # ============================================================
        # Terminal outcome
        # ============================================================

        assert [
            m.terminal
            for m in metrics
        ] == [
            0,
            0,
            0,
            1,
        ]

        assert [
            m.selection_success
            for m in metrics
        ] == [
            0,
            0,
            0,
            1,
        ]

        assert [
            m.terminal_failure
            for m in metrics
        ] == [
            0,
            0,
            0,
            0,
        ]

        # ============================================================
        # Diagnostic deltas.
        #
        # Initial STATE has diagnostics_valid=0, therefore STEP 0
        # deliberately has no valid defect delta.
        #
        # After loop 2:
        #     buggy_chains = 3
        #
        # After loop 3:
        #     buggy_chains = 0
        # ============================================================

        assert [
            m.diagnostics_delta_valid
            for m in metrics
        ] == [
            0,
            1,
            1,
            1,
        ]

        assert math.isclose(
            metrics[0]
            .delta_log_buggy_chains,
            0.0,
            abs_tol=1e-12,
        )

        assert math.isclose(
            metrics[1]
            .delta_log_buggy_chains,
            0.0,
            abs_tol=1e-12,
        )

        assert math.isclose(
            metrics[2]
            .delta_log_buggy_chains,
            math.log1p(3),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        assert math.isclose(
            metrics[3]
            .delta_log_buggy_chains,
            -math.log1p(3),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        # ============================================================
        # Log-growth terms must telescope over the episode.
        # ============================================================

        final_verts = (
            env.current_state[
                "verts"
            ]
        )

        final_tets = (
            env.current_state[
                "tets"
            ]
        )

        sum_log_vert = sum(
            m.log_vert_growth
            for m in metrics
        )

        sum_log_tet = sum(
            m.log_tet_growth
            for m in metrics
        )

        assert math.isclose(
            sum_log_vert,
            math.log(
                final_verts
                /
                initial_verts
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        assert math.isclose(
            sum_log_tet,
            math.log(
                final_tets
                /
                initial_tets
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        # ============================================================
        # Action-set reduction.
        #
        # Last step closes the REGULAR phase, therefore its drop is
        # much larger and must NOT be interpreted as mate count.
        # ============================================================

        assert [
            m.available_drop
            for m in metrics
        ] == [
            1,
            1,
            1,
            62,
        ]

        print()

        print(
            "Episode totals:"
        )

        print(
            "steps:",
            sum(
                m.step_cost
                for m in metrics
            ),
        )

        print(
            "sum log tet growth:",
            sum_log_tet,
        )

        print(
            "sum log vert growth:",
            sum_log_vert,
        )

        print(
            "final/initial tets:",
            (
                final_tets
                /
                initial_tets
            ),
        )

        print(
            "final/initial verts:",
            (
                final_verts
                /
                initial_verts
            ),
        )

        print()

        print(
            "PASS: raw transition metrics "
            "match the known Cylinder Stage-2 trajectory."
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
