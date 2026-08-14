import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    "/home/yjk/codes/loopycuts_rl"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from envs.loopycuts_env import LoopyCutsEnv


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

LOOP_FILE = (
    "/home/yjk/codes/LoopyCuts/"
    "test_data/BracketInches/"
    "BracketInches_rem_rem_loop.txt"
)


EXPECTED_ACTION_SEQUENCE = (
    list(range(29))
    +
    list(range(81, 90))
)


def main():
    env = LoopyCutsEnv(
        executable=EXECUTABLE,
        mesh_file=MESH,
        loop_file=LOOP_FILE,
        echo_logs=False,
    )

    try:
        observation, info = env.reset(
            seed=123
        )

        print(
            "Initial legal count:",
            len(env.legal_actions),
        )

        print(
            "Initial mask count:",
            int(observation["mask"].sum()),
        )

        assert (
            env.observation_space.contains(
                observation
            )
        )

        assert (
            info["reward_is_placeholder"]
            is False
        )

        assert (
            len(env.legal_actions)
            ==
            90
        )

        assert (
            int(observation["mask"].sum())
            ==
            90
        )

        actual_actions = []

        # ============================================================
        # Always choose the minimum authoritative legal loop ID.
        # ============================================================

        while not int(
            env.current_state[
                "terminal"
            ]
        ):
            assert env.legal_actions

            action = min(
                env.legal_actions
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

            actual_actions.append(
                action
            )

            step_result = (
                info[
                    "step_result"
                ]
            )

            print(
                f"step={step_result['step']:2d} "
                f"id={action:3d} "
                f"type={step_result['loop_type']:8s} "
                f"status={step_result['status']:9s} "
                f"conv="
                f"{step_result['converged_before']}"
                f"->{step_result['converged']} "
                f"phase="
                f"{env.current_state['regular_phase_closed']} "
                f"available="
                f"{len(env.legal_actions)} "
                f"terminated={terminated}"
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

            assert (
                info[
                    "reward_is_placeholder"
                ]
                is False
            )

            assert (
                env.observation_space.contains(
                    observation
                )
            )

            assert (
                action
                in env.executed_loop_ids
            )

            # --------------------------------------------------------
            # First convergence occurs after loop 28.
            # --------------------------------------------------------

            if action == 28:
                assert (
                    env.current_state[
                        "converged"
                    ]
                    ==
                    1
                )

                assert (
                    env.current_state[
                        "regular_phase_closed"
                    ]
                    ==
                    1
                )

                assert (
                    env.legal_actions
                    ==
                    tuple(
                        range(
                            81,
                            90,
                        )
                    )
                )

                assert (
                    int(
                        observation[
                            "mask"
                        ].sum()
                    )
                    ==
                    9
                )

                #
                # loop 29 is an existing REGULAR loop but is now
                # permanently illegal under frozen V1 semantics.
                #
                assert bool(
                    observation[
                        "obs"
                    ][
                        "exists"
                    ][29]
                )

                assert (
                    float(
                        observation[
                            "obs"
                        ][
                            "loops"
                        ][
                            29,
                            2,
                        ]
                    )
                    ==
                    1.0
                )

                assert (
                    float(
                        observation[
                            "obs"
                        ][
                            "loops"
                        ][
                            29,
                            8,
                        ]
                    )
                    ==
                    0.0
                )

                assert (
                    float(
                        observation[
                            "obs"
                        ][
                            "loops"
                        ][
                            29,
                            11,
                        ]
                    )
                    ==
                    0.0
                )

                # ----------------------------------------------------
                # Environment must reject a closed REGULAR action
                # BEFORE sending anything to the C++ server.
                # ----------------------------------------------------

                step_before = (
                    env.current_state[
                        "step"
                    ]
                )

                legal_before = (
                    env.legal_actions
                )

                try:
                    env.step(
                        29
                    )

                except ValueError:
                    pass

                else:
                    raise AssertionError(
                        "Closed REGULAR loop 29 "
                        "was incorrectly accepted"
                    )

                assert (
                    env.current_state[
                        "step"
                    ]
                    ==
                    step_before
                )

                assert (
                    env.legal_actions
                    ==
                    legal_before
                )

            # --------------------------------------------------------
            # loop 87 destroys current convergence.
            #
            # REGULAR phase MUST remain closed.
            # --------------------------------------------------------

            if action == 87:
                assert (
                    env.current_state[
                        "converged"
                    ]
                    ==
                    0
                )

                assert (
                    env.current_state[
                        "regular_phase_closed"
                    ]
                    ==
                    1
                )

                assert (
                    env.current_state[
                        "terminal"
                    ]
                    ==
                    0
                )

                assert (
                    env.legal_actions
                    ==
                    (
                        88,
                        89,
                    )
                )

                assert (
                    int(
                        observation[
                            "mask"
                        ].sum()
                    )
                    ==
                    2
                )

                #
                # REGULAR 29 must remain closed even though
                # converged changed 1 -> 0.
                #
                assert not bool(
                    observation[
                        "mask"
                    ][29]
                )

            # --------------------------------------------------------
            # Only the final loop should terminate the episode.
            # --------------------------------------------------------

            if action != 89:
                assert terminated is False

            else:
                assert terminated is True

        # ============================================================
        # Exact traversal regression.
        # ============================================================

        assert (
            actual_actions
            ==
            EXPECTED_ACTION_SEQUENCE
        )

        # ============================================================
        # Valid terminal FAILURE state.
        # ============================================================

        assert (
            env.current_state[
                "step"
            ]
            ==
            38
        )

        assert (
            env.current_state[
                "terminal"
            ]
            ==
            1
        )

        assert (
            env.current_state[
                "converged"
            ]
            ==
            0
        )

        assert (
            env.current_state[
                "regular_phase_closed"
            ]
            ==
            1
        )

        assert (
            env.current_state[
                "selection_success"
            ]
            ==
            0
        )

        assert (
            env.legal_actions
            ==
            ()
        )

        assert not bool(
            observation[
                "mask"
            ].any()
        )

        assert (
            len(
                env.executed_loop_ids
            )
            ==
            38
        )

        #
        # loop 29 still exists but was never executed.
        #
        assert bool(
            observation[
                "obs"
            ][
                "exists"
            ][29]
        )

        assert (
            float(
                observation[
                    "obs"
                ][
                    "loops"
                ][
                    29,
                    11,
                ]
            )
            ==
            0.0
        )

        # ============================================================
        # step() after terminal must fail at the Gym environment layer.
        # ============================================================

        try:
            env.step(
                88
            )

        except RuntimeError:
            pass

        else:
            raise AssertionError(
                "step() after Bracket terminal "
                "did not raise RuntimeError"
            )

        print()

        print(
            "===================================="
        )

        print(
            "BRACKET ENV TERMINAL"
        )

        print(
            "===================================="
        )

        print(
            "actions:",
            actual_actions,
        )

        print(
            "steps:",
            env.current_state[
                "step"
            ],
        )

        print(
            "converged:",
            env.current_state[
                "converged"
            ],
        )

        print(
            "regular_phase_closed:",
            env.current_state[
                "regular_phase_closed"
            ],
        )

        print(
            "terminal:",
            env.current_state[
                "terminal"
            ],
        )

        print(
            "selection_success:",
            env.current_state[
                "selection_success"
            ],
        )

        print(
            "terminal mask count:",
            int(
                observation[
                    "mask"
                ].sum()
            ),
        )

        print()

        print(
            "PASS: LoopyCutsEnv preserves "
            "Bracket terminal/non-converged "
            "V1 Stage-2 semantics."
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
