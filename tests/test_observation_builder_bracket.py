import math
import sys
from pathlib import Path

import numpy as np


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
from dataset_tools.loop_metadata import parse_loop_metadata
from observation.builder import (
    GLOBAL_DIM,
    LOOP_FEATURE_DIM,
    MAX_LOOPS,
    LoopyCutsObservationBuilder,
)


EXECUTABLE = (
    "/home/yjk/codes/LoopyCuts_v5/"
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


def check_shape(observation):
    assert (
        observation["obs"]["global"].shape
        ==
        (GLOBAL_DIM,)
    )

    assert (
        observation["obs"]["loops"].shape
        ==
        (
            MAX_LOOPS,
            LOOP_FEATURE_DIM,
        )
    )

    assert (
        observation["obs"]["exists"].shape
        ==
        (MAX_LOOPS,)
    )

    assert (
        observation["mask"].shape
        ==
        (MAX_LOOPS,)
    )

    assert (
        observation["obs"]["global"].dtype
        ==
        np.float32
    )

    assert (
        observation["obs"]["loops"].dtype
        ==
        np.float32
    )

    assert (
        observation["obs"]["exists"].dtype
        ==
        np.bool_
    )

    assert (
        observation["mask"].dtype
        ==
        np.bool_
    )


def build_observation(
    builder,
    client,
    executed,
):
    observation = builder.build(
        state=client.state,
        actions=client.actions,
        used=client.used,
        reverted=client.reverted,
        nico_bug=client.nico_bug,
        top_relevant=client.top_relevant,
        executed=executed,
    )

    check_shape(
        observation
    )

    return observation


def main():
    metadata = parse_loop_metadata(
        LOOP_FILE
    )

    print(
        "serialized loops:",
        len(metadata),
    )

    assert len(metadata) == 130

    executed = set()
    actual_actions = []

    with LoopyCutsClient(
        executable=EXECUTABLE,
        mesh_file=MESH,
        loop_file=LOOP_FILE,
        echo_logs=False,
    ) as client:

        initial_state = dict(
            client.state
        )

        initial_actions = list(
            client.actions
        )

        builder = (
            LoopyCutsObservationBuilder(
                metadata=metadata,
                initial_state=initial_state,
                initial_actions=initial_actions,
            )
        )

        print(
            "initial actionable:",
            builder.initial_actionable_count,
        )

        initial_observation = (
            build_observation(
                builder,
                client,
                executed,
            )
        )

        assert int(
            initial_observation[
                "obs"
            ][
                "exists"
            ].sum()
        ) == 130

        assert not bool(
            initial_observation[
                "obs"
            ][
                "exists"
            ][130:].any()
        )

        # ============================================================
        # Follow the frozen V1 "original forward order":
        #
        #     always choose the smallest currently legal original ID.
        #
        # Before convergence this gives 0..28.
        # After first convergence REGULAR is permanently closed,
        # therefore the next legal IDs are 81..89.
        # ============================================================

        observation_after_28 = None
        observation_after_87 = None
        terminal_observation = None

        while not int(
            client.state[
                "terminal"
            ]
        ):
            assert client.actions

            action = min(
                client.actions
            )

            actual_actions.append(
                action
            )

            (
                step_result,
                _,
                _,
            ) = client.step(
                action
            )

            executed.add(
                action
            )

            observation = (
                build_observation(
                    builder,
                    client,
                    executed,
                )
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
                f"{client.state['regular_phase_closed']} "
                f"available="
                f"{client.state['available']}"
            )

            if action == 28:
                observation_after_28 = (
                    observation
                )

                # -----------------------------------------------
                # First convergence.
                # REGULAR phase must close permanently.
                # -----------------------------------------------

                assert int(
                    client.state[
                        "converged"
                    ]
                ) == 1

                assert int(
                    client.state[
                        "regular_phase_closed"
                    ]
                ) == 1

                assert client.actions == (
                    list(
                        range(
                            81,
                            90,
                        )
                    )
                )

                mask = observation[
                    "mask"
                ]

                assert int(
                    mask.sum()
                ) == 9

                assert bool(
                    mask[
                        81:90
                    ].all()
                )

                assert not bool(
                    mask[
                        :81
                    ].any()
                )

                # -----------------------------------------------
                # A skipped REGULAR loop still exists but has
                # become permanently illegal.
                #
                # loop 29:
                #     serialized REGULAR
                #     exists=1
                #     legal=0
                #     executed=0
                # -----------------------------------------------

                loops = observation[
                    "obs"
                ][
                    "loops"
                ]

                exists = observation[
                    "obs"
                ][
                    "exists"
                ]

                assert bool(
                    exists[29]
                )

                assert float(
                    loops[
                        29,
                        2,
                    ]
                ) == 1.0

                assert float(
                    loops[
                        29,
                        8,
                    ]
                ) == 0.0

                assert float(
                    loops[
                        29,
                        11,
                    ]
                ) == 0.0

            if action == 87:
                observation_after_87 = (
                    observation
                )

                # -----------------------------------------------
                # This CONCAVE action destroys current
                # convergence, but REGULAR phase MUST remain
                # closed.
                # -----------------------------------------------

                assert int(
                    client.state[
                        "converged"
                    ]
                ) == 0

                assert int(
                    client.state[
                        "regular_phase_closed"
                    ]
                ) == 1

                assert int(
                    client.state[
                        "terminal"
                    ]
                ) == 0

                assert client.actions == [
                    88,
                    89,
                ]

                mask = observation[
                    "mask"
                ]

                assert int(
                    mask.sum()
                ) == 2

                assert bool(
                    mask[88]
                )

                assert bool(
                    mask[89]
                )

                assert not bool(
                    mask[29]
                )

                global_features = (
                    observation[
                        "obs"
                    ][
                        "global"
                    ]
                )

                assert float(
                    global_features[4]
                ) == 0.0

                assert float(
                    global_features[5]
                ) == 1.0

                assert math.isclose(
                    float(
                        global_features[1]
                    ),
                    (
                        2.0
                        /
                        builder.initial_actionable_count
                    ),
                    rel_tol=0.0,
                    abs_tol=1e-7,
                )

            if action == 89:
                terminal_observation = (
                    observation
                )

        # ============================================================
        # Exact forward traversal regression.
        # ============================================================

        assert (
            actual_actions
            ==
            EXPECTED_ACTION_SEQUENCE
        )

        assert (
            observation_after_28
            is not None
        )

        assert (
            observation_after_87
            is not None
        )

        assert (
            terminal_observation
            is not None
        )

        # ============================================================
        # Terminal failure:
        #
        #     no legal actions
        #     but NOT converged
        #
        # This is the second legitimate terminal form, distinct from
        # Cylinder's terminal+converged success.
        # ============================================================

        assert int(
            client.state[
                "terminal"
            ]
        ) == 1

        assert int(
            client.state[
                "converged"
            ]
        ) == 0

        assert int(
            client.state[
                "regular_phase_closed"
            ]
        ) == 1

        assert int(
            client.state[
                "selection_success"
            ]
        ) == 0

        assert client.actions == []

        mask = terminal_observation[
            "mask"
        ]

        assert not bool(
            mask.any()
        )

        global_features = (
            terminal_observation[
                "obs"
            ][
                "global"
            ]
        )

        assert float(
            global_features[1]
        ) == 0.0

        assert float(
            global_features[4]
        ) == 0.0

        assert float(
            global_features[5]
        ) == 1.0

        # ============================================================
        # Existing, skipped REGULAR loops must still be represented.
        # They are not padding and were never executed.
        # ============================================================

        loops = terminal_observation[
            "obs"
        ][
            "loops"
        ]

        exists = terminal_observation[
            "obs"
        ][
            "exists"
        ]

        assert bool(
            exists[29]
        )

        assert float(
            loops[
                29,
                2,
            ]
        ) == 1.0

        assert float(
            loops[
                29,
                8,
            ]
        ) == 0.0

        assert float(
            loops[
                29,
                11,
            ]
        ) == 0.0

        # ============================================================
        # All actually selected IDs must have executed=1.
        # ============================================================

        for loop_id in (
            actual_actions
        ):
            assert float(
                loops[
                    loop_id,
                    11,
                ]
            ) == 1.0

        # ============================================================
        # Padding from 130 to MAX_LOOPS remains zero.
        # ============================================================

        assert not bool(
            exists[
                130:
            ].any()
        )

        assert np.array_equal(
            loops[
                130:
            ],
            np.zeros(
                (
                    MAX_LOOPS - 130,
                    LOOP_FEATURE_DIM,
                ),
                dtype=np.float32,
            ),
        )

        print()

        print(
            "===================================="
        )

        print(
            "BRACKET OBSERVATION TERMINAL"
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
            client.state[
                "step"
            ],
        )

        print(
            "converged:",
            client.state[
                "converged"
            ],
        )

        print(
            "regular_phase_closed:",
            client.state[
                "regular_phase_closed"
            ],
        )

        print(
            "terminal:",
            client.state[
                "terminal"
            ],
        )

        print(
            "selection_success:",
            client.state[
                "selection_success"
            ],
        )

        print(
            "terminal mask count:",
            int(
                mask.sum()
            ),
        )

    print()

    print(
        "PASS: Observation Builder handles "
        "terminal + non-converged Bracket state "
        "without reopening REGULAR loops."
    )


if __name__ == "__main__":
    main()
