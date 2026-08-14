import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(
    "/home/yjk/codes/loopycuts_rl"
)

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
    "/home/yjk/codes/LoopyCuts/"
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


def assert_common_structure(
    observation,
):
    outer_obs = observation[
        "obs"
    ]

    global_features = (
        outer_obs[
            "global"
        ]
    )

    loop_features = (
        outer_obs[
            "loops"
        ]
    )

    exists = outer_obs[
        "exists"
    ]

    mask = observation[
        "mask"
    ]

    assert global_features.shape == (
        GLOBAL_DIM,
    )

    assert loop_features.shape == (
        MAX_LOOPS,
        LOOP_FEATURE_DIM,
    )

    assert exists.shape == (
        MAX_LOOPS,
    )

    assert mask.shape == (
        MAX_LOOPS,
    )

    assert global_features.dtype == (
        np.float32
    )

    assert loop_features.dtype == (
        np.float32
    )

    assert exists.dtype == (
        np.bool_
    )

    assert mask.dtype == (
        np.bool_
    )

    assert np.isfinite(
        global_features
    ).all()

    assert np.isfinite(
        loop_features
    ).all()


def main():
    metadata = parse_loop_metadata(
        LOOP_FILE
    )

    executed = set()

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

        # ============================================================
        # Initial observation
        # ============================================================

        observation = builder.build(
            state=client.state,
            actions=client.actions,
            used=client.used,
            reverted=client.reverted,
            nico_bug=client.nico_bug,
            top_relevant=client.top_relevant,
            executed=executed,
        )

        assert_common_structure(
            observation
        )

        global_features = (
            observation[
                "obs"
            ][
                "global"
            ]
        )

        loop_features = (
            observation[
                "obs"
            ][
                "loops"
            ]
        )

        exists = (
            observation[
                "obs"
            ][
                "exists"
            ]
        )

        mask = observation[
            "mask"
        ]

        print(
            "Initial global:"
        )

        print(
            global_features
        )

        print(
            "initial mask count:",
            int(
                mask.sum()
            ),
        )

        # ------------------------------------------------------------
        # Exists != legal.
        # ------------------------------------------------------------

        assert int(
            exists.sum()
        ) == 91

        assert bool(
            exists[
                :91
            ].all()
        )

        assert not bool(
            exists[
                91:
            ].any()
        )

        assert int(
            mask.sum()
        ) == 65

        assert bool(
            mask[
                :65
            ].all()
        )

        assert not bool(
            mask[
                65:
            ].any()
        )

        # ------------------------------------------------------------
        # Padding rows must be exactly zero.
        # ------------------------------------------------------------

        assert np.array_equal(
            loop_features[
                91:
            ],
            np.zeros(
                (
                    MAX_LOOPS - 91,
                    LOOP_FEATURE_DIM,
                ),
                dtype=np.float32,
            ),
        )

        # ------------------------------------------------------------
        # Initial global features.
        #
        # 2 CONCAVE + 63 REGULAR = 65 legal.
        # ------------------------------------------------------------

        assert math.isclose(
            float(
                global_features[0]
            ),
            0.0,
            abs_tol=1e-7,
        )

        assert math.isclose(
            float(
                global_features[1]
            ),
            1.0,
            abs_tol=1e-7,
        )

        assert math.isclose(
            float(
                global_features[2]
            ),
            2.0 / 65.0,
            abs_tol=1e-7,
        )

        assert math.isclose(
            float(
                global_features[3]
            ),
            63.0 / 65.0,
            abs_tol=1e-7,
        )

        assert float(
            global_features[4]
        ) == 0.0

        assert float(
            global_features[5]
        ) == 0.0

        assert math.isclose(
            float(
                global_features[6]
            ),
            0.0,
            abs_tol=1e-7,
        )

        assert math.isclose(
            float(
                global_features[7]
            ),
            0.0,
            abs_tol=1e-7,
        )

        assert np.array_equal(
            global_features[
                8:
            ],
            np.zeros(
                8,
                dtype=np.float32,
            ),
        )

        # ------------------------------------------------------------
        # Selected static/dynamic per-loop rows.
        # ------------------------------------------------------------

        #
        # loop 0: serialized CONCAVE, sharp, legal.
        #
        assert np.array_equal(
            loop_features[
                0,
                1:4,
            ],
            np.asarray(
                [
                    1.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
        )

        assert float(
            loop_features[
                0,
                4,
            ]
        ) == 1.0

        assert float(
            loop_features[
                0,
                5,
            ]
        ) == 0.0

        assert math.isclose(
            float(
                loop_features[
                    0,
                    7,
                ]
            ),
            1.0,
            abs_tol=1e-7,
        )

        assert float(
            loop_features[
                0,
                8,
            ]
        ) == 1.0

        assert float(
            loop_features[
                0,
                9,
            ]
        ) == 0.0

        assert float(
            loop_features[
                0,
                11,
            ]
        ) == 0.0

        #
        # loop 2: REGULAR and legal.
        #
        assert np.array_equal(
            loop_features[
                2,
                1:4,
            ],
            np.asarray(
                [
                    0.0,
                    1.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
        )

        assert float(
            loop_features[
                2,
                7,
            ]
        ) == 0.0

        assert float(
            loop_features[
                2,
                8,
            ]
        ) == 1.0

        #
        # loop 65: exists, CONVEX, but not legal.
        #
        assert bool(
            exists[
                65
            ]
        )

        assert np.array_equal(
            loop_features[
                65,
                1:4,
            ],
            np.asarray(
                [
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
        )

        assert float(
            loop_features[
                65,
                8,
            ]
        ) == 0.0

        #
        # loop 66: real loop, C++ detected Nico_bug, still not padding.
        #
        assert bool(
            exists[
                66
            ]
        )

        assert float(
            loop_features[
                66,
                12,
            ]
        ) == 1.0

        # ============================================================
        # STEP 0
        # ============================================================

        step_result, _, _ = (
            client.step(
                0
            )
        )

        assert (
            step_result[
                "status"
            ]
            ==
            "COMMITTED"
        )

        executed.add(
            0
        )

        observation = builder.build(
            state=client.state,
            actions=client.actions,
            used=client.used,
            reverted=client.reverted,
            nico_bug=client.nico_bug,
            top_relevant=client.top_relevant,
            executed=executed,
        )

        assert_common_structure(
            observation
        )

        global_features = (
            observation[
                "obs"
            ][
                "global"
            ]
        )

        loop_features = (
            observation[
                "obs"
            ][
                "loops"
            ]
        )

        mask = observation[
            "mask"
        ]

        print()

        print(
            "After STEP 0 global:"
        )

        print(
            global_features
        )

        assert int(
            mask.sum()
        ) == 64

        assert not bool(
            mask[
                0
            ]
        )

        assert bool(
            mask[
                1
            ]
        )

        assert math.isclose(
            float(
                global_features[0]
            ),
            1.0 / 65.0,
            abs_tol=1e-7,
        )

        assert math.isclose(
            float(
                global_features[1]
            ),
            64.0 / 65.0,
            abs_tol=1e-7,
        )

        assert math.isclose(
            float(
                global_features[2]
            ),
            1.0 / 65.0,
            abs_tol=1e-7,
        )

        assert math.isclose(
            float(
                global_features[3]
            ),
            63.0 / 65.0,
            abs_tol=1e-7,
        )

        assert float(
            global_features[6]
        ) > 0.0

        assert float(
            global_features[7]
        ) > 0.0

        assert float(
            global_features[12]
        ) == 1.0

        #
        # loop 0:
        # legal=0, used=1, reverted=0, executed=1.
        #
        assert np.array_equal(
            loop_features[
                0,
                8:12,
            ],
            np.asarray(
                [
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
        )

        # ============================================================
        # STEP 1 and STEP 2:
        # after loop 2, Cylinder is known to have buggy_chains=3.
        # ============================================================

        for loop_id in (
            1,
            2,
        ):
            step_result, _, _ = (
                client.step(
                    loop_id
                )
            )

            assert (
                step_result[
                    "status"
                ]
                ==
                "COMMITTED"
            )

            executed.add(
                loop_id
            )

        observation = builder.build(
            state=client.state,
            actions=client.actions,
            used=client.used,
            reverted=client.reverted,
            nico_bug=client.nico_bug,
            top_relevant=client.top_relevant,
            executed=executed,
        )

        global_features = (
            observation[
                "obs"
            ][
                "global"
            ]
        )

        print()

        print(
            "After STEP 2 global:"
        )

        print(
            global_features
        )

        assert float(
            global_features[12]
        ) == 1.0

        assert math.isclose(
            float(
                global_features[15]
            ),
            math.log1p(
                3
            ),
            rel_tol=0.0,
            abs_tol=1e-6,
        )

        # ============================================================
        # STEP 3 -> terminal.
        # ============================================================

        step_result, _, _ = (
            client.step(
                3
            )
        )

        assert (
            step_result[
                "status"
            ]
            ==
            "COMMITTED"
        )

        executed.add(
            3
        )

        observation = builder.build(
            state=client.state,
            actions=client.actions,
            used=client.used,
            reverted=client.reverted,
            nico_bug=client.nico_bug,
            top_relevant=client.top_relevant,
            executed=executed,
        )

        assert_common_structure(
            observation
        )

        global_features = (
            observation[
                "obs"
            ][
                "global"
            ]
        )

        loop_features = (
            observation[
                "obs"
            ][
                "loops"
            ]
        )

        mask = observation[
            "mask"
        ]

        print()

        print(
            "Terminal global:"
        )

        print(
            global_features
        )

        print(
            "terminal mask count:",
            int(
                mask.sum()
            ),
        )

        assert int(
            client.state[
                "terminal"
            ]
        ) == 1

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

        assert not bool(
            mask.any()
        )

        assert float(
            global_features[4]
        ) == 1.0

        assert float(
            global_features[5]
        ) == 1.0

        assert math.isclose(
            float(
                global_features[1]
            ),
            0.0,
            abs_tol=1e-7,
        )

        #
        # 0,1,2,3 were all really executed.
        #
        assert np.array_equal(
            loop_features[
                0:4,
                11,
            ],
            np.ones(
                4,
                dtype=np.float32,
            ),
        )

        #
        # Padding remains exactly zero even at terminal.
        #
        assert np.array_equal(
            loop_features[
                91:
            ],
            np.zeros(
                (
                    MAX_LOOPS - 91,
                    LOOP_FEATURE_DIM,
                ),
                dtype=np.float32,
            ),
        )

    print()

    print(
        "PASS: Observation Builder preserves "
        "original loop IDs, padding, dynamic state, "
        "diagnostics, and terminal all-False mask."
    )


if __name__ == "__main__":
    main()
