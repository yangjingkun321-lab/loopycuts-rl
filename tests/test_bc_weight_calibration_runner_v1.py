from __future__ import annotations

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


from training.protocol_v1 import (
    PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES,
    PROJECT_BC_WEIGHT_CALIBRATION_DEVICE,
    PROJECT_BC_WEIGHT_CALIBRATION_STAGE1_GRADIENT_STEPS,
    PROJECT_INITIAL_ALPHA,
)

from training.run_bc_weight_calibration_v1 import (
    assert_protocol,
    build_stage1_algorithm,
    make_adam_factory,
)


def main():
    assert_protocol()

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_DEVICE
        ==
        "cpu"
    )

    assert (
        PROJECT_BC_WEIGHT_CALIBRATION_STAGE1_GRADIENT_STEPS
        ==
        782
    )

    # Smoke lambda must deliberately stay outside
    # the formal candidate set.
    smoke_lambda = 2.0

    assert smoke_lambda not in (
        PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES
    )

    factory = (
        make_adam_factory(
            lr=0.001
        )
    )

    assert math.isclose(
        factory.lr,
        0.001,
    )

    algorithm, policy, auto_alpha = (
        build_stage1_algorithm(
            bc_weight=
                smoke_lambda,

            seed=
                42,
        )
    )

    assert algorithm.bc_enabled is True

    assert math.isclose(
        algorithm.bc_weight,
        smoke_lambda,
    )

    assert (
        policy.deterministic_eval
        is True
    )

    assert math.isclose(
        policy.exploration_epsilon,
        0.0,
    )

    assert math.isclose(
        auto_alpha.value,
        PROJECT_INITIAL_ALPHA,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )

    print(
        "PASS: BC calibration runner constructs "
        "the frozen Stage-I SAC configuration"
    )


if __name__ == "__main__":
    main()
