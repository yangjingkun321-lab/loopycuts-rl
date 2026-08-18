from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

from torch.optim import AdamW


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


from training.bc_weight_calibration_v1 import (
    CALIBRATION_RESULT_SCHEMA_VERSION,
    select_best_bc_weight,
)

from training.protocol_v1 import (
    PAPER_BATCH_SIZE,
    PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES,
    PROJECT_BC_WEIGHT_CALIBRATION_DEVICE,
    PROJECT_BC_WEIGHT_CALIBRATION_MODELS,
    PROJECT_BC_WEIGHT_CALIBRATION_SEEDS,
    PROJECT_BC_WEIGHT_CALIBRATION_STAGE1_GRADIENT_STEPS,
    PROJECT_INITIAL_ALPHA,
    PROJECT_MAIN_DEMO_EPISODES,
    PROJECT_MAIN_DEMO_TRANSITIONS,
)

from training.run_bc_weight_calibration_v1 import (
    RUNNER_VERSION,
    CalibrationRunnerError,
    assert_protocol,
    atomic_write_json,
    build_stage1_algorithm,
    formal_pair_output_name,
    load_formal_grid_from_pair_artifacts,
    make_actor_critic_adamw_factory,
    make_alpha_adam_factory,
)


def synthetic_pair_payload(
    *,
    bc_weight,
    seed,
    git_commit,
):
    evaluations = []

    for model in (
        PROJECT_BC_WEIGHT_CALIBRATION_MODELS
    ):
        evaluations.append(
            {
                "bc_weight":
                    float(
                        bc_weight
                    ),

                "seed":
                    int(
                        seed
                    ),

                "model":
                    model,

                "outcome":
                    "FULL_HEX",

                "episode_return":
                    1.0,

                "final_hex":
                    100,

                "final_total_polys":
                    100,
            }
        )

    return {
        "runner_version":
            RUNNER_VERSION,

        "calibration_version":
            "bc_weight_calibration_v2",

        "result_schema_version":
            CALIBRATION_RESULT_SCHEMA_VERSION,

        "run_kind":
            "FORMAL_BC_WEIGHT_CALIBRATION_PAIR",

        "selector_eligible":
            True,

        "formal_grid_member":
            True,

        "git_commit":
            git_commit,

        "bc_weight":
            float(
                bc_weight
            ),

        "seed":
            int(
                seed
            ),

        "device":
            "cpu",

        "demo_provenance":
            {
                "episodes":
                    PROJECT_MAIN_DEMO_EPISODES,

                "transitions":
                    PROJECT_MAIN_DEMO_TRANSITIONS,

                "random_seed":
                    int(
                        seed
                    ),
            },

        "stage1":
            {
                "gradient_updates":
                    PROJECT_BC_WEIGHT_CALIBRATION_STAGE1_GRADIENT_STEPS,

                "sampled_demo_transitions":
                    (
                        PROJECT_BC_WEIGHT_CALIBRATION_STAGE1_GRADIENT_STEPS
                        *
                        PAPER_BATCH_SIZE
                    ),
            },

        "evaluations":
            evaluations,
    }


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


    # ============================================================
    # Frozen Stage-I construction.
    # ============================================================

    smoke_lambda = 2.0

    assert smoke_lambda not in (
        PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES
    )

    actor_critic_factory = (
        make_actor_critic_adamw_factory(
            lr=0.001
        )
    )

    assert (
        actor_critic_factory.optim_class
        is AdamW
    )

    assert math.isclose(
        actor_critic_factory.kwargs[
            "lr"
        ],
        0.001,
    )

    assert (
        actor_critic_factory.kwargs[
            "betas"
        ]
        ==
        (
            0.99,
            0.999,
        )
    )

    assert math.isclose(
        actor_critic_factory.kwargs[
            "eps"
        ],
        1.0e-8,
    )

    assert math.isclose(
        actor_critic_factory.kwargs[
            "weight_decay"
        ],
        0.01,
    )

    alpha_factory = (
        make_alpha_adam_factory(
            lr=3.0e-4
        )
    )

    assert math.isclose(
        alpha_factory.lr,
        3.0e-4,
    )

    assert (
        alpha_factory.betas
        ==
        (
            0.9,
            0.999,
        )
    )

    assert math.isclose(
        alpha_factory.weight_decay,
        0.0,
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


    # ============================================================
    # Frozen formal filenames.
    # ============================================================

    names = {
        formal_pair_output_name(
            bc_weight=
                bc_weight,

            seed=
                seed,
        )
        for bc_weight in
        PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES
        for seed in
        PROJECT_BC_WEIGHT_CALIBRATION_SEEDS
    }

    assert len(
        names
    ) == 15

    assert (
        "formal_lambda_0p1_seed_42.json"
        in
        names
    )

    assert (
        "formal_lambda_3_seed_44.json"
        in
        names
    )


    # ============================================================
    # Synthetic complete 15-pair / 75-row grid.
    # ============================================================

    synthetic_commit = (
        "synthetic-test-commit"
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(
            tmp
        )

        for bc_weight in (
            PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES
        ):
            for seed in (
                PROJECT_BC_WEIGHT_CALIBRATION_SEEDS
            ):
                path = (
                    root
                    /
                    formal_pair_output_name(
                        bc_weight=
                            bc_weight,

                        seed=
                            seed,
                    )
                )

                atomic_write_json(
                    path=
                        path,

                    payload=
                        synthetic_pair_payload(
                            bc_weight=
                                bc_weight,

                            seed=
                                seed,

                            git_commit=
                                synthetic_commit,
                        ),
                )

        (
            rows,
            artifact_records,
        ) = load_formal_grid_from_pair_artifacts(
            output_root=
                root,

            expected_git_commit=
                synthetic_commit,
        )

        assert len(
            artifact_records
        ) == 15

        assert len(
            rows
        ) == 75

        winner, summaries = (
            select_best_bc_weight(
                rows
            )
        )

        # All synthetic candidates are exactly tied,
        # therefore the frozen final tie-break chooses
        # the smaller lambda.
        assert math.isclose(
            winner.bc_weight,
            0.1,
        )

        assert len(
            summaries
        ) == 5


        # --------------------------------------------------------
        # Existing formal artifact must never be overwritten.
        # --------------------------------------------------------

        first_path = (
            root
            /
            formal_pair_output_name(
                bc_weight=
                    0.1,

                seed=
                    42,
            )
        )

        try:
            atomic_write_json(
                path=
                    first_path,

                payload=
                    {},
            )

        except CalibrationRunnerError:
            pass

        else:
            raise AssertionError(
                "Formal artifact overwrite "
                "was unexpectedly accepted"
            )


        # --------------------------------------------------------
        # Incomplete formal grid must hard-fail.
        # --------------------------------------------------------

        missing_path = (
            root
            /
            formal_pair_output_name(
                bc_weight=
                    3.0,

                seed=
                    44,
            )
        )

        missing_path.unlink()

        try:
            load_formal_grid_from_pair_artifacts(
                output_root=
                    root,

                expected_git_commit=
                    synthetic_commit,
            )

        except CalibrationRunnerError:
            pass

        else:
            raise AssertionError(
                "Incomplete formal pair grid "
                "was unexpectedly accepted"
            )


    print(
        "PASS: BC calibration runner supports "
        "atomic formal pairs and strict 75-row selection"
    )


if __name__ == "__main__":
    main()
