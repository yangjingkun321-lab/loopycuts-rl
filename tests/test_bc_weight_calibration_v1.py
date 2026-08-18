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


from training.bc_weight_calibration_v1 import (
    BCWeightCalibrationError,
    CalibrationEpisodeResult,
    CandidateSummary,
    expected_result_keys,
    select_best_bc_weight,
    validate_complete_result_grid,
)

from training.protocol_v1 import (
    PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES,
    PROJECT_BC_WEIGHT_CALIBRATION_MODELS,
    PROJECT_BC_WEIGHT_CALIBRATION_SEEDS,
)


def make_complete_grid():
    rows = []

    for bc_weight in (
        PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES
    ):
        for seed in (
            PROJECT_BC_WEIGHT_CALIBRATION_SEEDS
        ):
            for model in (
                PROJECT_BC_WEIGHT_CALIBRATION_MODELS
            ):
                rows.append(
                    CalibrationEpisodeResult(
                        bc_weight=
                            bc_weight,

                        seed=
                            seed,

                        model=
                            model,

                        outcome=
                            "FULL_HEX",

                        episode_return=
                            1.0,

                        final_hex=
                            100,

                        final_total_polys=
                            100,
                    )
                )

    return rows


def main():
    # ------------------------------------------------------------
    # Frozen grid size.
    # ------------------------------------------------------------

    expected = (
        expected_result_keys()
    )

    assert len(
        expected
    ) == 75

    rows = (
        make_complete_grid()
    )

    validated = (
        validate_complete_result_grid(
            rows
        )
    )

    assert len(
        validated
    ) == 75


    # ------------------------------------------------------------
    # Completely tied candidates:
    # lower lambda must win.
    # ------------------------------------------------------------

    winner, summaries = (
        select_best_bc_weight(
            rows
        )
    )

    assert math.isclose(
        winner.bc_weight,
        0.1,
    )

    assert len(
        summaries
    ) == 5


    # ------------------------------------------------------------
    # Exact lexicographic precedence.
    # ------------------------------------------------------------

    # More FULL_HEX dominates everything below it.
    a = CandidateSummary(
        bc_weight=3.0,
        episodes=15,
        full_hex_count=15,
        finalization_crash_count=5,
        aggregate_nonhex_fraction=0.9,
        mean_episode_return=-100.0,
    )

    b = CandidateSummary(
        bc_weight=0.1,
        episodes=15,
        full_hex_count=14,
        finalization_crash_count=0,
        aggregate_nonhex_fraction=0.0,
        mean_episode_return=100.0,
    )

    assert (
        a.selection_key()
        <
        b.selection_key()
    )


    # Same FULL_HEX: fewer crashes wins.
    a = CandidateSummary(
        bc_weight=3.0,
        episodes=15,
        full_hex_count=10,
        finalization_crash_count=0,
        aggregate_nonhex_fraction=0.9,
        mean_episode_return=-100.0,
    )

    b = CandidateSummary(
        bc_weight=0.1,
        episodes=15,
        full_hex_count=10,
        finalization_crash_count=1,
        aggregate_nonhex_fraction=0.0,
        mean_episode_return=100.0,
    )

    assert (
        a.selection_key()
        <
        b.selection_key()
    )


    # Same FULL_HEX/crash:
    # lower non-hex fraction wins.
    a = CandidateSummary(
        bc_weight=3.0,
        episodes=15,
        full_hex_count=10,
        finalization_crash_count=0,
        aggregate_nonhex_fraction=0.1,
        mean_episode_return=-100.0,
    )

    b = CandidateSummary(
        bc_weight=0.1,
        episodes=15,
        full_hex_count=10,
        finalization_crash_count=0,
        aggregate_nonhex_fraction=0.2,
        mean_episode_return=100.0,
    )

    assert (
        a.selection_key()
        <
        b.selection_key()
    )


    # Same above metrics:
    # higher mean return wins.
    a = CandidateSummary(
        bc_weight=3.0,
        episodes=15,
        full_hex_count=10,
        finalization_crash_count=0,
        aggregate_nonhex_fraction=0.1,
        mean_episode_return=2.0,
    )

    b = CandidateSummary(
        bc_weight=0.1,
        episodes=15,
        full_hex_count=10,
        finalization_crash_count=0,
        aggregate_nonhex_fraction=0.1,
        mean_episode_return=1.0,
    )

    assert (
        a.selection_key()
        <
        b.selection_key()
    )


    # Complete tie:
    # smaller lambda wins.
    a = CandidateSummary(
        bc_weight=0.1,
        episodes=15,
        full_hex_count=10,
        finalization_crash_count=0,
        aggregate_nonhex_fraction=0.1,
        mean_episode_return=1.0,
    )

    b = CandidateSummary(
        bc_weight=0.3,
        episodes=15,
        full_hex_count=10,
        finalization_crash_count=0,
        aggregate_nonhex_fraction=0.1,
        mean_episode_return=1.0,
    )

    assert (
        a.selection_key()
        <
        b.selection_key()
    )


    # ------------------------------------------------------------
    # Incomplete grid must hard-fail.
    # ------------------------------------------------------------

    incomplete = rows[:-1]

    try:
        validate_complete_result_grid(
            incomplete
        )

    except BCWeightCalibrationError:
        pass

    else:
        raise AssertionError(
            "Incomplete calibration grid "
            "was unexpectedly accepted"
        )


    # ------------------------------------------------------------
    # Duplicate grid row must hard-fail.
    # ------------------------------------------------------------

    duplicate = (
        rows
        +
        [
            rows[0]
        ]
    )

    try:
        validate_complete_result_grid(
            duplicate
        )

    except BCWeightCalibrationError:
        pass

    else:
        raise AssertionError(
            "Duplicate calibration row "
            "was unexpectedly accepted"
        )


    print(
        "PASS: BC-weight calibration selector "
        "implements the frozen lexicographic rule"
    )


if __name__ == "__main__":
    main()
