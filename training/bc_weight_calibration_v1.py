from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


from training.protocol_v1 import (
    PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES,
    PROJECT_BC_WEIGHT_CALIBRATION_MODELS,
    PROJECT_BC_WEIGHT_CALIBRATION_SEEDS,
)


CALIBRATION_RESULT_SCHEMA_VERSION = (
    "bc_weight_calibration_result_v1"
)

VALID_OUTCOMES = {
    "FULL_HEX",
    "NON_FULL_HEX",
    "FINALIZATION_CRASH",
}


class BCWeightCalibrationError(
    RuntimeError
):
    pass


@dataclass(
    frozen=True
)
class CalibrationEpisodeResult:
    bc_weight: float
    seed: int
    model: str

    outcome: str

    episode_return: float

    final_hex: int | None
    final_total_polys: int | None


@dataclass(
    frozen=True
)
class CandidateSummary:
    bc_weight: float

    episodes: int

    full_hex_count: int
    finalization_crash_count: int

    aggregate_nonhex_fraction: float
    mean_episode_return: float

    def selection_key(
        self,
    ):
        """
        Frozen lexicographic selection rule:

            1. MAX FULL_HEX
            2. MIN FINALIZATION_CRASH
            3. MIN aggregate non-hex fraction
            4. MAX mean episode return
            5. MIN BC weight
        """

        return (
            -int(
                self.full_hex_count
            ),
            int(
                self.finalization_crash_count
            ),
            float(
                self.aggregate_nonhex_fraction
            ),
            -float(
                self.mean_episode_return
            ),
            float(
                self.bc_weight
            ),
        )


def expected_result_keys():
    return {
        (
            float(
                bc_weight
            ),
            int(
                seed
            ),
            str(
                model
            ),
        )
        for bc_weight in
        PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES
        for seed in
        PROJECT_BC_WEIGHT_CALIBRATION_SEEDS
        for model in
        PROJECT_BC_WEIGHT_CALIBRATION_MODELS
    }


def validate_episode_result(
    row: CalibrationEpisodeResult,
):
    if (
        not math.isfinite(
            float(
                row.bc_weight
            )
        )
        or
        float(
            row.bc_weight
        )
        <=
        0.0
    ):
        raise BCWeightCalibrationError(
            "bc_weight must be finite and positive"
        )

    if not math.isfinite(
        float(
            row.episode_return
        )
    ):
        raise BCWeightCalibrationError(
            "episode_return must be finite"
        )

    if (
        row.outcome
        not in
        VALID_OUTCOMES
    ):
        raise BCWeightCalibrationError(
            f"invalid outcome: {row.outcome!r}"
        )

    if (
        row.outcome
        ==
        "FINALIZATION_CRASH"
    ):
        return

    if (
        row.final_hex
        is None
        or
        row.final_total_polys
        is None
    ):
        raise BCWeightCalibrationError(
            "completed finalization requires "
            "final_hex and final_total_polys"
        )

    final_hex = int(
        row.final_hex
    )

    final_total = int(
        row.final_total_polys
    )

    if final_total <= 0:
        raise BCWeightCalibrationError(
            "final_total_polys must be positive"
        )

    if (
        final_hex < 0
        or
        final_hex > final_total
    ):
        raise BCWeightCalibrationError(
            "invalid final hex/poly counts"
        )

    if (
        row.outcome
        ==
        "FULL_HEX"
        and
        final_hex != final_total
    ):
        raise BCWeightCalibrationError(
            "FULL_HEX requires "
            "final_hex == final_total_polys"
        )

    if (
        row.outcome
        ==
        "NON_FULL_HEX"
        and
        final_hex >= final_total
    ):
        raise BCWeightCalibrationError(
            "NON_FULL_HEX requires "
            "final_hex < final_total_polys"
        )


def validate_complete_result_grid(
    rows: Iterable[
        CalibrationEpisodeResult
    ],
):
    rows = tuple(
        rows
    )

    expected = (
        expected_result_keys()
    )

    observed = set()

    for row in rows:
        validate_episode_result(
            row
        )

        key = (
            float(
                row.bc_weight
            ),
            int(
                row.seed
            ),
            str(
                row.model
            ),
        )

        if key not in expected:
            raise BCWeightCalibrationError(
                "result is outside the frozen "
                f"calibration grid: {key}"
            )

        if key in observed:
            raise BCWeightCalibrationError(
                "duplicate calibration result: "
                f"{key}"
            )

        observed.add(
            key
        )

    missing = (
        expected
        -
        observed
    )

    if missing:
        raise BCWeightCalibrationError(
            "calibration result grid is incomplete; "
            f"missing={len(missing)}"
        )

    extra = (
        observed
        -
        expected
    )

    if extra:
        raise BCWeightCalibrationError(
            "calibration result grid has "
            f"unexpected rows: {extra}"
        )

    return rows


def summarize_candidate(
    rows: Iterable[
        CalibrationEpisodeResult
    ],
    *,
    bc_weight: float,
):
    selected = tuple(
        row
        for row in rows
        if math.isclose(
            float(
                row.bc_weight
            ),
            float(
                bc_weight
            ),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )

    expected_episode_count = (
        len(
            PROJECT_BC_WEIGHT_CALIBRATION_SEEDS
        )
        *
        len(
            PROJECT_BC_WEIGHT_CALIBRATION_MODELS
        )
    )

    if (
        len(
            selected
        )
        !=
        expected_episode_count
    ):
        raise BCWeightCalibrationError(
            f"candidate {bc_weight} has "
            f"{len(selected)} episodes; expected "
            f"{expected_episode_count}"
        )

    full_hex_count = sum(
        row.outcome
        ==
        "FULL_HEX"
        for row in selected
    )

    crash_count = sum(
        row.outcome
        ==
        "FINALIZATION_CRASH"
        for row in selected
    )

    completed = tuple(
        row
        for row in selected
        if row.outcome
        !=
        "FINALIZATION_CRASH"
    )

    nonhex_numerator = sum(
        int(
            row.final_total_polys
        )
        -
        int(
            row.final_hex
        )
        for row in completed
    )

    nonhex_denominator = sum(
        int(
            row.final_total_polys
        )
        for row in completed
    )

    if nonhex_denominator > 0:
        aggregate_nonhex_fraction = (
            float(
                nonhex_numerator
            )
            /
            float(
                nonhex_denominator
            )
        )
    else:
        aggregate_nonhex_fraction = (
            float("inf")
        )

    mean_episode_return = (
        sum(
            float(
                row.episode_return
            )
            for row in selected
        )
        /
        len(
            selected
        )
    )

    return CandidateSummary(
        bc_weight=
            float(
                bc_weight
            ),

        episodes=
            len(
                selected
            ),

        full_hex_count=
            int(
                full_hex_count
            ),

        finalization_crash_count=
            int(
                crash_count
            ),

        aggregate_nonhex_fraction=
            float(
                aggregate_nonhex_fraction
            ),

        mean_episode_return=
            float(
                mean_episode_return
            ),
    )


def summarize_all_candidates(
    rows: Iterable[
        CalibrationEpisodeResult
    ],
):
    rows = (
        validate_complete_result_grid(
            rows
        )
    )

    return tuple(
        summarize_candidate(
            rows,
            bc_weight=
                bc_weight,
        )
        for bc_weight in
        PROJECT_BC_WEIGHT_CALIBRATION_CANDIDATES
    )


def select_best_bc_weight(
    rows: Iterable[
        CalibrationEpisodeResult
    ],
):
    summaries = (
        summarize_all_candidates(
            rows
        )
    )

    winner = min(
        summaries,
        key=lambda summary:
            summary.selection_key(),
    )

    return (
        winner,
        summaries,
    )
