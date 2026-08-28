from __future__ import annotations

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
        str(
            PROJECT_ROOT
        ),
    )


from bridge.cpp_client import (
    RLServerProcessError,
)

from finalization.outcome import (
    OUTCOME_CRASH,
    OUTCOME_FULL_HEX,
    OUTCOME_NON_FULL_HEX,
)

from finalization.quality_outcome_v1 import (
    QualityFinalizationOutcomeError,
    evaluate_terminal_quality_finalization,
)


MECH02_QUALITY = {
    "model":
        "mechanical02",

    "hex":
        880,

    "total_polys":
        880,

    "nonhex":
        0,

    "d_c":
        1.0,

    "q_missing":
        0.99853225619739827,

    "q_spurious":
        0.99861570853523696,

    "q_shape":
        0.99853225619739827,

    "sharp_active":
        1,

    "sharp_metrics_valid":
        1,

    "q_sharp":
        0.99019972262228984,

    "q_fidelity":
        0.988746363116073,
}


BONE_QUALITY = {
    "model":
        "bone_femur",

    "hex":
        2102,

    "total_polys":
        2106,

    "nonhex":
        4,

    "d_c":
        0.88780577166826335,

    "q_missing":
        0.9974300309629075,

    "q_spurious":
        0.99804809503350089,

    "q_shape":
        0.9974300309629075,

    "sharp_active":
        0,

    "sharp_metrics_valid":
        0,

    "q_sharp":
        "NA",

    "q_fidelity":
        0.9974300309629075,
}


class FakeClient:
    def __init__(
        self,
        *,
        final_result=None,
        quality_record=None,
        final_state=None,
        error=None,
    ):
        self.state = {
            "terminal":
                1,

            "finalized":
                0,
        }

        self.final_result = (
            final_result
        )

        self.quality_record = (
            quality_record
        )

        self.final_state = (
            final_state
        )

        self.error = (
            error
        )

        self.calls = []

    def finalize_quality(
        self,
        quality_ref_path,
    ):
        self.calls.append(
            quality_ref_path
        )

        if self.error is not None:
            raise self.error

        return (
            dict(
                self.final_result
            ),
            dict(
                self.quality_record
            ),
            dict(
                self.final_state
            ),
        )


# ================================================================
# FULL_HEX
# ================================================================

client = FakeClient(
    final_result={
        "hex":
            880,

        "total_polys":
            880,

        "full_hex":
            1,
    },

    quality_record=
        MECH02_QUALITY,

    final_state={
        "terminal":
            1,

        "finalized":
            1,
    },
)


outcome = (
    evaluate_terminal_quality_finalization(
        client,
        quality_ref_path=
            "/tmp/mechanical02.quality_ref_v1",

        expected_model=
            "mechanical02",
    )
)


assert outcome.outcome == OUTCOME_FULL_HEX
assert outcome.completed is True
assert outcome.crashed is False

assert outcome.final_hex == 880
assert outcome.final_total_polys == 880
assert outcome.full_hex == 1

assert outcome.terminal_quality is not None

assert (
    outcome.terminal_quality.model
    ==
    "mechanical02"
)

assert (
    outcome.terminal_quality.q_fidelity
    ==
    0.988746363116073
)


# ================================================================
# NON_FULL_HEX
# ================================================================

client = FakeClient(
    final_result={
        "hex":
            2102,

        "total_polys":
            2106,

        "full_hex":
            0,
    },

    quality_record=
        BONE_QUALITY,

    final_state={
        "terminal":
            1,

        "finalized":
            1,
    },
)


outcome = (
    evaluate_terminal_quality_finalization(
        client,
        quality_ref_path=
            "/tmp/bone_femur.quality_ref_v1",

        expected_model=
            "bone_femur",
    )
)


assert (
    outcome.outcome
    ==
    OUTCOME_NON_FULL_HEX
)

assert outcome.final_hex == 2102
assert outcome.final_total_polys == 2106
assert outcome.full_hex == 0

assert (
    outcome.terminal_quality.nonhex
    ==
    4
)


# ================================================================
# Actual C++ process crash retains frozen crash taxonomy.
# ================================================================

crash = RLServerProcessError(
    phase=
        "FINALIZE_QUALITY",

    return_code=
        -6,

    expected_prefix=
        "[RL] ACTIONS",

    lines=[
        "partial finalization log",
    ],
)


client = FakeClient(
    error=
        crash,
)


outcome = (
    evaluate_terminal_quality_finalization(
        client,
        quality_ref_path=
            "/tmp/crash.quality_ref_v1",

        expected_model=
            "crash_model",
    )
)


assert outcome.outcome == OUTCOME_CRASH
assert outcome.completed is False
assert outcome.crashed is True

assert outcome.return_code == -6
assert outcome.signal_number == 6
assert outcome.terminal_quality is None


# ================================================================
# Model identity must fail closed.
# ================================================================

client = FakeClient(
    final_result={
        "hex":
            880,

        "total_polys":
            880,

        "full_hex":
            1,
    },

    quality_record=
        MECH02_QUALITY,

    final_state={
        "terminal":
            1,

        "finalized":
            1,
    },
)


try:
    evaluate_terminal_quality_finalization(
        client,
        quality_ref_path=
            "/tmp/wrong.quality_ref_v1",

        expected_model=
            "not_mechanical02",
    )

except QualityFinalizationOutcomeError:
    pass

else:
    raise AssertionError(
        "model mismatch was accepted"
    )


# ================================================================
# Independent protocol records must agree.
# ================================================================

client = FakeClient(
    final_result={
        "hex":
            879,

        "total_polys":
            880,

        "full_hex":
            0,
    },

    quality_record=
        MECH02_QUALITY,

    final_state={
        "terminal":
            1,

        "finalized":
            1,
    },
)


try:
    evaluate_terminal_quality_finalization(
        client,
        quality_ref_path=
            "/tmp/mechanical02.quality_ref_v1",

        expected_model=
            "mechanical02",
    )

except QualityFinalizationOutcomeError:
    pass

else:
    raise AssertionError(
        "FINAL_RESULT/quality mismatch was accepted"
    )


print(
    "PASS: FULL_HEX quality finalization binds all protocol records"
)

print(
    "PASS: NON_FULL_HEX quality finalization binds all protocol records"
)

print(
    "PASS: real C++ finalization crash retains frozen crash taxonomy"
)

print(
    "PASS: quality-ref model mismatch fails closed"
)

print(
    "PASS: FINAL_RESULT/FINALIZE_QUALITY disagreement fails closed"
)
