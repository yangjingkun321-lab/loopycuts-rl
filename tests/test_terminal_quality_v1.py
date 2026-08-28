from __future__ import annotations

import struct
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


from finalization.terminal_quality_v1 import (
    TerminalQualityError,
    parse_terminal_quality_facts,
)


def bits(value):
    return struct.unpack(
        "=Q",
        struct.pack(
            "=d",
            float(value),
        ),
    )[0]


CASES = [
    {
        "model": "mechanical02",
        "hex": 880,
        "total_polys": 880,
        "nonhex": 0,
        "d_c": 1.0,
        "q_missing": 0.99853225619739827,
        "q_spurious": 0.99861570853523696,
        "q_shape": 0.99853225619739827,
        "sharp_active": 1,
        "sharp_metrics_valid": 1,
        "q_sharp": 0.99019972262228984,
        "q_fidelity": 0.988746363116073,
    },

    {
        "model": "cactus",
        "hex": 534,
        "total_polys": 534,
        "nonhex": 0,
        "d_c": 1.0,
        "q_missing": 0.21984521141416016,
        "q_spurious": 0.99783098101164747,
        "q_shape": 0.21984521141416016,
        "sharp_active": 0,
        "sharp_metrics_valid": 0,
        "q_sharp": "NA",
        "q_fidelity": 0.21984521141416016,
    },

    {
        "model": "bone_femur",
        "hex": 2102,
        "total_polys": 2106,
        "nonhex": 4,
        "d_c": 0.88780577166826335,
        "q_missing": 0.9974300309629075,
        "q_spurious": 0.99804809503350089,
        "q_shape": 0.9974300309629075,
        "sharp_active": 0,
        "sharp_metrics_valid": 0,
        "q_sharp": "NA",
        "q_fidelity": 0.9974300309629075,
    },

    {
        "model": "mechanical08",
        "hex": 11100,
        "total_polys": 11136,
        "nonhex": 36,
        "d_c": 0.78134529313860679,
        "q_missing": 1.0,
        "q_spurious": 1.0,
        "q_shape": 1.0,
        "sharp_active": 1,
        "sharp_metrics_valid": 1,
        "q_sharp": 0.99483903795655504,
        "q_fidelity": 0.99483903795655504,
    },
]


for record in CASES:
    facts = parse_terminal_quality_facts(
        dict(record)
    )

    assert facts.model == record["model"]

    assert facts.hex == record["hex"]
    assert facts.total_polys == record["total_polys"]
    assert facts.nonhex == record["nonhex"]

    assert bits(facts.d_c) == bits(record["d_c"])
    assert bits(facts.q_missing) == bits(record["q_missing"])
    assert bits(facts.q_spurious) == bits(record["q_spurious"])
    assert bits(facts.q_shape) == bits(record["q_shape"])
    assert bits(facts.q_fidelity) == bits(record["q_fidelity"])

    if record["sharp_active"]:
        assert facts.q_sharp is not None

        assert (
            bits(facts.q_sharp)
            ==
            bits(record["q_sharp"])
        )

        assert (
            bits(
                facts.q_shape
                *
                facts.q_sharp
            )
            ==
            bits(
                facts.q_fidelity
            )
        )

    else:
        assert facts.q_sharp is None

        assert (
            bits(facts.q_shape)
            ==
            bits(facts.q_fidelity)
        )


# ------------------------------------------------------------------
# Fail closed on inconsistent shape.
# ------------------------------------------------------------------

bad = dict(
    CASES[0]
)

bad["q_shape"] = 0.5

try:
    parse_terminal_quality_facts(
        bad
    )
except TerminalQualityError:
    pass
else:
    raise AssertionError(
        "invalid q_shape was accepted"
    )


# ------------------------------------------------------------------
# Fail closed on SHARP branch corruption.
# ------------------------------------------------------------------

bad = dict(
    CASES[1]
)

bad["sharp_metrics_valid"] = 1

try:
    parse_terminal_quality_facts(
        bad
    )
except TerminalQualityError:
    pass
else:
    raise AssertionError(
        "inactive SHARP with metrics_valid=1 was accepted"
    )


# ------------------------------------------------------------------
# Fail closed on count corruption.
# ------------------------------------------------------------------

bad = dict(
    CASES[2]
)

bad["nonhex"] = 5

try:
    parse_terminal_quality_facts(
        bad
    )
except TerminalQualityError:
    pass
else:
    raise AssertionError(
        "inconsistent nonhex count was accepted"
    )


print(
    "PASS: four frozen terminal-quality quadrants parse bit-exact"
)

print(
    "PASS: active SHARP multiplication is bit-exact"
)

print(
    "PASS: inactive SHARP identity is bit-exact"
)

print(
    "PASS: corrupted quality protocol records fail closed"
)
