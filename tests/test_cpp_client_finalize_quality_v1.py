from __future__ import annotations

import tempfile
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


from bridge.cpp_client import (
    LoopyCutsClient,
)


with tempfile.TemporaryDirectory() as tmp:
    ref = (
        Path(tmp)
        /
        "mechanical02.quality_ref_v1"
    )

    ref.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    client = object.__new__(
        LoopyCutsClient
    )

    client.state = None
    client.actions = []
    client.used = []
    client.reverted = []
    client.nico_bug = []
    client.top_relevant = []

    sent = []
    reads = []

    client._send = (
        lambda command, phase:
            sent.append(
                (
                    command,
                    phase,
                )
            )
    )

    lines = [
        "[RL] FINALIZE_QUALITY_BEGIN",

        (
            "[RL] FINAL_RESULT "
            "hex=880 "
            "total_polys=880 "
            "full_hex=1"
        ),

        (
            "[RL] FINALIZE_QUALITY "
            "model=mechanical02 "
            "hex=880 "
            "total_polys=880 "
            "nonhex=0 "
            "d_c=1 "
            "q_missing=0.99853225619739827 "
            "q_spurious=0.99861570853523696 "
            "q_shape=0.99853225619739827 "
            "sharp_active=1 "
            "sharp_metrics_valid=1 "
            "q_sharp=0.99019972262228984 "
            "q_fidelity=0.988746363116073"
        ),

        "[RL] FINALIZE_QUALITY_END",

        (
            "[RL] STATE "
            "step=21 "
            "available=0 "
            "terminal=1 "
            "finalized=1"
        ),

        "[RL] USED",
        "[RL] REVERTED",
        "[RL] NICO_BUG",
        "[RL] TOP_RELEVANT",
        "[RL] ACTIONS",
    ]

    client._read_until_with_finalize_eval_swap_guard = (
        lambda prefix, phase:
            (
                reads.append(
                    (
                        prefix,
                        phase,
                    )
                )
                or
                list(lines)
            )
    )

    (
        final_result,
        quality_record,
        final_state,
    ) = client.finalize_quality(
        ref
    )

    assert sent == [
        (
            (
                "FINALIZE_QUALITY "
                f"{ref.resolve()}"
            ),
            "FINALIZE_QUALITY",
        )
    ]

    assert reads == [
        (
            "[RL] ACTIONS",
            "FINALIZE_QUALITY",
        )
    ]

    assert final_result == {
        "hex": 880,
        "total_polys": 880,
        "full_hex": 1,
    }

    assert (
        quality_record["model"]
        ==
        "mechanical02"
    )

    assert (
        quality_record["q_fidelity"]
        ==
        0.988746363116073
    )

    assert final_state == {
        "step": 21,
        "available": 0,
        "terminal": 1,
        "finalized": 1,
    }

    assert client.actions == []


print(
    "PASS: FINALIZE_QUALITY command is emitted exactly"
)

print(
    "PASS: FINAL_RESULT and quality records parse independently"
)

print(
    "PASS: final STATE/ACTIONS still update through legacy parser"
)
