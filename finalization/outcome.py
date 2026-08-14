from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

from bridge.cpp_client import (
    LoopyCutsClient,
    RLServerProcessError,
)


OUTCOME_FULL_HEX = "FULL_HEX"
OUTCOME_NON_FULL_HEX = "NON_FULL_HEX"
OUTCOME_CRASH = "FINALIZATION_CRASH"


class FinalizationOutcomeError(
    RuntimeError
):
    pass


@dataclass(frozen=True)
class FinalizationOutcome:
    outcome: str

    completed: bool
    crashed: bool

    return_code: int | None
    signal_number: int | None
    signal_name: str | None

    final_hex: int | None
    final_total_polys: int | None
    full_hex: int | None

    final_result: dict[str, Any] | None
    final_state: dict[str, Any] | None

    log_tail: tuple[str, ...]

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(
            self
        )


def evaluate_terminal_finalization(
    client: LoopyCutsClient,
) -> FinalizationOutcome:
    """
    Run no-save FINALIZE_EVAL on one terminal Stage-2 state.

    Valid geometric/runtime outcomes:
        FULL_HEX
        NON_FULL_HEX
        FINALIZATION_CRASH

    Infrastructure/protocol errors are NOT converted into mesh-quality
    outcomes and must propagate to the caller.
    """

    state = client.state

    if state is None:
        raise FinalizationOutcomeError(
            "Client has no current state"
        )

    if not int(
        state[
            "terminal"
        ]
    ):
        raise FinalizationOutcomeError(
            "FINALIZE_EVAL requires a terminal "
            "Stage-2 selection state"
        )

    if int(
        state.get(
            "finalized",
            0,
        )
    ):
        raise FinalizationOutcomeError(
            "Stage-2 state is already finalized"
        )

    try:
        (
            final_result,
            final_state,
        ) = client.finalize_eval()

    except RLServerProcessError as exc:
        #
        # Only an actual non-zero child-process failure is classified
        # as FINALIZATION_CRASH.
        #
        # EOF / process disappearance without a meaningful failure
        # return code is infrastructure failure, not an RL outcome.
        #
        if (
            exc.return_code is None
            or exc.return_code == 0
        ):
            raise

        return FinalizationOutcome(
            outcome=OUTCOME_CRASH,

            completed=False,
            crashed=True,

            return_code=(
                exc.return_code
            ),

            signal_number=(
                exc.signal_number
            ),

            signal_name=(
                exc.signal_name
            ),

            final_hex=None,
            final_total_polys=None,
            full_hex=None,

            final_result=None,
            final_state=None,

            log_tail=tuple(
                exc.lines[
                    -50:
                ]
            ),
        )

    if final_result is None:
        raise FinalizationOutcomeError(
            "FINALIZE_EVAL completed without FINAL_RESULT"
        )

    required = {
        "hex",
        "total_polys",
        "full_hex",
    }

    missing = (
        required
        -
        set(
            final_result
        )
    )

    if missing:
        raise FinalizationOutcomeError(
            "FINAL_RESULT is missing fields: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    final_hex = int(
        final_result[
            "hex"
        ]
    )

    final_total_polys = int(
        final_result[
            "total_polys"
        ]
    )

    full_hex = int(
        final_result[
            "full_hex"
        ]
    )

    if full_hex not in (
        0,
        1,
    ):
        raise FinalizationOutcomeError(
            f"Invalid full_hex value: "
            f"{full_hex}"
        )

    #
    # Validate FINAL_RESULT itself.
    #
    if full_hex:
        if (
            final_hex
            !=
            final_total_polys
        ):
            raise FinalizationOutcomeError(
                "full_hex=1 but "
                "hex != total_polys"
            )

        outcome = (
            OUTCOME_FULL_HEX
        )

    else:
        if (
            final_hex
            >=
            final_total_polys
        ):
            raise FinalizationOutcomeError(
                "full_hex=0 but "
                "hex >= total_polys"
            )

        outcome = (
            OUTCOME_NON_FULL_HEX
        )

    if final_state is None:
        raise FinalizationOutcomeError(
            "FINALIZE_EVAL completed without final STATE"
        )

    if not int(
        final_state[
            "finalized"
        ]
    ):
        raise FinalizationOutcomeError(
            "FINALIZE_EVAL completed but "
            "finalized state flag is false"
        )

    return FinalizationOutcome(
        outcome=outcome,

        completed=True,
        crashed=False,

        return_code=None,
        signal_number=None,
        signal_name=None,

        final_hex=final_hex,
        final_total_polys=(
            final_total_polys
        ),
        full_hex=full_hex,

        final_result=dict(
            final_result
        ),

        final_state=dict(
            final_state
        ),

        log_tail=(),
    )
