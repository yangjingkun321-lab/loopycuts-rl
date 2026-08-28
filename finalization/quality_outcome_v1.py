from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any


from bridge.cpp_client import (
    LoopyCutsClient,
    RLServerProcessError,
)

from finalization.outcome import (
    OUTCOME_CRASH,
    OUTCOME_FULL_HEX,
    OUTCOME_NON_FULL_HEX,
)

from finalization.terminal_quality_v1 import (
    TerminalQualityFacts,
    parse_terminal_quality_facts,
)


QUALITY_FINALIZATION_OUTCOME_V1_VERSION = (
    "quality_finalization_outcome_v1"
)


class QualityFinalizationOutcomeError(
    RuntimeError
):
    pass


@dataclass(
    frozen=True
)
class QualityFinalizationOutcome:
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

    terminal_quality: TerminalQualityFacts | None

    log_tail: tuple[str, ...]

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(
            self
        )


def evaluate_terminal_quality_finalization(
    client: LoopyCutsClient,
    *,
    quality_ref_path,
    expected_model: str,
) -> QualityFinalizationOutcome:
    """
    Run FINALIZE_QUALITY on one genuine Stage-2 terminal state.

    C++ remains authoritative for:
        final cell counts,
        D_C,
        Q_missing,
        Q_spurious,
        Q_shape,
        Q_sharp,
        Q_fidelity.

    Python only validates cross-protocol consistency.

    ResourceGuard exceptions deliberately propagate to the outer
    environment wrapper, exactly as in the existing FINALIZE_EVAL
    path.
    """

    expected_model = str(
        expected_model
    )

    if not expected_model:
        raise QualityFinalizationOutcomeError(
            "expected_model must be non-empty"
        )

    state = client.state

    if state is None:
        raise QualityFinalizationOutcomeError(
            "Client has no current state"
        )

    if not int(
        state[
            "terminal"
        ]
    ):
        raise QualityFinalizationOutcomeError(
            "FINALIZE_QUALITY requires a terminal "
            "Stage-2 selection state"
        )

    if int(
        state.get(
            "finalized",
            0,
        )
    ):
        raise QualityFinalizationOutcomeError(
            "Stage-2 state is already finalized"
        )

    try:
        (
            final_result,
            quality_record,
            final_state,
        ) = client.finalize_quality(
            quality_ref_path
        )

    except RLServerProcessError as exc:
        #
        # Preserve the frozen legacy classification:
        #
        # actual non-zero C++ process failure
        #     -> FINALIZATION_CRASH
        #
        # EOF/infrastructure ambiguity
        #     -> propagate
        #
        if (
            exc.return_code is None
            or
            exc.return_code == 0
        ):
            raise

        return QualityFinalizationOutcome(
            outcome=
                OUTCOME_CRASH,

            completed=
                False,

            crashed=
                True,

            return_code=
                exc.return_code,

            signal_number=
                exc.signal_number,

            signal_name=
                exc.signal_name,

            final_hex=
                None,

            final_total_polys=
                None,

            full_hex=
                None,

            final_result=
                None,

            final_state=
                None,

            terminal_quality=
                None,

            log_tail=
                tuple(
                    exc.lines[
                        -50:
                    ]
                ),
        )

    if not isinstance(
        final_result,
        dict,
    ):
        raise QualityFinalizationOutcomeError(
            "FINALIZE_QUALITY completed without FINAL_RESULT"
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
        raise QualityFinalizationOutcomeError(
            "FINAL_RESULT is missing fields: "
            +
            ", ".join(
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
        raise QualityFinalizationOutcomeError(
            "full_hex must be 0 or 1"
        )

    if full_hex:
        if (
            final_hex
            !=
            final_total_polys
        ):
            raise QualityFinalizationOutcomeError(
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
            raise QualityFinalizationOutcomeError(
                "full_hex=0 but "
                "hex >= total_polys"
            )

        outcome = (
            OUTCOME_NON_FULL_HEX
        )

    terminal_quality = (
        parse_terminal_quality_facts(
            quality_record
        )
    )

    #
    # Model identity must come back exactly from the C++ quality ref.
    #
    if (
        terminal_quality.model
        !=
        expected_model
    ):
        raise QualityFinalizationOutcomeError(
            "FINALIZE_QUALITY model mismatch: "
            f"expected={expected_model!r}, "
            f"actual={terminal_quality.model!r}"
        )

    #
    # FINAL_RESULT and FINALIZE_QUALITY are independent protocol
    # records. They must agree exactly on final cell counts.
    #
    if (
        terminal_quality.hex
        !=
        final_hex
    ):
        raise QualityFinalizationOutcomeError(
            "FINAL_RESULT/FINALIZE_QUALITY hex mismatch"
        )

    if (
        terminal_quality.total_polys
        !=
        final_total_polys
    ):
        raise QualityFinalizationOutcomeError(
            "FINAL_RESULT/FINALIZE_QUALITY "
            "total_polys mismatch"
        )

    if final_state is None:
        raise QualityFinalizationOutcomeError(
            "FINALIZE_QUALITY completed without final STATE"
        )

    if not int(
        final_state[
            "finalized"
        ]
    ):
        raise QualityFinalizationOutcomeError(
            "FINALIZE_QUALITY completed but "
            "finalized state flag is false"
        )

    if not int(
        final_state[
            "terminal"
        ]
    ):
        raise QualityFinalizationOutcomeError(
            "FINALIZE_QUALITY final STATE "
            "lost terminal flag"
        )

    return QualityFinalizationOutcome(
        outcome=
            outcome,

        completed=
            True,

        crashed=
            False,

        return_code=
            None,

        signal_number=
            None,

        signal_name=
            None,

        final_hex=
            final_hex,

        final_total_polys=
            final_total_polys,

        full_hex=
            full_hex,

        final_result=
            dict(
                final_result
            ),

        final_state=
            dict(
                final_state
            ),

        terminal_quality=
            terminal_quality,

        log_tail=
            (),
    )
