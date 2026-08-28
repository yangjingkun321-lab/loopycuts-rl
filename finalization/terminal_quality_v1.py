from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

import math
import struct


TERMINAL_QUALITY_V1_VERSION = (
    "terminal_quality_facts_v1"
)


class TerminalQualityError(
    ValueError
):
    pass


def _binary(
    value,
    *,
    name: str,
) -> int:
    value = int(
        value
    )

    if value not in (
        0,
        1,
    ):
        raise TerminalQualityError(
            f"{name} must be 0 or 1"
        )

    return value


def _unit_float(
    value,
    *,
    name: str,
) -> float:
    value = float(
        value
    )

    if not math.isfinite(
        value
    ):
        raise TerminalQualityError(
            f"{name} must be finite"
        )

    if not (
        0.0
        <=
        value
        <=
        1.0
    ):
        raise TerminalQualityError(
            f"{name} must be in [0, 1]"
        )

    return value


def _bits(
    value,
):
    return struct.pack(
        "=d",
        float(
            value
        ),
    )


@dataclass(
    frozen=True
)
class TerminalQualityFacts:
    """
    Exact scalar facts emitted by C++ FINALIZE_QUALITY.

    Python does not reconstruct geometry, nearest-distance metrics,
    SHARP metrics, D_C, or Q_fidelity.
    """

    model: str

    hex: int
    total_polys: int
    nonhex: int

    d_c: float

    q_missing: float
    q_spurious: float
    q_shape: float

    sharp_active: int
    sharp_metrics_valid: int

    q_sharp: float | None

    q_fidelity: float

    def to_dict(
        self,
    ) -> dict:
        return asdict(
            self
        )

    @property
    def utility(
        self,
    ) -> float:
        return (
            float(
                self.d_c
            )
            *
            float(
                self.q_fidelity
            )
        )


_REQUIRED_FIELDS = {
    "model",

    "hex",
    "total_polys",
    "nonhex",

    "d_c",

    "q_missing",
    "q_spurious",
    "q_shape",

    "sharp_active",
    "sharp_metrics_valid",
    "q_sharp",

    "q_fidelity",
}


def parse_terminal_quality_facts(
    data,
) -> TerminalQualityFacts:
    if not isinstance(
        data,
        dict,
    ):
        raise TerminalQualityError(
            "FINALIZE_QUALITY record must be a dict"
        )

    keys = set(
        data
    )

    missing = (
        _REQUIRED_FIELDS
        -
        keys
    )

    if missing:
        raise TerminalQualityError(
            "FINALIZE_QUALITY record is missing fields: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )

    extra = (
        keys
        -
        _REQUIRED_FIELDS
    )

    if extra:
        raise TerminalQualityError(
            "FINALIZE_QUALITY record has unexpected fields: "
            +
            ", ".join(
                sorted(
                    extra
                )
            )
        )

    model = str(
        data[
            "model"
        ]
    )

    if not model:
        raise TerminalQualityError(
            "model must be non-empty"
        )

    final_hex = int(
        data[
            "hex"
        ]
    )

    total_polys = int(
        data[
            "total_polys"
        ]
    )

    nonhex = int(
        data[
            "nonhex"
        ]
    )

    if total_polys <= 0:
        raise TerminalQualityError(
            "total_polys must be positive"
        )

    if not (
        0
        <=
        final_hex
        <=
        total_polys
    ):
        raise TerminalQualityError(
            "hex must lie in [0, total_polys]"
        )

    if (
        nonhex
        !=
        total_polys
        -
        final_hex
    ):
        raise TerminalQualityError(
            "nonhex != total_polys - hex"
        )

    d_c = _unit_float(
        data[
            "d_c"
        ],
        name="d_c",
    )

    q_missing = _unit_float(
        data[
            "q_missing"
        ],
        name="q_missing",
    )

    q_spurious = _unit_float(
        data[
            "q_spurious"
        ],
        name="q_spurious",
    )

    q_shape = _unit_float(
        data[
            "q_shape"
        ],
        name="q_shape",
    )

    q_fidelity = _unit_float(
        data[
            "q_fidelity"
        ],
        name="q_fidelity",
    )

    sharp_active = _binary(
        data[
            "sharp_active"
        ],
        name="sharp_active",
    )

    sharp_metrics_valid = _binary(
        data[
            "sharp_metrics_valid"
        ],
        name="sharp_metrics_valid",
    )

    expected_q_shape = min(
        q_missing,
        q_spurious,
    )

    if (
        _bits(
            q_shape
        )
        !=
        _bits(
            expected_q_shape
        )
    ):
        raise TerminalQualityError(
            "q_shape is not bit-exact "
            "min(q_missing, q_spurious)"
        )

    raw_q_sharp = data[
        "q_sharp"
    ]

    if sharp_active:
        if not sharp_metrics_valid:
            raise TerminalQualityError(
                "active SHARP requires "
                "sharp_metrics_valid=1"
            )

        if raw_q_sharp == "NA":
            raise TerminalQualityError(
                "active SHARP requires numeric q_sharp"
            )

        q_sharp = _unit_float(
            raw_q_sharp,
            name="q_sharp",
        )

        expected_q_fidelity = (
            q_shape
            *
            q_sharp
        )

        if (
            _bits(
                q_fidelity
            )
            !=
            _bits(
                expected_q_fidelity
            )
        ):
            raise TerminalQualityError(
                "active q_fidelity != "
                "q_shape * q_sharp bit-exact"
            )

    else:
        if sharp_metrics_valid:
            raise TerminalQualityError(
                "inactive SHARP requires "
                "sharp_metrics_valid=0"
            )

        if raw_q_sharp != "NA":
            raise TerminalQualityError(
                "inactive SHARP requires q_sharp=NA"
            )

        q_sharp = None

        if (
            _bits(
                q_fidelity
            )
            !=
            _bits(
                q_shape
            )
        ):
            raise TerminalQualityError(
                "inactive q_fidelity != "
                "q_shape bit-exact"
            )

    return TerminalQualityFacts(
        model=
            model,

        hex=
            final_hex,

        total_polys=
            total_polys,

        nonhex=
            nonhex,

        d_c=
            d_c,

        q_missing=
            q_missing,

        q_spurious=
            q_spurious,

        q_shape=
            q_shape,

        sharp_active=
            sharp_active,

        sharp_metrics_valid=
            sharp_metrics_valid,

        q_sharp=
            q_sharp,

        q_fidelity=
            q_fidelity,
    )
