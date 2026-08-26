from __future__ import annotations

import hashlib
import math
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCHEMA_MAGIC = "LOOPYCUTS_QUALITY_REF_V1"
SCHEMA_END = "END_LOOPYCUTS_QUALITY_REF_V1"

METRIC_CONTRACT_V3_SHA256 = (
    "060d7f40293303aaa8fbdcbbe3999a303"
    "b14ee6b9dc0fd8bb93f21abed6d731d"
)

GEOMETRY_SAMPLE_COUNT_V1 = 30_000
FINAL_DRAW_COUNT_V1 = 30_000

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class InputGeometrySample:
    x: float
    y: float
    z: float
    h: float


@dataclass(frozen=True)
class FinalSurfaceDraw:
    area_draw: float
    u: float
    v: float


@dataclass(frozen=True)
class SharpSample:
    x: float
    y: float
    z: float
    h: float
    theta: float


@dataclass(frozen=True)
class QualityRefV1:
    model: str

    metric_contract_sha256: str
    stage2_input_sha256: str

    sharp_present: bool
    sharp_declared_count: int | None
    sharp_file_sha256: str | None
    sharp_source_obj_sha256: str | None

    input_sample_seed_u64: int
    final_draw_seed_u64: int

    input_geometry: tuple[InputGeometrySample, ...]
    final_draws: tuple[FinalSurfaceDraw, ...]
    sharp_samples: tuple[SharpSample, ...]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def _finite(value: float, name: str) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite; got {value!r}"
        )

    return value


def _validate_sha256(
    value: str,
    name: str,
) -> str:
    value = str(value)

    if not SHA256_RE.fullmatch(value):
        raise ValueError(
            f"{name} must be lowercase SHA256; "
            f"got {value!r}"
        )

    return value


def _validate_model_name(model: str) -> str:
    model = str(model)

    if not model:
        raise ValueError(
            "model must not be empty"
        )

    if any(ch.isspace() for ch in model):
        raise ValueError(
            "model must not contain whitespace"
        )

    return model


def validate_quality_ref_v1(
    ref: QualityRefV1,
    *,
    require_v1_sample_counts: bool = True,
) -> None:
    _validate_model_name(
        ref.model
    )

    _validate_sha256(
        ref.metric_contract_sha256,
        "metric_contract_sha256",
    )

    _validate_sha256(
        ref.stage2_input_sha256,
        "stage2_input_sha256",
    )

    if (
        ref.metric_contract_sha256
        !=
        METRIC_CONTRACT_V3_SHA256
    ):
        raise ValueError(
            "Unsupported metric-contract SHA256: "
            f"{ref.metric_contract_sha256}"
        )

    for name, value in (
        (
            "input_sample_seed_u64",
            ref.input_sample_seed_u64,
        ),
        (
            "final_draw_seed_u64",
            ref.final_draw_seed_u64,
        ),
    ):
        if not isinstance(value, int):
            raise TypeError(
                f"{name} must be int"
            )

        if not (
            0 <= value <= (2**64 - 1)
        ):
            raise ValueError(
                f"{name} outside uint64 range"
            )

    sharp_declared_count = (
        ref.sharp_declared_count
    )

    if sharp_declared_count is None:

        if ref.sharp_present:
            raise ValueError(
                "sharp_present=True requires "
                "an explicit SHARP source"
            )

        if ref.sharp_file_sha256 is not None:
            raise ValueError(
                "No SHARP source requires "
                "sharp_file_sha256=None"
            )

        if ref.sharp_source_obj_sha256 is not None:
            raise ValueError(
                "No SHARP source requires "
                "sharp_source_obj_sha256=None"
            )

        if ref.sharp_samples:
            raise ValueError(
                "No SHARP source requires "
                "zero SHARP samples"
            )

    else:

        if (
            not isinstance(
                sharp_declared_count,
                int,
            )
            or
            isinstance(
                sharp_declared_count,
                bool,
            )
        ):
            raise TypeError(
                "sharp_declared_count must "
                "be int or None"
            )

        if sharp_declared_count < 0:
            raise ValueError(
                "sharp_declared_count must "
                "be non-negative"
            )

        if ref.sharp_file_sha256 is None:
            raise ValueError(
                "Explicit SHARP source requires "
                "sharp_file_sha256"
            )

        if ref.sharp_source_obj_sha256 is None:
            raise ValueError(
                "Explicit SHARP source requires "
                "sharp_source_obj_sha256"
            )

        _validate_sha256(
            ref.sharp_file_sha256,
            "sharp_file_sha256",
        )

        _validate_sha256(
            ref.sharp_source_obj_sha256,
            "sharp_source_obj_sha256",
        )

        if sharp_declared_count == 0:

            if ref.sharp_present:
                raise ValueError(
                    "SHARP_PRESENT must be false "
                    "when declared count is zero"
                )

            if ref.sharp_samples:
                raise ValueError(
                    "Zero-feature SHARP source "
                    "requires zero SHARP samples"
                )

        else:

            if not ref.sharp_present:
                raise ValueError(
                    "Positive SHARP declared count "
                    "requires SHARP_PRESENT=1"
                )

            if not ref.sharp_samples:
                raise ValueError(
                    "Positive SHARP declared count "
                    "requires SHARP samples"
                )

    if require_v1_sample_counts:
        if (
            len(ref.input_geometry)
            !=
            GEOMETRY_SAMPLE_COUNT_V1
        ):
            raise ValueError(
                "V1 requires exactly "
                f"{GEOMETRY_SAMPLE_COUNT_V1} "
                "Input geometry samples; got "
                f"{len(ref.input_geometry)}"
            )

        if (
            len(ref.final_draws)
            !=
            FINAL_DRAW_COUNT_V1
        ):
            raise ValueError(
                "V1 requires exactly "
                f"{FINAL_DRAW_COUNT_V1} "
                "Final surface draws; got "
                f"{len(ref.final_draws)}"
            )

    for i, sample in enumerate(
        ref.input_geometry
    ):
        for field_name, value in (
            ("x", sample.x),
            ("y", sample.y),
            ("z", sample.z),
            ("h", sample.h),
        ):
            _finite(
                value,
                f"input_geometry[{i}].{field_name}",
            )

        if sample.h <= 0.0:
            raise ValueError(
                f"input_geometry[{i}].h "
                "must be positive"
            )

    for i, draw in enumerate(
        ref.final_draws
    ):
        for field_name, value in (
            ("area_draw", draw.area_draw),
            ("u", draw.u),
            ("v", draw.v),
        ):
            _finite(
                value,
                f"final_draws[{i}].{field_name}",
            )

        for field_name, value in (
            ("area_draw", draw.area_draw),
            ("u", draw.u),
            ("v", draw.v),
        ):
            if not (
                0.0 <= value < 1.0
            ):
                raise ValueError(
                    f"final_draws[{i}]."
                    f"{field_name} "
                    "must satisfy 0 <= x < 1"
                )

    for i, sample in enumerate(
        ref.sharp_samples
    ):
        for field_name, value in (
            ("x", sample.x),
            ("y", sample.y),
            ("z", sample.z),
            ("h", sample.h),
            ("theta", sample.theta),
        ):
            _finite(
                value,
                f"sharp_samples[{i}]."
                f"{field_name}",
            )

        if sample.h <= 0.0:
            raise ValueError(
                f"sharp_samples[{i}].h "
                "must be positive"
            )

        # Frozen continuous-SHARP V3 semantics:
        #
        # theta = degrees(acos(abs(dot(n0,n1))))
        #
        # Therefore theta is in [0, 90].
        #
        # A zero-strength declared SHARP edge is not useful as a
        # relative-strength denominator and is rejected here.
        if not (
            0.0 < sample.theta <= 90.0
        ):
            raise ValueError(
                f"sharp_samples[{i}].theta "
                "must satisfy 0 < theta <= 90"
            )


def _format_float(value: float) -> str:
    value = _finite(
        value,
        "canonical float",
    )

    text = format(
        value,
        ".17g",
    )

    # Canonicalize negative zero.
    if text in {
        "-0",
        "-0.0",
    }:
        return "0"

    return text


def _header_sha(
    value: str | None,
) -> str:
    return (
        "NONE"
        if value is None
        else value
    )


def quality_ref_v1_to_text(
    ref: QualityRefV1,
    *,
    require_v1_sample_counts: bool = True,
) -> str:
    validate_quality_ref_v1(
        ref,
        require_v1_sample_counts=
            require_v1_sample_counts,
    )

    lines: list[str] = [
        SCHEMA_MAGIC,
        "",
        f"MODEL {ref.model}",
        (
            "METRIC_CONTRACT_SHA256 "
            f"{ref.metric_contract_sha256}"
        ),
        (
            "STAGE2_INPUT_SHA256 "
            f"{ref.stage2_input_sha256}"
        ),
        (
            "SHARP_PRESENT "
            f"{1 if ref.sharp_present else 0}"
        ),
        (
            "SHARP_DECLARED_COUNT "
            f"{'NONE' if ref.sharp_declared_count is None else ref.sharp_declared_count}"
        ),
        (
            "SHARP_FILE_SHA256 "
            f"{_header_sha(ref.sharp_file_sha256)}"
        ),
        (
            "SHARP_SOURCE_OBJ_SHA256 "
            f"{_header_sha(ref.sharp_source_obj_sha256)}"
        ),
        (
            "INPUT_SAMPLE_SEED_U64 "
            f"{ref.input_sample_seed_u64}"
        ),
        (
            "FINAL_DRAW_SEED_U64 "
            f"{ref.final_draw_seed_u64}"
        ),
        (
            "GEOMETRY_SAMPLE_COUNT "
            f"{len(ref.input_geometry)}"
        ),
        (
            "FINAL_DRAW_COUNT "
            f"{len(ref.final_draws)}"
        ),
        (
            "SHARP_SAMPLE_COUNT "
            f"{len(ref.sharp_samples)}"
        ),
        "",
        "BEGIN_INPUT_GEOMETRY",
    ]

    for sample in ref.input_geometry:
        lines.append(
            " ".join(
                (
                    _format_float(sample.x),
                    _format_float(sample.y),
                    _format_float(sample.z),
                    _format_float(sample.h),
                )
            )
        )

    lines.extend(
        (
            "END_INPUT_GEOMETRY",
            "",
            "BEGIN_FINAL_DRAWS",
        )
    )

    for draw in ref.final_draws:
        lines.append(
            " ".join(
                (
                    _format_float(draw.area_draw),
                    _format_float(draw.u),
                    _format_float(draw.v),
                )
            )
        )

    lines.extend(
        (
            "END_FINAL_DRAWS",
            "",
            "BEGIN_SHARP",
        )
    )

    for sample in ref.sharp_samples:
        lines.append(
            " ".join(
                (
                    _format_float(sample.x),
                    _format_float(sample.y),
                    _format_float(sample.z),
                    _format_float(sample.h),
                    _format_float(sample.theta),
                )
            )
        )

    lines.extend(
        (
            "END_SHARP",
            "",
            SCHEMA_END,
            "",
        )
    )

    return "\n".join(
        lines
    )


def write_quality_ref_v1(
    path: str | Path,
    ref: QualityRefV1,
    *,
    require_v1_sample_counts: bool = True,
) -> None:
    text = quality_ref_v1_to_text(
        ref,
        require_v1_sample_counts=
            require_v1_sample_counts,
    )

    Path(path).write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )


def _parse_header_line(
    line: str,
    expected_key: str,
) -> str:
    parts = line.split(
        maxsplit=1
    )

    if (
        len(parts) != 2
        or
        parts[0] != expected_key
    ):
        raise ValueError(
            f"Expected header "
            f"{expected_key!r}; got {line!r}"
        )

    return parts[1]


def _parse_float_row(
    line: str,
    expected_fields: int,
    context: str,
) -> tuple[float, ...]:
    parts = line.split()

    if len(parts) != expected_fields:
        raise ValueError(
            f"{context}: expected "
            f"{expected_fields} fields; "
            f"got {len(parts)}"
        )

    values = tuple(
        float(x)
        for x in parts
    )

    for i, value in enumerate(
        values
    ):
        _finite(
            value,
            f"{context}[{i}]",
        )

    return values


def read_quality_ref_v1(
    path: str | Path,
    *,
    require_v1_sample_counts: bool = True,
    require_canonical_bytes: bool = True,
) -> QualityRefV1:
    path = Path(path)

    raw = path.read_bytes()

    try:
        text = raw.decode(
            "utf-8"
        )
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path}: not valid UTF-8"
        ) from exc

    # Canonical V1 files use LF only.
    if b"\r" in raw:
        raise ValueError(
            f"{path}: CR bytes are not canonical"
        )

    lines = text.splitlines()

    cursor = 0

    def take() -> str:
        nonlocal cursor

        if cursor >= len(lines):
            raise ValueError(
                f"{path}: unexpected EOF"
            )

        line = lines[cursor]

        cursor += 1

        return line

    def expect(value: str) -> None:
        line = take()

        if line != value:
            raise ValueError(
                f"{path}: expected {value!r}; "
                f"got {line!r}"
            )

    expect(
        SCHEMA_MAGIC
    )

    expect(
        ""
    )

    model = _parse_header_line(
        take(),
        "MODEL",
    )

    metric_contract_sha256 = (
        _parse_header_line(
            take(),
            "METRIC_CONTRACT_SHA256",
        )
    )

    stage2_input_sha256 = (
        _parse_header_line(
            take(),
            "STAGE2_INPUT_SHA256",
        )
    )

    sharp_present_text = (
        _parse_header_line(
            take(),
            "SHARP_PRESENT",
        )
    )

    if sharp_present_text not in {
        "0",
        "1",
    }:
        raise ValueError(
            "SHARP_PRESENT must be 0 or 1"
        )

    sharp_present = (
        sharp_present_text == "1"
    )

    sharp_declared_count_text = (
        _parse_header_line(
            take(),
            "SHARP_DECLARED_COUNT",
        )
    )

    if sharp_declared_count_text == "NONE":
        sharp_declared_count = None
    else:
        sharp_declared_count = int(
            sharp_declared_count_text
        )

    sharp_file_sha256_text = (
        _parse_header_line(
            take(),
            "SHARP_FILE_SHA256",
        )
    )

    sharp_source_obj_sha256_text = (
        _parse_header_line(
            take(),
            "SHARP_SOURCE_OBJ_SHA256",
        )
    )

    sharp_file_sha256 = (
        None
        if sharp_file_sha256_text == "NONE"
        else sharp_file_sha256_text
    )

    sharp_source_obj_sha256 = (
        None
        if sharp_source_obj_sha256_text == "NONE"
        else sharp_source_obj_sha256_text
    )

    input_sample_seed_u64 = int(
        _parse_header_line(
            take(),
            "INPUT_SAMPLE_SEED_U64",
        )
    )

    final_draw_seed_u64 = int(
        _parse_header_line(
            take(),
            "FINAL_DRAW_SEED_U64",
        )
    )

    geometry_sample_count = int(
        _parse_header_line(
            take(),
            "GEOMETRY_SAMPLE_COUNT",
        )
    )

    final_draw_count = int(
        _parse_header_line(
            take(),
            "FINAL_DRAW_COUNT",
        )
    )

    sharp_sample_count = int(
        _parse_header_line(
            take(),
            "SHARP_SAMPLE_COUNT",
        )
    )

    for name, count in (
        (
            "GEOMETRY_SAMPLE_COUNT",
            geometry_sample_count,
        ),
        (
            "FINAL_DRAW_COUNT",
            final_draw_count,
        ),
        (
            "SHARP_SAMPLE_COUNT",
            sharp_sample_count,
        ),
    ):
        if count < 0:
            raise ValueError(
                f"{name} must be non-negative"
            )

    expect(
        ""
    )

    expect(
        "BEGIN_INPUT_GEOMETRY"
    )

    input_geometry = []

    for i in range(
        geometry_sample_count
    ):
        x, y, z, h = _parse_float_row(
            take(),
            4,
            f"INPUT_GEOMETRY row {i}",
        )

        input_geometry.append(
            InputGeometrySample(
                x=x,
                y=y,
                z=z,
                h=h,
            )
        )

    expect(
        "END_INPUT_GEOMETRY"
    )

    expect(
        ""
    )

    expect(
        "BEGIN_FINAL_DRAWS"
    )

    final_draws = []

    for i in range(
        final_draw_count
    ):
        (
            area_draw,
            u,
            v,
        ) = _parse_float_row(
            take(),
            3,
            f"FINAL_DRAWS row {i}",
        )

        final_draws.append(
            FinalSurfaceDraw(
                area_draw=area_draw,
                u=u,
                v=v,
            )
        )

    expect(
        "END_FINAL_DRAWS"
    )

    expect(
        ""
    )

    expect(
        "BEGIN_SHARP"
    )

    sharp_samples = []

    for i in range(
        sharp_sample_count
    ):
        (
            x,
            y,
            z,
            h,
            theta,
        ) = _parse_float_row(
            take(),
            5,
            f"SHARP row {i}",
        )

        sharp_samples.append(
            SharpSample(
                x=x,
                y=y,
                z=z,
                h=h,
                theta=theta,
            )
        )

    expect(
        "END_SHARP"
    )

    expect(
        ""
    )

    expect(
        SCHEMA_END
    )

    if cursor != len(lines):
        raise ValueError(
            f"{path}: trailing content after "
            f"{SCHEMA_END}"
        )

    ref = QualityRefV1(
        model=model,

        metric_contract_sha256=
            metric_contract_sha256,

        stage2_input_sha256=
            stage2_input_sha256,

        sharp_present=
            sharp_present,

        sharp_declared_count=
            sharp_declared_count,

        sharp_file_sha256=
            sharp_file_sha256,

        sharp_source_obj_sha256=
            sharp_source_obj_sha256,

        input_sample_seed_u64=
            input_sample_seed_u64,

        final_draw_seed_u64=
            final_draw_seed_u64,

        input_geometry=
            tuple(input_geometry),

        final_draws=
            tuple(final_draws),

        sharp_samples=
            tuple(sharp_samples),
    )

    validate_quality_ref_v1(
        ref,
        require_v1_sample_counts=
            require_v1_sample_counts,
    )

    if require_canonical_bytes:
        canonical = (
            quality_ref_v1_to_text(
                ref,
                require_v1_sample_counts=
                    require_v1_sample_counts,
            )
            .encode(
                "utf-8"
            )
        )

        if canonical != raw:
            raise ValueError(
                f"{path}: valid data but bytes "
                "are not canonical V1 encoding"
            )

    return ref
