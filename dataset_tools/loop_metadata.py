from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VALID_LOOP_TYPES = {
    "CONCAVE",
    "REGULAR",
    "CONVEX",
}


@dataclass(frozen=True)
class LoopMetadata:
    """
    Static metadata serialized in a LoopyCuts Stage-1 _loop.txt file.

    Important:
        loop_type is the ORIGINAL SERIALIZED type from _loop.txt.

    Stage-2 may later change some loop objects dynamically, for example
    to TOP_RELEVANT. Such runtime state must come from the C++ RL server
    and is deliberately not represented here.
    """

    loop_id: int

    loop_type: str

    closed: bool

    flawed: bool

    num_segments: int

    num_sharp_segments: int

    sharp_fraction: float

    serialized_position: float


class LoopMetadataParseError(ValueError):
    pass


def _require_line(
    lines: list[str],
    cursor: int,
    *,
    context: str,
) -> tuple[str, int]:
    if cursor >= len(lines):
        raise LoopMetadataParseError(
            f"Unexpected end of loop file while reading {context}"
        )

    return lines[cursor], cursor + 1


def parse_loop_metadata(
    loop_file: str | Path,
) -> list[LoopMetadata]:
    """
    Parse only the static metadata needed by the RL observation layer.

    The parser follows the exact format emitted by
    LoopSplitter::SaveLoopInfo():

        num_loops

        TYPE
        Closed|Open
        Cross OK|Cross FAIL
        num_segments

        face_id edge_offset sharp_flag
        ... repeated num_segments times

    It intentionally does NOT try to reconstruct Stage-2 runtime state
    such as used, reverted, Nico_bug or TOP_RELEVANT.
    """

    path = Path(loop_file)

    if not path.is_file():
        raise FileNotFoundError(
            f"Loop file does not exist: {path}"
        )

    #
    # SaveLoopInfo() emits one logical item per line.
    # Empty lines are not part of the author format, so reject them
    # instead of silently changing line positions.
    #
    raw_lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    if not raw_lines:
        raise LoopMetadataParseError(
            f"Loop file is empty: {path}"
        )

    lines = [
        line.strip()
        for line in raw_lines
    ]

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        if line == "":
            raise LoopMetadataParseError(
                f"Unexpected empty line at "
                f"{path}:{line_number}"
            )

    # ------------------------------------------------------------
    # Number of serialized loops
    # ------------------------------------------------------------

    try:
        num_loops = int(
            lines[0]
        )
    except ValueError as exc:
        raise LoopMetadataParseError(
            f"First line must contain the loop count, "
            f"got {lines[0]!r}"
        ) from exc

    if num_loops < 0:
        raise LoopMetadataParseError(
            f"Loop count must be non-negative, got {num_loops}"
        )

    cursor = 1

    metadata: list[LoopMetadata] = []

    # ------------------------------------------------------------
    # Individual loops
    # ------------------------------------------------------------

    for loop_id in range(
        num_loops
    ):
        loop_type_line, cursor = _require_line(
            lines,
            cursor,
            context=f"loop {loop_id} type",
        )

        loop_type = (
            loop_type_line
        )

        if loop_type not in VALID_LOOP_TYPES:
            raise LoopMetadataParseError(
                f"Loop {loop_id}: invalid type "
                f"{loop_type!r}"
            )

        # --------------------------------------------------------

        closed_line, cursor = _require_line(
            lines,
            cursor,
            context=f"loop {loop_id} Closed/Open flag",
        )

        if closed_line == "Closed":
            closed = True

        elif closed_line == "Open":
            closed = False

        else:
            raise LoopMetadataParseError(
                f"Loop {loop_id}: expected "
                f"'Closed' or 'Open', "
                f"got {closed_line!r}"
            )

        # --------------------------------------------------------

        cross_line, cursor = _require_line(
            lines,
            cursor,
            context=f"loop {loop_id} Cross status",
        )

        if cross_line == "Cross OK":
            flawed = False

        elif cross_line == "Cross FAIL":
            flawed = True

        else:
            raise LoopMetadataParseError(
                f"Loop {loop_id}: expected "
                f"'Cross OK' or 'Cross FAIL', "
                f"got {cross_line!r}"
            )

        # --------------------------------------------------------

        num_segments_line, cursor = _require_line(
            lines,
            cursor,
            context=f"loop {loop_id} segment count",
        )

        try:
            num_segments = int(
                num_segments_line
            )
        except ValueError as exc:
            raise LoopMetadataParseError(
                f"Loop {loop_id}: invalid segment count "
                f"{num_segments_line!r}"
            ) from exc

        if num_segments < 0:
            raise LoopMetadataParseError(
                f"Loop {loop_id}: segment count must be "
                f"non-negative, got {num_segments}"
            )

        # --------------------------------------------------------
        # Segment records:
        #
        #     face_id edge_offset sharp_flag
        #
        # We do not need face_id/edge_offset for Observation V1,
        # but still validate every row strictly so a malformed file
        # cannot silently produce incorrect metadata.
        # --------------------------------------------------------

        num_sharp_segments = 0

        for segment_id in range(
            num_segments
        ):
            segment_line, cursor = _require_line(
                lines,
                cursor,
                context=(
                    f"loop {loop_id} "
                    f"segment {segment_id}"
                ),
            )

            tokens = (
                segment_line.split()
            )

            if len(tokens) != 3:
                raise LoopMetadataParseError(
                    f"Loop {loop_id}, segment {segment_id}: "
                    f"expected 3 integers, got "
                    f"{segment_line!r}"
                )

            try:
                face_id = int(
                    tokens[0]
                )

                edge_offset = int(
                    tokens[1]
                )

                sharp_flag = int(
                    tokens[2]
                )

            except ValueError as exc:
                raise LoopMetadataParseError(
                    f"Loop {loop_id}, segment {segment_id}: "
                    f"non-integer segment record "
                    f"{segment_line!r}"
                ) from exc

            if face_id < 0:
                raise LoopMetadataParseError(
                    f"Loop {loop_id}, segment {segment_id}: "
                    f"negative face id {face_id}"
                )

            if edge_offset < 0:
                raise LoopMetadataParseError(
                    f"Loop {loop_id}, segment {segment_id}: "
                    f"negative edge offset {edge_offset}"
                )

            if sharp_flag not in (
                0,
                1,
            ):
                raise LoopMetadataParseError(
                    f"Loop {loop_id}, segment {segment_id}: "
                    f"sharp flag must be 0 or 1, "
                    f"got {sharp_flag}"
                )

            num_sharp_segments += (
                sharp_flag
            )

        # --------------------------------------------------------

        if num_segments == 0:
            sharp_fraction = 0.0

        else:
            sharp_fraction = (
                num_sharp_segments
                /
                num_segments
            )

        if num_loops <= 1:
            serialized_position = 0.0

        else:
            serialized_position = (
                loop_id
                /
                (num_loops - 1)
            )

        metadata.append(
            LoopMetadata(
                loop_id=loop_id,
                loop_type=loop_type,
                closed=closed,
                flawed=flawed,
                num_segments=num_segments,
                num_sharp_segments=(
                    num_sharp_segments
                ),
                sharp_fraction=(
                    sharp_fraction
                ),
                serialized_position=(
                    serialized_position
                ),
            )
        )

    # ------------------------------------------------------------
    # Strictly reject trailing data.
    # ------------------------------------------------------------

    if cursor != len(lines):
        raise LoopMetadataParseError(
            f"Unexpected trailing data after "
            f"{num_loops} loops: "
            f"{len(lines) - cursor} extra line(s)"
        )

    return metadata
