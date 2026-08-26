from __future__ import annotations

import argparse

from pathlib import Path

from quality.quality_ref_v1 import (
    read_quality_ref_v1,
    sha256_file,
)


def read_declared_count(
    path,
):
    lines = [
        line.strip()
        for line in Path(
            path
        ).read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if (
            line.strip()
            and
            not line.lstrip().startswith(
                "#"
            )
        )
    ]

    if not lines:
        raise RuntimeError(
            f"{path}: empty SHARP"
        )

    declared = int(
        lines[0]
    )

    if declared < 0:
        raise RuntimeError(
            f"{path}: negative SHARP count"
        )

    if declared != len(
        lines[1:]
    ):
        raise RuntimeError(
            f"{path}: declared count "
            "does not match records"
        )

    return declared


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "quality_ref",
    )

    parser.add_argument(
        "--stage2-input",
        default=None,
    )

    parser.add_argument(
        "--sharp-file",
        default=None,
    )

    parser.add_argument(
        "--sharp-source-obj",
        default=None,
    )

    args = parser.parse_args()

    ref = read_quality_ref_v1(
        args.quality_ref,
        require_v1_sample_counts=True,
        require_canonical_bytes=True,
    )


    # ========================================================
    # Stage2 provenance.
    # ========================================================

    if args.stage2_input is not None:

        actual = sha256_file(
            args.stage2_input
        )

        if actual != ref.stage2_input_sha256:
            raise RuntimeError(
                "Stage2 input SHA mismatch: "
                f"ref={ref.stage2_input_sha256} "
                f"actual={actual}"
            )


    # ========================================================
    # SHARP provenance.
    #
    # None:
    #   no SHARP source exists.
    #
    # 0:
    #   explicit SHARP source exists and declares zero records.
    #
    # >0:
    #   explicit active SHARP source.
    # ========================================================

    if ref.sharp_declared_count is None:

        if (
            args.sharp_file is not None
            or
            args.sharp_source_obj is not None
        ):
            raise RuntimeError(
                "Ref declares no SHARP source "
                "but SHARP provenance arguments "
                "were supplied"
            )

    else:

        if (
            args.sharp_file is None
            or
            args.sharp_source_obj is None
        ):
            raise RuntimeError(
                "Ref contains explicit SHARP "
                "provenance; --sharp-file and "
                "--sharp-source-obj are required"
            )

        actual_sharp_sha = sha256_file(
            args.sharp_file
        )

        actual_obj_sha = sha256_file(
            args.sharp_source_obj
        )

        if (
            actual_sharp_sha
            !=
            ref.sharp_file_sha256
        ):
            raise RuntimeError(
                "SHARP SHA mismatch"
            )

        if (
            actual_obj_sha
            !=
            ref.sharp_source_obj_sha256
        ):
            raise RuntimeError(
                "SHARP source OBJ SHA mismatch"
            )

        actual_declared = (
            read_declared_count(
                args.sharp_file
            )
        )

        if (
            actual_declared
            !=
            ref.sharp_declared_count
        ):
            raise RuntimeError(
                "SHARP declared-count mismatch: "
                f"ref={ref.sharp_declared_count} "
                f"actual={actual_declared}"
            )

        actual_present = (
            actual_declared > 0
        )

        if actual_present != ref.sharp_present:
            raise RuntimeError(
                "SHARP_PRESENT mismatch"
            )


    print(
        "PASS: quality_ref_v1"
    )

    print(
        "model =",
        ref.model,
    )

    print(
        "geometry_samples =",
        len(ref.input_geometry),
    )

    print(
        "final_draws =",
        len(ref.final_draws),
    )

    print(
        "sharp_present =",
        int(ref.sharp_present),
    )

    print(
        "sharp_declared_count =",
        ref.sharp_declared_count,
    )

    print(
        "sharp_samples =",
        len(ref.sharp_samples),
    )

    print(
        "quality_ref_sha256 =",
        sha256_file(
            args.quality_ref
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
