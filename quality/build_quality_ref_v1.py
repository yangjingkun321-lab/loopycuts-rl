from __future__ import annotations

import argparse
import hashlib
import math
import os
import subprocess
import tempfile

from collections import Counter
from collections import defaultdict
from pathlib import Path

import numpy as np

from quality.quality_ref_v1 import (
    FINAL_DRAW_COUNT_V1,
    GEOMETRY_SAMPLE_COUNT_V1,
    METRIC_CONTRACT_V3_SHA256,
    FinalSurfaceDraw,
    InputGeometrySample,
    QualityRefV1,
    SharpSample,
    read_quality_ref_v1,
    sha256_file,
    write_quality_ref_v1,
)


SHARP_SAMPLE_SPACING_FACTOR = 0.25


def stable_seed(
    model: str,
    role: str,
) -> int:
    digest = hashlib.sha256(
        (
            "loopycuts_seed42_local_geometry42_v1"
            "|"
            + model
            + "|"
            + role
        ).encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        "little",
    )


def read_obj(
    path: str | Path,
):
    vertices = []
    faces = []

    with Path(path).open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:
        for raw in f:
            line = raw.strip()

            if (
                not line
                or line.startswith("#")
            ):
                continue

            if line.startswith("v "):
                t = line.split()

                vertices.append(
                    (
                        float(t[1]),
                        float(t[2]),
                        float(t[3]),
                    )
                )

            elif line.startswith("f "):
                face = []

                for token in line.split()[1:]:
                    idx = int(
                        token.split("/")[0]
                    )

                    if idx > 0:
                        idx -= 1
                    else:
                        idx = (
                            len(vertices)
                            + idx
                        )

                    face.append(idx)

                if len(face) < 3:
                    raise RuntimeError(
                        f"{path}: face <3 vertices"
                    )

                faces.append(
                    tuple(face)
                )

    vertices = np.asarray(
        vertices,
        dtype=np.float64,
    )

    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not faces
    ):
        raise RuntimeError(
            f"{path}: invalid OBJ"
        )

    return (
        vertices,
        faces,
    )


def triangulate_valid(
    vertices,
    faces,
):
    triangles = []

    for face in faces:
        for i in range(
            1,
            len(face) - 1,
        ):
            tri = (
                int(face[0]),
                int(face[i]),
                int(face[i + 1]),
            )

            if len(set(tri)) != 3:
                continue

            p = vertices[
                np.asarray(
                    tri,
                    dtype=np.int64,
                )
            ]

            area2 = float(
                np.linalg.norm(
                    np.cross(
                        p[1] - p[0],
                        p[2] - p[0],
                    )
                )
            )

            if area2 <= 1e-18:
                continue

            triangles.append(tri)

    if not triangles:
        raise RuntimeError(
            "No valid triangles"
        )

    return np.asarray(
        triangles,
        dtype=np.int64,
    )


def triangle_local_scales(
    vertices,
    triangles,
):
    xyz = vertices[
        triangles
    ]

    l01 = np.linalg.norm(
        xyz[:, 0] - xyz[:, 1],
        axis=1,
    )

    l12 = np.linalg.norm(
        xyz[:, 1] - xyz[:, 2],
        axis=1,
    )

    l20 = np.linalg.norm(
        xyz[:, 2] - xyz[:, 0],
        axis=1,
    )

    lengths = np.stack(
        [
            l01,
            l12,
            l20,
        ],
        axis=1,
    )

    scales = np.median(
        lengths,
        axis=1,
    )

    if (
        not np.isfinite(scales).all()
        or bool(
            (scales <= 1e-18).any()
        )
    ):
        raise RuntimeError(
            "Invalid local triangle scale"
        )

    return scales


def sample_surface(
    vertices,
    triangles,
    count: int,
    seed: int,
):
    xyz = vertices[
        triangles
    ]

    areas = (
        0.5
        *
        np.linalg.norm(
            np.cross(
                xyz[:, 1] - xyz[:, 0],
                xyz[:, 2] - xyz[:, 0],
            ),
            axis=1,
        )
    )

    if (
        not np.isfinite(areas).all()
        or float(
            np.sum(areas)
        ) <= 0.0
    ):
        raise RuntimeError(
            "Invalid triangle areas"
        )

    rng = np.random.default_rng(
        seed
    )

    triangle_ids = rng.choice(
        len(triangles),
        size=count,
        replace=True,
        p=areas / areas.sum(),
    )

    selected = xyz[
        triangle_ids
    ]

    u = rng.random(count)
    v = rng.random(count)

    su = np.sqrt(u)

    points = (
        (1.0 - su)[:, None]
        * selected[:, 0]
        +
        (
            su
            *
            (1.0 - v)
        )[:, None]
        * selected[:, 1]
        +
        (
            su
            *
            v
        )[:, None]
        * selected[:, 2]
    )

    return (
        points,
        triangle_ids,
    )


def make_final_draws(
    model: str,
):
    seed = stable_seed(
        model,
        "rl_final",
    )

    rng = np.random.default_rng(
        seed
    )

    #
    # This explicit sequence was audited on all 42 frozen
    # seed42 RL Final meshes and reproduced weighted
    # Generator.choice() triangle IDs and sample points exactly.
    #
    area_draw = rng.random(
        FINAL_DRAW_COUNT_V1
    )

    u = rng.random(
        FINAL_DRAW_COUNT_V1
    )

    v = rng.random(
        FINAL_DRAW_COUNT_V1
    )

    return (
        seed,
        tuple(
            FinalSurfaceDraw(
                area_draw=float(a),
                u=float(uu),
                v=float(vv),
            )
            for a, uu, vv
            in zip(
                area_draw,
                u,
                v,
                strict=True,
            )
        ),
    )


def face_normal(
    vertices,
    face,
):
    points = vertices[
        np.asarray(
            face,
            dtype=np.int64,
        )
    ]

    normal = np.zeros(
        3,
        dtype=np.float64,
    )

    for i, p in enumerate(points):
        q = points[
            (i + 1)
            %
            len(points)
        ]

        normal[0] += (
            (p[1] - q[1])
            *
            (p[2] + q[2])
        )

        normal[1] += (
            (p[2] - q[2])
            *
            (p[0] + q[0])
        )

        normal[2] += (
            (p[0] - q[0])
            *
            (p[1] + q[1])
        )

    length = float(
        np.linalg.norm(normal)
    )

    if length <= 1e-18:
        return None

    return (
        normal
        /
        length
    )


def mesh_edge_angles(
    vertices,
    faces,
):
    normals = [
        face_normal(
            vertices,
            face,
        )
        for face in faces
    ]

    adjacency = defaultdict(list)

    for face_id, face in enumerate(
        faces
    ):
        for i in range(
            len(face)
        ):
            a = int(face[i])

            b = int(
                face[
                    (i + 1)
                    %
                    len(face)
                ]
            )

            if a == b:
                continue

            edge = (
                (a, b)
                if a < b
                else (b, a)
            )

            adjacency[
                edge
            ].append(
                face_id
            )

    angles = {}

    for (
        edge,
        incident_faces,
    ) in adjacency.items():
        if len(
            incident_faces
        ) != 2:
            continue

        n0 = normals[
            incident_faces[0]
        ]

        n1 = normals[
            incident_faces[1]
        ]

        if (
            n0 is None
            or n1 is None
        ):
            continue

        dot = abs(
            float(
                np.dot(
                    n0,
                    n1,
                )
            )
        )

        dot = min(
            1.0,
            max(
                0.0,
                dot,
            ),
        )

        angles[edge] = math.degrees(
            math.acos(dot)
        )

    return angles


def read_sharp_declared_count(
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

    records = lines[1:]

    if declared != len(records):
        raise RuntimeError(
            f"{path}: "
            f"declared={declared}, "
            f"records={len(records)}"
        )

    return declared


def parse_sharp(
    path,
    vertices,
    faces,
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

    records = lines[1:]

    if declared != len(records):
        raise RuntimeError(
            f"{path}: "
            f"declared={declared}, "
            f"records={len(records)}"
        )

    edge_types = {}
    multiplicity = Counter()

    for line in records:
        values = [
            int(x)
            for x in (
                line
                .replace(",", " ")
                .split()
            )
        ]

        if len(values) < 3:
            raise RuntimeError(
                f"{path}: bad SHARP row"
            )

        (
            feature_type,
            face_id,
            edge_id,
        ) = values[:3]

        if feature_type not in (
            0,
            1,
        ):
            raise RuntimeError(
                f"{path}: bad SHARP type"
            )

        if not (
            0 <= face_id < len(faces)
        ):
            raise RuntimeError(
                f"{path}: bad face id"
            )

        if edge_id not in (
            0,
            1,
            2,
        ):
            raise RuntimeError(
                f"{path}: bad local edge"
            )

        face = faces[
            face_id
        ]

        if len(face) != 3:
            raise RuntimeError(
                f"{path}: SHARP source "
                "must be triangular"
            )

        a = int(
            face[edge_id]
        )

        b = int(
            face[
                (edge_id + 1)
                %
                3
            ]
        )

        edge = (
            (a, b)
            if a < b
            else (b, a)
        )

        if (
            edge in edge_types
            and
            edge_types[edge]
            != feature_type
        ):
            raise RuntimeError(
                f"{path}: conflicting "
                f"feature type for {edge}"
            )

        edge_types[
            edge
        ] = feature_type

        multiplicity[
            edge
        ] += 1

    bad = [
        (
            edge,
            count,
        )
        for edge, count
        in multiplicity.items()
        if count != 2
    ]

    if bad:
        raise RuntimeError(
            f"{path}: SHARP not "
            f"two-sided: {bad[:20]}"
        )

    return edge_types


def sample_sharp_with_targets(
    vertices,
    edges,
    source_angles,
    spacing,
):
    if (
        not math.isfinite(spacing)
        or spacing <= 0.0
    ):
        raise RuntimeError(
            "Invalid SHARP sample spacing"
        )

    point_parts = []
    angle_parts = []

    for edge in edges:
        a, b = edge

        p0 = vertices[a]
        p1 = vertices[b]

        length = float(
            np.linalg.norm(
                p1 - p0
            )
        )

        if length <= 1e-18:
            continue

        n = max(
            1,
            int(
                math.ceil(
                    length / spacing
                )
            ),
        )

        t = (
            np.arange(
                n,
                dtype=np.float64,
            )
            +
            0.5
        ) / float(n)

        points = (
            (1.0 - t)[:, None]
            * p0
            +
            t[:, None]
            * p1
        )

        target_angle = float(
            source_angles[
                edge
            ]
        )

        point_parts.append(
            points
        )

        angle_parts.append(
            np.full(
                n,
                target_angle,
                dtype=np.float64,
            )
        )

    if not point_parts:
        raise RuntimeError(
            "No SHARP samples"
        )

    return (
        np.concatenate(
            point_parts,
            axis=0,
        ),
        np.concatenate(
            angle_parts,
            axis=0,
        ),
    )


def write_surface(
    path,
    vertices,
    triangles,
):
    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{len(vertices)} "
            f"{len(triangles)}\n"
        )

        for p in vertices:
            f.write(
                f"{p[0]:.17g} "
                f"{p[1]:.17g} "
                f"{p[2]:.17g}\n"
            )

        for tri in triangles:
            f.write(
                f"{int(tri[0])} "
                f"{int(tri[1])} "
                f"{int(tri[2])}\n"
            )


def write_queries(
    path,
    points,
):
    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{len(points)}\n"
        )

        for p in points:
            f.write(
                f"{p[0]:.17g} "
                f"{p[1]:.17g} "
                f"{p[2]:.17g}\n"
            )


def run_point_triangle_backend(
    executable,
    surface_file,
    points,
    work_dir,
):
    query_file = (
        Path(work_dir)
        /
        "sharp_to_stage2_input_queries.txt"
    )

    output_file = (
        Path(work_dir)
        /
        "sharp_to_stage2_input_output.txt"
    )

    write_queries(
        query_file,
        points,
    )

    completed = subprocess.run(
        [
            str(executable),
            str(surface_file),
            str(query_file),
            str(output_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Point-triangle backend failed:\n"
            +
            completed.stdout
        )

    lines = output_file.read_text(
        encoding="utf-8"
    ).splitlines()

    if not lines:
        raise RuntimeError(
            "Empty point-triangle backend output"
        )

    count = int(
        lines[0]
    )

    if count != len(points):
        raise RuntimeError(
            "Point-triangle output count mismatch"
        )

    if len(lines) != count + 1:
        raise RuntimeError(
            "Point-triangle output rows mismatch"
        )

    nearest_ids = np.empty(
        count,
        dtype=np.int64,
    )

    distances = np.empty(
        count,
        dtype=np.float64,
    )

    for i, line in enumerate(
        lines[1:]
    ):
        values = line.split()

        if len(values) != 5:
            raise RuntimeError(
                "Malformed point-triangle row"
            )

        nearest_ids[i] = int(
            values[0]
        )

        distances[i] = float(
            values[1]
        )

    if (
        not np.isfinite(
            distances
        ).all()
        or bool(
            (distances < 0.0).any()
        )
    ):
        raise RuntimeError(
            "Invalid point-triangle distances"
        )

    return (
        nearest_ids,
        distances,
    )


def build_ref(
    *,
    model,
    stage2_input,
    sharp_file,
    sharp_source_obj,
    point_triangle_backend,
):
    stage2_input = Path(
        stage2_input
    )

    if not stage2_input.is_file():
        raise RuntimeError(
            f"Missing Stage2 input: "
            f"{stage2_input}"
        )

    (
        input_vertices,
        input_faces,
    ) = read_obj(
        stage2_input
    )

    input_triangles = (
        triangulate_valid(
            input_vertices,
            input_faces,
        )
    )

    input_scales = (
        triangle_local_scales(
            input_vertices,
            input_triangles,
        )
    )

    input_seed = stable_seed(
        model,
        "input",
    )

    (
        input_points,
        input_triangle_ids,
    ) = sample_surface(
        input_vertices,
        input_triangles,
        GEOMETRY_SAMPLE_COUNT_V1,
        input_seed,
    )

    input_sample_scales = (
        input_scales[
            input_triangle_ids
        ]
    )

    input_geometry = tuple(
        InputGeometrySample(
            x=float(p[0]),
            y=float(p[1]),
            z=float(p[2]),
            h=float(h),
        )
        for p, h
        in zip(
            input_points,
            input_sample_scales,
            strict=True,
        )
    )

    (
        final_draw_seed,
        final_draws,
    ) = make_final_draws(
        model
    )

    sharp_present = False
    sharp_declared_count = None

    sharp_samples = ()

    sharp_file_sha256 = None
    sharp_source_obj_sha256 = None

    if sharp_file is not None:

        if sharp_source_obj is None:
            raise RuntimeError(
                "SHARP source OBJ is required"
            )

        sharp_file = Path(
            sharp_file
        )

        sharp_source_obj = Path(
            sharp_source_obj
        )

        if not sharp_file.is_file():
            raise RuntimeError(
                f"Missing SHARP: {sharp_file}"
            )

        if not sharp_source_obj.is_file():
            raise RuntimeError(
                "Missing SHARP source OBJ: "
                f"{sharp_source_obj}"
            )

        sharp_declared_count = (
            read_sharp_declared_count(
                sharp_file
            )
        )

        sharp_file_sha256 = (
            sha256_file(
                sharp_file
            )
        )

        sharp_source_obj_sha256 = (
            sha256_file(
                sharp_source_obj
            )
        )

        (
            sharp_vertices,
            sharp_faces,
        ) = read_obj(
            sharp_source_obj
        )

        edge_types = parse_sharp(
            sharp_file,
            sharp_vertices,
            sharp_faces,
        )

        if sharp_declared_count == 0:

            if edge_types:
                raise RuntimeError(
                    "Zero-declared SHARP unexpectedly "
                    "produced feature edges"
                )

            sharp_present = False

        else:

            sharp_present = True

            if point_triangle_backend is None:
                raise RuntimeError(
                    "Point-triangle backend is "
                    "required for positive-count "
                    "SHARP reference generation"
                )

            point_triangle_backend = Path(
                point_triangle_backend
            )

            if not point_triangle_backend.is_file():
                raise RuntimeError(
                    "Missing point-triangle backend: "
                    f"{point_triangle_backend}"
                )

            if not os.access(
                point_triangle_backend,
                os.X_OK,
            ):
                raise RuntimeError(
                    "Point-triangle backend is "
                    "not executable"
                )

            source_angles = (
                mesh_edge_angles(
                    sharp_vertices,
                    sharp_faces,
                )
            )

            missing = [
                edge
                for edge in edge_types
                if edge not in source_angles
            ]

            if missing:
                raise RuntimeError(
                    f"{model}: missing source "
                    f"angles {missing[:10]}"
                )

            global_input_scale = float(
                np.median(
                    input_scales
                )
            )

            spacing = (
                SHARP_SAMPLE_SPACING_FACTOR
                *
                global_input_scale
            )

            (
                sharp_points,
                target_angles,
            ) = sample_sharp_with_targets(
                sharp_vertices,
                edge_types.keys(),
                source_angles,
                spacing,
            )

            if (
                not np.isfinite(
                    target_angles
                ).all()
                or bool(
                    (
                        target_angles
                        <=
                        0.0
                    ).any()
                )
                or bool(
                    (
                        target_angles
                        >
                        90.0
                    ).any()
                )
            ):
                raise RuntimeError(
                    "Invalid continuous-SHARP "
                    "reference angle"
                )

            with tempfile.TemporaryDirectory(
                prefix=
                    "loopycuts_quality_ref_v1_"
                    f"{model}_"
            ) as tmp_dir:

                surface_file = (
                    Path(tmp_dir)
                    /
                    "stage2_input_surface.txt"
                )

                write_surface(
                    surface_file,
                    input_vertices,
                    input_triangles,
                )

                (
                    nearest_ids,
                    _
                ) = run_point_triangle_backend(
                    point_triangle_backend,
                    surface_file,
                    sharp_points,
                    tmp_dir,
                )

            if (
                bool(
                    (
                        nearest_ids
                        <
                        0
                    ).any()
                )
                or bool(
                    (
                        nearest_ids
                        >=
                        len(
                            input_triangles
                        )
                    ).any()
                )
            ):
                raise RuntimeError(
                    "Bad nearest Stage2 "
                    "Input triangle"
                )

            sharp_local_h = (
                input_scales[
                    nearest_ids
                ]
            )

            sharp_samples = tuple(
                SharpSample(
                    x=float(p[0]),
                    y=float(p[1]),
                    z=float(p[2]),
                    h=float(h),
                    theta=float(theta),
                )
                for p, h, theta
                in zip(
                    sharp_points,
                    sharp_local_h,
                    target_angles,
                    strict=True,
                )
            )

    return QualityRefV1(
        model=model,

        metric_contract_sha256=
            METRIC_CONTRACT_V3_SHA256,

        stage2_input_sha256=
            sha256_file(
                stage2_input
            ),

        sharp_present=
            sharp_present,

        sharp_declared_count=
            sharp_declared_count,

        sharp_file_sha256=
            sharp_file_sha256,

        sharp_source_obj_sha256=
            sharp_source_obj_sha256,

        input_sample_seed_u64=
            input_seed,

        final_draw_seed_u64=
            final_draw_seed,

        input_geometry=
            input_geometry,

        final_draws=
            final_draws,

        sharp_samples=
            sharp_samples,
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--stage2-input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--point-triangle-backend",
        default=None,
    )

    sharp_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    sharp_group.add_argument(
        "--sharp-file",
        default=None,
    )

    sharp_group.add_argument(
        "--no-sharp",
        action="store_true",
    )

    parser.add_argument(
        "--sharp-source-obj",
        default=None,
    )

    args = parser.parse_args()

    if args.no_sharp:
        if args.sharp_source_obj is not None:
            parser.error(
                "--no-sharp cannot be combined "
                "with --sharp-source-obj"
            )

        sharp_file = None
        sharp_source_obj = None

    else:
        if args.sharp_source_obj is None:
            parser.error(
                "--sharp-file requires "
                "--sharp-source-obj"
            )

        sharp_file = args.sharp_file
        sharp_source_obj = (
            args.sharp_source_obj
        )

    ref = build_ref(
        model=args.model,
        stage2_input=
            args.stage2_input,
        sharp_file=
            sharp_file,
        sharp_source_obj=
            sharp_source_obj,
        point_triangle_backend=
            args.point_triangle_backend,
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output.with_name(
        output.name
        +
        ".tmp"
    )

    try:
        write_quality_ref_v1(
            temporary,
            ref,
            require_v1_sample_counts=True,
        )

        loaded = read_quality_ref_v1(
            temporary,
            require_v1_sample_counts=True,
            require_canonical_bytes=True,
        )

        if loaded != ref:
            raise RuntimeError(
                "Generated ref failed "
                "object round-trip"
            )

        os.replace(
            temporary,
            output,
        )

    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        "PASS: built quality_ref_v1"
    )

    print(
        "model =",
        ref.model,
    )

    print(
        "input_sample_seed_u64 =",
        ref.input_sample_seed_u64,
    )

    print(
        "final_draw_seed_u64 =",
        ref.final_draw_seed_u64,
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
        "sharp_declared_count =",
        ref.sharp_declared_count,
    )

    print(
        "sharp_samples =",
        len(ref.sharp_samples),
    )

    print(
        "output =",
        output,
    )

    print(
        "sha256 =",
        sha256_file(output),
    )

    print(
        "bytes =",
        output.stat().st_size,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
