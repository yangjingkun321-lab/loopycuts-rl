from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


CORPUS_CSV = Path(
    "/home/yjk/loopycuts_test/"
    "rl_corpus/"
    "loop_corpus.csv"
)

EXTERNAL_CYLINDER = Path(
    "/home/yjk/loopycuts_inputs/"
    "cylinder_plate_clean/"
    "cylinder_plate_rem_splitted.obj"
)


AUTHOR_MODEL_NAMES = [
    "cylinder_plate",
    "bimba",
    "busto_bimba",
    "Plate1",
    "Plate2",
    "Plate3",
    "Plate4",
    "tris_open",
    "tris_closed",
    "bone1",
    "bone_femur",
]


def load_author_mesh_paths():
    """
    Resolve author meshes from the already audited corpus manifest.

    IMPORTANT:
        Do NOT derive mesh filenames from directory/model names.
        LoopyCuts test_data contains models whose actual filenames
        do not follow a uniform naming convention.
    """

    import csv

    if not CORPUS_CSV.is_file():
        raise FileNotFoundError(
            CORPUS_CSV
        )

    rows = {}

    with CORPUS_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(
            f
        )

        required = {
            "model",
            "mesh_file",
            "mesh_exists",
        }

        missing_columns = (
            required
            -
            set(
                reader.fieldnames
                or []
            )
        )

        if missing_columns:
            raise RuntimeError(
                "Corpus CSV is missing fields: "
                +
                ", ".join(
                    sorted(
                        missing_columns
                    )
                )
            )

        for row in reader:
            model = row[
                "model"
            ]

            if model in rows:
                raise RuntimeError(
                    f"Duplicate corpus model: "
                    f"{model!r}"
                )

            rows[
                model
            ] = row

    missing_models = (
        set(
            AUTHOR_MODEL_NAMES
        )
        -
        set(
            rows
        )
    )

    if missing_models:
        raise RuntimeError(
            "Candidate family audit models are "
            "missing from corpus: "
            +
            ", ".join(
                sorted(
                    missing_models
                )
            )
        )

    result = {}

    for model in (
        AUTHOR_MODEL_NAMES
    ):
        row = rows[
            model
        ]

        mesh_file = Path(
            row[
                "mesh_file"
            ]
        )

        mesh_exists_flag = (
            str(
                row[
                    "mesh_exists"
                ]
            ).strip().lower()
            in {
                "true",
                "1",
                "yes",
            }
        )

        if not mesh_exists_flag:
            raise RuntimeError(
                f"Corpus says mesh does not exist "
                f"for {model}: {mesh_file}"
            )

        if not mesh_file.is_file():
            raise FileNotFoundError(
                f"Corpus mesh path does not exist "
                f"for {model}: {mesh_file}"
            )

        result[
            model
        ] = mesh_file

    return result


AUTHOR_MESHES = (
    load_author_mesh_paths()
)


MODELS = {
    "cylinder_author":
        AUTHOR_MESHES[
            "cylinder_plate"
        ],

    "cylinder_external":
        EXTERNAL_CYLINDER,

    "bimba":
        AUTHOR_MESHES[
            "bimba"
        ],

    "busto_bimba":
        AUTHOR_MESHES[
            "busto_bimba"
        ],

    "Plate1":
        AUTHOR_MESHES[
            "Plate1"
        ],

    "Plate2":
        AUTHOR_MESHES[
            "Plate2"
        ],

    "Plate3":
        AUTHOR_MESHES[
            "Plate3"
        ],

    "Plate4":
        AUTHOR_MESHES[
            "Plate4"
        ],

    "tris_open":
        AUTHOR_MESHES[
            "tris_open"
        ],

    "tris_closed":
        AUTHOR_MESHES[
            "tris_closed"
        ],

    "bone1":
        AUTHOR_MESHES[
            "bone1"
        ],

    "bone_femur":
        AUTHOR_MESHES[
            "bone_femur"
        ],
}


GROUPS = {
    "cylinder": [
        "cylinder_author",
        "cylinder_external",
    ],

    "bimba": [
        "bimba",
        "busto_bimba",
    ],

    "plate": [
        "Plate1",
        "Plate2",
        "Plate3",
        "Plate4",
    ],

    "tris": [
        "tris_open",
        "tris_closed",
    ],

    "bone": [
        "bone1",
        "bone_femur",
    ],
}


SEED = 20260813
NUM_SURFACE_SAMPLES = 30000


def read_obj(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    verts = []
    triangles = []

    with path.open(
        "r",
        errors="ignore",
    ) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()

                if len(parts) >= 4:
                    verts.append(
                        (
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                        )
                    )

            elif line.startswith("f "):
                parts = line.split()[1:]

                ids = []

                for token in parts:
                    raw = token.split("/")[0]

                    if not raw:
                        continue

                    idx = int(raw)

                    if idx > 0:
                        idx -= 1
                    else:
                        idx = len(verts) + idx

                    ids.append(idx)

                if len(ids) < 3:
                    continue

                #
                # Fan triangulation for polygons.
                #
                for i in range(
                    1,
                    len(ids) - 1,
                ):
                    triangles.append(
                        (
                            ids[0],
                            ids[i],
                            ids[i + 1],
                        )
                    )

    vertices = np.asarray(
        verts,
        dtype=np.float64,
    )

    faces = np.asarray(
        triangles,
        dtype=np.int64,
    )

    if (
        vertices.ndim != 2
        or
        vertices.shape[1] != 3
        or
        len(vertices) == 0
    ):
        raise RuntimeError(
            f"Invalid vertex array: {path}"
        )

    if (
        faces.ndim != 2
        or
        faces.shape[1] != 3
        or
        len(faces) == 0
    ):
        raise RuntimeError(
            f"No triangular surface available: {path}"
        )

    return vertices, faces


def mesh_descriptor(
    vertices,
    faces,
):
    bbox_min = vertices.min(
        axis=0
    )

    bbox_max = vertices.max(
        axis=0
    )

    extents = (
        bbox_max
        -
        bbox_min
    )

    diagonal = float(
        np.linalg.norm(
            extents
        )
    )

    if diagonal <= 0.0:
        raise RuntimeError(
            "Degenerate bounding box"
        )

    sorted_extents = np.sort(
        extents
        /
        diagonal
    )

    return {
        "vertices":
            int(
                len(vertices)
            ),

        "triangles":
            int(
                len(faces)
            ),

        "diagonal":
            diagonal,

        "extent_norm_x":
            float(
                extents[0]
                /
                diagonal
            ),

        "extent_norm_y":
            float(
                extents[1]
                /
                diagonal
            ),

        "extent_norm_z":
            float(
                extents[2]
                /
                diagonal
            ),

        "sorted_extent_0":
            float(
                sorted_extents[0]
            ),

        "sorted_extent_1":
            float(
                sorted_extents[1]
            ),

        "sorted_extent_2":
            float(
                sorted_extents[2]
            ),
    }


def normalize_vertices(
    vertices,
):
    bbox_min = vertices.min(
        axis=0
    )

    bbox_max = vertices.max(
        axis=0
    )

    center = (
        bbox_min
        +
        bbox_max
    ) * 0.5

    diagonal = float(
        np.linalg.norm(
            bbox_max
            -
            bbox_min
        )
    )

    if diagonal <= 0.0:
        raise RuntimeError(
            "Degenerate mesh"
        )

    return (
        vertices - center
    ) / diagonal


def sample_surface(
    vertices,
    faces,
    *,
    seed,
    count,
):
    v0 = vertices[
        faces[:, 0]
    ]

    v1 = vertices[
        faces[:, 1]
    ]

    v2 = vertices[
        faces[:, 2]
    ]

    cross = np.cross(
        v1 - v0,
        v2 - v0,
    )

    area = (
        0.5
        *
        np.linalg.norm(
            cross,
            axis=1,
        )
    )

    valid = (
        area
        >
        0.0
    )

    if not bool(
        valid.any()
    ):
        raise RuntimeError(
            "All triangles are degenerate"
        )

    v0 = v0[valid]
    v1 = v1[valid]
    v2 = v2[valid]
    area = area[valid]

    probability = (
        area
        /
        area.sum()
    )

    rng = np.random.default_rng(
        seed
    )

    triangle_ids = rng.choice(
        len(area),
        size=count,
        replace=True,
        p=probability,
    )

    a = rng.random(
        count
    )

    b = rng.random(
        count
    )

    #
    # Uniform barycentric surface sampling.
    #
    sqrt_a = np.sqrt(
        a
    )

    w0 = (
        1.0
        -
        sqrt_a
    )

    w1 = (
        sqrt_a
        *
        (
            1.0
            -
            b
        )
    )

    w2 = (
        sqrt_a
        *
        b
    )

    samples = (
        w0[:, None]
        *
        v0[
            triangle_ids
        ]
        +
        w1[:, None]
        *
        v1[
            triangle_ids
        ]
        +
        w2[:, None]
        *
        v2[
            triangle_ids
        ]
    )

    return samples


def nearest_statistics(
    source,
    target,
):
    tree = cKDTree(
        target
    )

    distances, _ = tree.query(
        source,
        k=1,
        workers=-1,
    )

    return {
        "mean":
            float(
                distances.mean()
            ),

        "median":
            float(
                np.median(
                    distances
                )
            ),

        "p95":
            float(
                np.percentile(
                    distances,
                    95,
                )
            ),

        "p99":
            float(
                np.percentile(
                    distances,
                    99,
                )
            ),

        "max":
            float(
                distances.max()
            ),
    }


def compare(
    name_a,
    mesh_a,
    name_b,
    mesh_b,
):
    points_a = mesh_a[
        "samples"
    ]

    points_b = mesh_b[
        "samples"
    ]

    ab = nearest_statistics(
        points_a,
        points_b,
    )

    ba = nearest_statistics(
        points_b,
        points_a,
    )

    symmetric = {
        key:
            max(
                ab[key],
                ba[key],
            )
        for key in ab
    }

    desc_a = mesh_a[
        "descriptor"
    ]

    desc_b = mesh_b[
        "descriptor"
    ]

    extent_a = np.array(
        [
            desc_a[
                "sorted_extent_0"
            ],
            desc_a[
                "sorted_extent_1"
            ],
            desc_a[
                "sorted_extent_2"
            ],
        ]
    )

    extent_b = np.array(
        [
            desc_b[
                "sorted_extent_0"
            ],
            desc_b[
                "sorted_extent_1"
            ],
            desc_b[
                "sorted_extent_2"
            ],
        ]
    )

    extent_difference = float(
        np.linalg.norm(
            extent_a
            -
            extent_b
        )
    )

    print(
        f"{name_a:20s} <-> "
        f"{name_b:20s}  "
        f"extent_delta={extent_difference:.6f}  "
        f"NN median={symmetric['median']:.6f}  "
        f"p95={symmetric['p95']:.6f}  "
        f"p99={symmetric['p99']:.6f}  "
        f"max={symmetric['max']:.6f}"
    )


def main():
    loaded = {}

    print(
        "=" * 110
    )

    print(
        "MESH DESCRIPTORS"
    )

    print(
        "=" * 110
    )

    for index, (
        name,
        path,
    ) in enumerate(
        MODELS.items()
    ):
        vertices, faces = (
            read_obj(
                path
            )
        )

        descriptor = (
            mesh_descriptor(
                vertices,
                faces,
            )
        )

        normalized = (
            normalize_vertices(
                vertices
            )
        )

        samples = (
            sample_surface(
                normalized,
                faces,
                seed=(
                    SEED
                    +
                    index
                ),
                count=NUM_SURFACE_SAMPLES,
            )
        )

        loaded[
            name
        ] = {
            "descriptor":
                descriptor,

            "samples":
                samples,
        }

        print(
            f"{name:20s}  "
            f"V={descriptor['vertices']:7d}  "
            f"F={descriptor['triangles']:7d}  "
            f"diag={descriptor['diagonal']:.6g}  "
            f"extent(sorted)="
            f"("
            f"{descriptor['sorted_extent_0']:.4f}, "
            f"{descriptor['sorted_extent_1']:.4f}, "
            f"{descriptor['sorted_extent_2']:.4f}"
            f")"
        )

    print()
    print(
        "=" * 110
    )

    print(
        "WITHIN-CANDIDATE-FAMILY NORMALIZED SURFACE DISTANCES"
    )

    print(
        "=" * 110
    )

    for family, members in (
        GROUPS.items()
    ):
        print()
        print(
            f"[{family}]"
        )

        for i in range(
            len(
                members
            )
        ):
            for j in range(
                i + 1,
                len(
                    members
                ),
            ):
                a = members[i]
                b = members[j]

                compare(
                    a,
                    loaded[a],
                    b,
                    loaded[b],
                )


if __name__ == "__main__":
    main()
