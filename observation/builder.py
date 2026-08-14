from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from dataset_tools.loop_metadata import LoopMetadata


MAX_LOOPS = 331

GLOBAL_DIM = 16
LOOP_FEATURE_DIM = 14


class ObservationBuildError(ValueError):
    pass


class LoopyCutsObservationBuilder:
    """
    Build the fixed-size RL observation from:

        1. static _loop.txt metadata;
        2. current C++ RL server state;
        3. authoritative C++ ACTIONS;
        4. dynamic per-loop status;
        5. Python episode execution history.

    This class performs NO LoopyCuts geometry and contains NO action
    legality rules. `actions` from the C++ RL server are authoritative.
    """

    def __init__(
        self,
        *,
        metadata: Sequence[LoopMetadata],
        initial_state: dict,
        initial_actions: Sequence[int],
        max_loops: int = MAX_LOOPS,
    ):
        self.metadata = list(metadata)

        self.num_loops = len(
            self.metadata
        )

        self.max_loops = int(
            max_loops
        )

        if self.max_loops <= 0:
            raise ObservationBuildError(
                "max_loops must be positive"
            )

        if self.num_loops > self.max_loops:
            raise ObservationBuildError(
                f"Model has {self.num_loops} loops, "
                f"but MAX_LOOPS={self.max_loops}. "
                "Do not truncate or remap loop IDs."
            )

        # ------------------------------------------------------------
        # Serialized loop IDs must exactly be 0..N-1.
        # ------------------------------------------------------------

        expected_ids = list(
            range(
                self.num_loops
            )
        )

        actual_ids = [
            loop.loop_id
            for loop in self.metadata
        ]

        if actual_ids != expected_ids:
            raise ObservationBuildError(
                "Loop metadata IDs are not exactly contiguous "
                "serialization IDs 0..N-1"
            )

        # ------------------------------------------------------------
        # Freeze initial normalization references.
        # ------------------------------------------------------------

        self._validate_state(
            initial_state
        )

        self.initial_verts = int(
            initial_state[
                "verts"
            ]
        )

        self.initial_tets = int(
            initial_state[
                "tets"
            ]
        )

        if self.initial_verts <= 0:
            raise ObservationBuildError(
                "Initial vertex count must be positive"
            )

        if self.initial_tets <= 0:
            raise ObservationBuildError(
                "Initial tet count must be positive"
            )

        initial_actions = self._validate_id_list(
            initial_actions,
            name="initial_actions",
        )

        if len(initial_actions) == 0:
            raise ObservationBuildError(
                "Initial state has no legal actions"
            )

        self.initial_actionable_count = len(
            initial_actions
        )

        if (
            int(
                initial_state[
                    "available"
                ]
            )
            !=
            self.initial_actionable_count
        ):
            raise ObservationBuildError(
                "initial_state['available'] does not match "
                "the C++ initial ACTIONS list"
            )

    # ------------------------------------------------------------------

    def _validate_state(
        self,
        state: dict,
    ) -> None:
        required = {
            "step",
            "loops",
            "available",
            "verts",
            "tets",
            "mm_verts",
            "mm_edges",
            "mm_faces",
            "mm_polys",
            "converged",
            "regular_phase_closed",
            "terminal",
            "selection_success",
            "finalized",
            "diagnostics_valid",
            "nonmanifold_polys",
            "high_genus_polys",
            "buggy_chains",
        }

        missing = (
            required
            -
            set(
                state.keys()
            )
        )

        if missing:
            raise ObservationBuildError(
                "C++ STATE is missing fields: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        if int(
            state[
                "loops"
            ]
        ) != self.num_loops:
            raise ObservationBuildError(
                "C++ STATE loop count does not match "
                "_loop.txt metadata"
            )

        nonnegative_fields = (
            "step",
            "available",
            "verts",
            "tets",
            "mm_verts",
            "mm_edges",
            "mm_faces",
            "mm_polys",
            "nonmanifold_polys",
            "high_genus_polys",
            "buggy_chains",
        )

        for name in nonnegative_fields:
            if int(
                state[
                    name
                ]
            ) < 0:
                raise ObservationBuildError(
                    f"STATE field {name!r} "
                    "must be non-negative"
                )

        binary_fields = (
            "converged",
            "regular_phase_closed",
            "terminal",
            "selection_success",
            "finalized",
            "diagnostics_valid",
        )

        for name in binary_fields:
            if int(
                state[
                    name
                ]
            ) not in (
                0,
                1,
            ):
                raise ObservationBuildError(
                    f"STATE field {name!r} "
                    "must be 0 or 1"
                )

        if not int(
            state[
                "diagnostics_valid"
            ]
        ):
            if (
                int(
                    state[
                        "nonmanifold_polys"
                    ]
                )
                != 0
                or
                int(
                    state[
                        "high_genus_polys"
                    ]
                )
                != 0
                or
                int(
                    state[
                        "buggy_chains"
                    ]
                )
                != 0
            ):
                raise ObservationBuildError(
                    "diagnostics_valid=0 but diagnostic "
                    "counts are non-zero"
                )

    # ------------------------------------------------------------------

    def _validate_id_list(
        self,
        values: Iterable[int],
        *,
        name: str,
    ) -> list[int]:
        result = [
            int(value)
            for value in values
        ]

        if len(
            result
        ) != len(
            set(
                result
            )
        ):
            raise ObservationBuildError(
                f"{name} contains duplicate loop IDs"
            )

        for loop_id in result:
            if not (
                0
                <=
                loop_id
                <
                self.num_loops
            ):
                raise ObservationBuildError(
                    f"{name} contains invalid loop ID "
                    f"{loop_id}; model has "
                    f"{self.num_loops} loops"
                )

        return result

    # ------------------------------------------------------------------

    def build(
        self,
        *,
        state: dict,
        actions: Sequence[int],
        used: Sequence[int],
        reverted: Sequence[int],
        nico_bug: Sequence[int],
        top_relevant: Sequence[int],
        executed: Iterable[int],
    ) -> dict:
        """
        Return:

            {
                "obs": {
                    "global": float32[16],
                    "loops":  float32[MAX_LOOPS, 14],
                    "exists": bool[MAX_LOOPS],
                },
                "mask": bool[MAX_LOOPS],
            }
        """

        self._validate_state(
            state
        )

        actions = self._validate_id_list(
            actions,
            name="actions",
        )

        used = self._validate_id_list(
            used,
            name="used",
        )

        reverted = self._validate_id_list(
            reverted,
            name="reverted",
        )

        nico_bug = self._validate_id_list(
            nico_bug,
            name="nico_bug",
        )

        top_relevant = self._validate_id_list(
            top_relevant,
            name="top_relevant",
        )

        executed = self._validate_id_list(
            executed,
            name="executed",
        )

        if (
            len(
                actions
            )
            !=
            int(
                state[
                    "available"
                ]
            )
        ):
            raise ObservationBuildError(
                "STATE available count does not match "
                "authoritative C++ ACTIONS"
            )

        if (
            int(
                state[
                    "terminal"
                ]
            )
            and actions
        ):
            raise ObservationBuildError(
                "C++ STATE says terminal=1 but ACTIONS "
                "is not empty"
            )

        if (
            not int(
                state[
                    "terminal"
                ]
            )
            and not actions
        ):
            raise ObservationBuildError(
                "C++ STATE says terminal=0 but ACTIONS "
                "is empty"
            )

        # ------------------------------------------------------------
        # Convert dynamic IDs to sets for O(1) membership.
        # ------------------------------------------------------------

        action_set = set(
            actions
        )

        used_set = set(
            used
        )

        reverted_set = set(
            reverted
        )

        nico_bug_set = set(
            nico_bug
        )

        top_relevant_set = set(
            top_relevant
        )

        executed_set = set(
            executed
        )

        # ------------------------------------------------------------
        # Fixed masks.
        # ------------------------------------------------------------

        exists = np.zeros(
            self.max_loops,
            dtype=np.bool_,
        )

        exists[
            : self.num_loops
        ] = True

        action_mask = np.zeros(
            self.max_loops,
            dtype=np.bool_,
        )

        if actions:
            action_mask[
                np.asarray(
                    actions,
                    dtype=np.int64,
                )
            ] = True

        # ------------------------------------------------------------
        # Per-loop feature matrix.
        #
        #  0 serialized_position
        #  1 is_concave
        #  2 is_regular
        #  3 is_convex
        #  4 closed
        #  5 flawed
        #  6 log1p(num_segments)
        #  7 sharp_fraction
        #  8 legal
        #  9 used
        # 10 reverted
        # 11 executed
        # 12 nico_bug
        # 13 top_relevant
        # ------------------------------------------------------------

        loop_features = np.zeros(
            (
                self.max_loops,
                LOOP_FEATURE_DIM,
            ),
            dtype=np.float32,
        )

        legal_concave_count = 0
        legal_regular_count = 0

        for meta in self.metadata:
            loop_id = (
                meta.loop_id
            )

            is_concave = (
                meta.loop_type
                ==
                "CONCAVE"
            )

            is_regular = (
                meta.loop_type
                ==
                "REGULAR"
            )

            is_convex = (
                meta.loop_type
                ==
                "CONVEX"
            )

            if loop_id in action_set:
                #
                # Under frozen V1 semantics the C++ server should
                # never expose a serialized CONVEX loop as legal.
                #
                if is_convex:
                    raise ObservationBuildError(
                        "C++ ACTIONS contains a loop serialized "
                        f"as CONVEX: loop_id={loop_id}"
                    )

                if is_concave:
                    legal_concave_count += 1

                elif is_regular:
                    legal_regular_count += 1

            row = loop_features[
                loop_id
            ]

            row[0] = np.float32(
                meta.serialized_position
            )

            row[1] = np.float32(
                is_concave
            )

            row[2] = np.float32(
                is_regular
            )

            row[3] = np.float32(
                is_convex
            )

            row[4] = np.float32(
                meta.closed
            )

            row[5] = np.float32(
                meta.flawed
            )

            row[6] = np.float32(
                np.log1p(
                    meta.num_segments
                )
            )

            row[7] = np.float32(
                meta.sharp_fraction
            )

            row[8] = np.float32(
                loop_id
                in action_set
            )

            row[9] = np.float32(
                loop_id
                in used_set
            )

            row[10] = np.float32(
                loop_id
                in reverted_set
            )

            row[11] = np.float32(
                loop_id
                in executed_set
            )

            row[12] = np.float32(
                loop_id
                in nico_bug_set
            )

            row[13] = np.float32(
                loop_id
                in top_relevant_set
            )

        if (
            legal_concave_count
            +
            legal_regular_count
            !=
            len(
                actions
            )
        ):
            raise ObservationBuildError(
                "Legal action type accounting does not match "
                "the authoritative ACTIONS list"
            )

        # ------------------------------------------------------------
        # Global features.
        #
        # Fractions use initial_actionable_count as denominator.
        # Therefore:
        #
        # legal_concave_fraction + legal_regular_fraction
        #     == available_fraction
        #
        #  0 step_fraction
        #  1 available_fraction
        #  2 legal_concave_fraction
        #  3 legal_regular_fraction
        #  4 converged
        #  5 regular_phase_closed
        #  6 log(current_verts / initial_verts)
        #  7 log(current_tets / initial_tets)
        #  8 log1p(mm_verts)
        #  9 log1p(mm_edges)
        # 10 log1p(mm_faces)
        # 11 log1p(mm_polys)
        # 12 diagnostics_valid
        # 13 log1p(nonmanifold_polys)
        # 14 log1p(high_genus_polys)
        # 15 log1p(buggy_chains)
        # ------------------------------------------------------------

        denominator = float(
            self.initial_actionable_count
        )

        current_verts = int(
            state[
                "verts"
            ]
        )

        current_tets = int(
            state[
                "tets"
            ]
        )

        if current_verts <= 0:
            raise ObservationBuildError(
                "Current vertex count must be positive"
            )

        if current_tets <= 0:
            raise ObservationBuildError(
                "Current tet count must be positive"
            )

        global_features = np.asarray(
            [
                int(
                    state[
                        "step"
                    ]
                )
                /
                denominator,

                len(
                    actions
                )
                /
                denominator,

                legal_concave_count
                /
                denominator,

                legal_regular_count
                /
                denominator,

                int(
                    state[
                        "converged"
                    ]
                ),

                int(
                    state[
                        "regular_phase_closed"
                    ]
                ),

                np.log(
                    current_verts
                    /
                    self.initial_verts
                ),

                np.log(
                    current_tets
                    /
                    self.initial_tets
                ),

                np.log1p(
                    int(
                        state[
                            "mm_verts"
                        ]
                    )
                ),

                np.log1p(
                    int(
                        state[
                            "mm_edges"
                        ]
                    )
                ),

                np.log1p(
                    int(
                        state[
                            "mm_faces"
                        ]
                    )
                ),

                np.log1p(
                    int(
                        state[
                            "mm_polys"
                        ]
                    )
                ),

                int(
                    state[
                        "diagnostics_valid"
                    ]
                ),

                np.log1p(
                    int(
                        state[
                            "nonmanifold_polys"
                        ]
                    )
                ),

                np.log1p(
                    int(
                        state[
                            "high_genus_polys"
                        ]
                    )
                ),

                np.log1p(
                    int(
                        state[
                            "buggy_chains"
                        ]
                    )
                ),
            ],
            dtype=np.float32,
        )

        if global_features.shape != (
            GLOBAL_DIM,
        ):
            raise ObservationBuildError(
                "Internal error: incorrect global feature shape"
            )

        if not np.isfinite(
            global_features
        ).all():
            raise ObservationBuildError(
                "Global observation contains non-finite values"
            )

        if not np.isfinite(
            loop_features
        ).all():
            raise ObservationBuildError(
                "Per-loop observation contains non-finite values"
            )

        return {
            "obs": {
                "global":
                    global_features,

                "loops":
                    loop_features,

                "exists":
                    exists,
            },

            "mask":
                action_mask,
        }
