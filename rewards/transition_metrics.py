from __future__ import annotations

from dataclasses import asdict, dataclass
import math


class TransitionMetricError(ValueError):
    pass


@dataclass(frozen=True)
class TransitionMetrics:
    """
    Raw, UNWEIGHTED measurements of one real LoopyCuts Stage-2 step.

    IMPORTANT:
        These are not yet a reward function.

        Phase 2D-A only records measurable transition quantities.
        Reward signs / coefficients are deliberately deferred until
        empirical scale analysis is complete.
    """

    step: int
    loop_id: int
    status: str

    committed: int
    reverted: int

    step_cost: float

    log_tet_growth: float
    log_vert_growth: float

    step_time: float

    convergence_delta: int

    first_convergence: int
    phase_closed_this_step: int

    terminal: int
    selection_success: int
    terminal_failure: int

    diagnostics_delta_valid: int

    delta_log_nonmanifold: float
    delta_log_high_genus: float
    delta_log_buggy_chains: float

    post_log_nonmanifold: float
    post_log_high_genus: float
    post_log_buggy_chains: float

    delta_log_mm_polys: float

    available_before: int
    available_after: int
    available_drop: int

    def to_dict(self) -> dict:
        return asdict(self)


_REQUIRED_STATE_FIELDS = {
    "step",
    "available",
    "verts",
    "tets",
    "mm_polys",
    "converged",
    "regular_phase_closed",
    "terminal",
    "selection_success",
    "diagnostics_valid",
    "nonmanifold_polys",
    "high_genus_polys",
    "buggy_chains",
}


_REQUIRED_STEP_FIELDS = {
    "step",
    "loop_id",
    "status",
    "committed",
    "reverted",
    "verts_before",
    "verts_after",
    "tets_before",
    "tets_after",
    "step_time",
    "converged_before",
    "regular_phase_closed_before",
    "converged",
    "regular_phase_closed",
}


def _require_fields(
    data: dict,
    required: set[str],
    *,
    name: str,
) -> None:
    missing = required - set(
        data.keys()
    )

    if missing:
        raise TransitionMetricError(
            f"{name} is missing fields: "
            + ", ".join(
                sorted(missing)
            )
        )


def _binary(
    value,
    *,
    name: str,
) -> int:
    value = int(value)

    if value not in (
        0,
        1,
    ):
        raise TransitionMetricError(
            f"{name} must be 0 or 1, got {value}"
        )

    return value


def extract_transition_metrics(
    *,
    state_before: dict,
    step_result: dict,
    state_after: dict,
) -> TransitionMetrics:
    """
    Extract one unweighted transition record.

    The function also cross-checks C++ STEP_RESULT against the
    surrounding STATE messages so corrupted/misaligned transitions
    cannot silently enter later reward analysis.
    """

    _require_fields(
        state_before,
        _REQUIRED_STATE_FIELDS,
        name="state_before",
    )

    _require_fields(
        state_after,
        _REQUIRED_STATE_FIELDS,
        name="state_after",
    )

    _require_fields(
        step_result,
        _REQUIRED_STEP_FIELDS,
        name="step_result",
    )

    # ================================================================
    # Basic C++ protocol consistency
    # ================================================================

    if (
        int(
            step_result[
                "verts_before"
            ]
        )
        !=
        int(
            state_before[
                "verts"
            ]
        )
    ):
        raise TransitionMetricError(
            "STEP_RESULT verts_before does not match STATE before"
        )

    if (
        int(
            step_result[
                "tets_before"
            ]
        )
        !=
        int(
            state_before[
                "tets"
            ]
        )
    ):
        raise TransitionMetricError(
            "STEP_RESULT tets_before does not match STATE before"
        )

    if (
        int(
            step_result[
                "verts_after"
            ]
        )
        !=
        int(
            state_after[
                "verts"
            ]
        )
    ):
        raise TransitionMetricError(
            "STEP_RESULT verts_after does not match STATE after"
        )

    if (
        int(
            step_result[
                "tets_after"
            ]
        )
        !=
        int(
            state_after[
                "tets"
            ]
        )
    ):
        raise TransitionMetricError(
            "STEP_RESULT tets_after does not match STATE after"
        )

    if (
        int(
            step_result[
                "converged_before"
            ]
        )
        !=
        int(
            state_before[
                "converged"
            ]
        )
    ):
        raise TransitionMetricError(
            "STEP_RESULT converged_before does not match STATE before"
        )

    if (
        int(
            step_result[
                "converged"
            ]
        )
        !=
        int(
            state_after[
                "converged"
            ]
        )
    ):
        raise TransitionMetricError(
            "STEP_RESULT converged does not match STATE after"
        )

    if (
        int(
            step_result[
                "regular_phase_closed_before"
            ]
        )
        !=
        int(
            state_before[
                "regular_phase_closed"
            ]
        )
    ):
        raise TransitionMetricError(
            "STEP_RESULT regular_phase_closed_before "
            "does not match STATE before"
        )

    if (
        int(
            step_result[
                "regular_phase_closed"
            ]
        )
        !=
        int(
            state_after[
                "regular_phase_closed"
            ]
        )
    ):
        raise TransitionMetricError(
            "STEP_RESULT regular_phase_closed "
            "does not match STATE after"
        )

    if (
        int(
            state_after[
                "step"
            ]
        )
        !=
        int(
            state_before[
                "step"
            ]
        )
        + 1
    ):
        raise TransitionMetricError(
            "STATE step counter did not increase by exactly one"
        )

    if (
        int(
            step_result[
                "step"
            ]
        )
        !=
        int(
            state_after[
                "step"
            ]
        )
    ):
        raise TransitionMetricError(
            "STEP_RESULT step does not match STATE after"
        )

    # ================================================================
    # Counts / ratios
    # ================================================================

    verts_before = int(
        state_before[
            "verts"
        ]
    )

    verts_after = int(
        state_after[
            "verts"
        ]
    )

    tets_before = int(
        state_before[
            "tets"
        ]
    )

    tets_after = int(
        state_after[
            "tets"
        ]
    )

    if min(
        verts_before,
        verts_after,
        tets_before,
        tets_after,
    ) <= 0:
        raise TransitionMetricError(
            "Vertex/tet counts must be positive"
        )

    log_vert_growth = math.log(
        verts_after
        /
        verts_before
    )

    log_tet_growth = math.log(
        tets_after
        /
        tets_before
    )

    # ================================================================
    # Discrete action outcome
    # ================================================================

    committed = _binary(
        step_result[
            "committed"
        ],
        name="committed",
    )

    reverted = _binary(
        step_result[
            "reverted"
        ],
        name="reverted",
    )

    #
    # With current legal-action semantics, real geometric attempts
    # normally resolve to COMMITTED or REVERTED.
    #
    # Do not force mutual exclusivity here at the metrics layer;
    # preserve what the C++ protocol actually reports.
    #

    # ================================================================
    # Convergence / phase transition
    # ================================================================

    converged_before = _binary(
        state_before[
            "converged"
        ],
        name="state_before.converged",
    )

    converged_after = _binary(
        state_after[
            "converged"
        ],
        name="state_after.converged",
    )

    convergence_delta = (
        converged_after
        -
        converged_before
    )

    phase_before = _binary(
        state_before[
            "regular_phase_closed"
        ],
        name="state_before.regular_phase_closed",
    )

    phase_after = _binary(
        state_after[
            "regular_phase_closed"
        ],
        name="state_after.regular_phase_closed",
    )

    if phase_after < phase_before:
        raise TransitionMetricError(
            "regular_phase_closed violated monotonic V1 semantics"
        )

    phase_closed_this_step = int(
        phase_before == 0
        and
        phase_after == 1
    )

    first_convergence = int(
        phase_closed_this_step
        and
        converged_after == 1
    )

    # ================================================================
    # Terminal state
    # ================================================================

    terminal = _binary(
        state_after[
            "terminal"
        ],
        name="state_after.terminal",
    )

    selection_success = _binary(
        state_after[
            "selection_success"
        ],
        name="state_after.selection_success",
    )

    if (
        selection_success
        and
        not terminal
    ):
        raise TransitionMetricError(
            "selection_success=1 requires terminal=1"
        )

    terminal_failure = int(
        terminal
        and
        not selection_success
    )

    # ================================================================
    # MeshExtractor diagnostics
    # ================================================================

    diagnostics_before = _binary(
        state_before[
            "diagnostics_valid"
        ],
        name="state_before.diagnostics_valid",
    )

    diagnostics_after = _binary(
        state_after[
            "diagnostics_valid"
        ],
        name="state_after.diagnostics_valid",
    )

    #
    # Every real STEP currently constructs MeshExtractor and therefore
    # should yield valid post-step diagnostics.
    #
    if not diagnostics_after:
        raise TransitionMetricError(
            "Post-step diagnostics are unexpectedly invalid"
        )

    before_nonmanifold = int(
        state_before[
            "nonmanifold_polys"
        ]
    )

    before_high_genus = int(
        state_before[
            "high_genus_polys"
        ]
    )

    before_buggy = int(
        state_before[
            "buggy_chains"
        ]
    )

    after_nonmanifold = int(
        state_after[
            "nonmanifold_polys"
        ]
    )

    after_high_genus = int(
        state_after[
            "high_genus_polys"
        ]
    )

    after_buggy = int(
        state_after[
            "buggy_chains"
        ]
    )

    for name, value in (
        (
            "before_nonmanifold",
            before_nonmanifold,
        ),
        (
            "before_high_genus",
            before_high_genus,
        ),
        (
            "before_buggy",
            before_buggy,
        ),
        (
            "after_nonmanifold",
            after_nonmanifold,
        ),
        (
            "after_high_genus",
            after_high_genus,
        ),
        (
            "after_buggy",
            after_buggy,
        ),
    ):
        if value < 0:
            raise TransitionMetricError(
                f"{name} must be non-negative"
            )

    diagnostics_delta_valid = int(
        diagnostics_before
        and
        diagnostics_after
    )

    if diagnostics_delta_valid:
        delta_log_nonmanifold = (
            math.log1p(
                after_nonmanifold
            )
            -
            math.log1p(
                before_nonmanifold
            )
        )

        delta_log_high_genus = (
            math.log1p(
                after_high_genus
            )
            -
            math.log1p(
                before_high_genus
            )
        )

        delta_log_buggy_chains = (
            math.log1p(
                after_buggy
            )
            -
            math.log1p(
                before_buggy
            )
        )

    else:
        #
        # Initial state has no MeshExtractor diagnostics yet.
        # Do NOT pretend its zeros form a valid delta baseline.
        #
        delta_log_nonmanifold = 0.0
        delta_log_high_genus = 0.0
        delta_log_buggy_chains = 0.0

    post_log_nonmanifold = math.log1p(
        after_nonmanifold
    )

    post_log_high_genus = math.log1p(
        after_high_genus
    )

    post_log_buggy_chains = math.log1p(
        after_buggy
    )

    # ================================================================
    # Meta-poly change.
    #
    # Audit quantity only. More meta polys are NOT assumed better.
    # ================================================================

    mm_before = int(
        state_before[
            "mm_polys"
        ]
    )

    mm_after = int(
        state_after[
            "mm_polys"
        ]
    )

    if mm_before < 0 or mm_after < 0:
        raise TransitionMetricError(
            "mm_polys must be non-negative"
        )

    delta_log_mm_polys = (
        math.log1p(
            mm_after
        )
        -
        math.log1p(
            mm_before
        )
    )

    # ================================================================
    # Action-set change.
    #
    # Audit quantity only.
    #
    # It can reflect:
    #   - the selected action itself,
    #   - find_mates() consumption,
    #   - first-convergence REGULAR phase closure.
    #
    # Therefore it is NOT interpreted directly as progress.
    # ================================================================

    available_before = int(
        state_before[
            "available"
        ]
    )

    available_after = int(
        state_after[
            "available"
        ]
    )

    available_drop = (
        available_before
        -
        available_after
    )

    return TransitionMetrics(
        step=int(
            step_result[
                "step"
            ]
        ),

        loop_id=int(
            step_result[
                "loop_id"
            ]
        ),

        status=str(
            step_result[
                "status"
            ]
        ),

        committed=committed,

        reverted=reverted,

        #
        # Constant per-action cost. A later negative weight on this
        # would prefer shorter trajectories, but NO weight is chosen
        # during Phase 2D-A.
        #
        step_cost=1.0,

        log_tet_growth=(
            log_tet_growth
        ),

        log_vert_growth=(
            log_vert_growth
        ),

        step_time=float(
            step_result[
                "step_time"
            ]
        ),

        convergence_delta=(
            convergence_delta
        ),

        first_convergence=(
            first_convergence
        ),

        phase_closed_this_step=(
            phase_closed_this_step
        ),

        terminal=terminal,

        selection_success=(
            selection_success
        ),

        terminal_failure=(
            terminal_failure
        ),

        diagnostics_delta_valid=(
            diagnostics_delta_valid
        ),

        delta_log_nonmanifold=(
            delta_log_nonmanifold
        ),

        delta_log_high_genus=(
            delta_log_high_genus
        ),

        delta_log_buggy_chains=(
            delta_log_buggy_chains
        ),

        post_log_nonmanifold=(
            post_log_nonmanifold
        ),

        post_log_high_genus=(
            post_log_high_genus
        ),

        post_log_buggy_chains=(
            post_log_buggy_chains
        ),

        delta_log_mm_polys=(
            delta_log_mm_polys
        ),

        available_before=(
            available_before
        ),

        available_after=(
            available_after
        ),

        available_drop=(
            available_drop
        ),
    )
