from __future__ import annotations

import struct
import sys

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from finalization.terminal_quality_v1 import (
    parse_terminal_quality_facts,
)

from rewards.reward_v3 import (
    OUTCOME_CRASH,
    OUTCOME_FULL_HEX,
    OUTCOME_NON_FULL_HEX,
    OUTCOME_RESOURCE_ABORT,
    compute_reward_v3,
)

from rewards.reward_v5 import (
    DEFAULT_REWARD_V5_WEIGHTS,
    REWARD_V5_VERSION,
    compute_reward_v5,
)

from rewards.transition_metrics import (
    TransitionMetrics,
)


def bits(value):
    return struct.unpack(
        "=Q",
        struct.pack(
            "=d",
            float(value),
        ),
    )[0]


def assert_float_exact(
    actual,
    expected,
    *,
    name,
):
    if bits(actual) != bits(expected):
        raise AssertionError(
            f"{name}: "
            f"{actual!r} != {expected!r}; "
            f"bits "
            f"{bits(actual):016x} != "
            f"{bits(expected):016x}"
        )


def metrics(
    *,
    step=7,
    reverted=0,
    convergence_delta=0,
    first_convergence=0,
    terminal=0,
):
    return TransitionMetrics(
        step=
            int(step),

        loop_id=
            3,

        status=
            (
                "REVERTED"
                if reverted
                else
                "COMMITTED"
            ),

        committed=
            int(not reverted),

        reverted=
            int(reverted),

        step_cost=
            1.0,

        log_tet_growth=
            0.125,

        log_vert_growth=
            0.050,

        step_time=
            0.25,

        convergence_delta=
            int(convergence_delta),

        first_convergence=
            int(first_convergence),

        phase_closed_this_step=
            0,

        terminal=
            int(terminal),

        selection_success=
            int(terminal),

        terminal_failure=
            0,

        diagnostics_delta_valid=
            1,

        delta_log_nonmanifold=
            0.0,

        delta_log_high_genus=
            0.0,

        delta_log_buggy_chains=
            0.0,

        post_log_nonmanifold=
            0.0,

        post_log_high_genus=
            0.0,

        post_log_buggy_chains=
            0.0,

        delta_log_mm_polys=
            0.0,

        available_before=
            10,

        available_after=
            (
                0
                if terminal
                else
                9
            ),

        available_drop=
            (
                10
                if terminal
                else
                1
            ),
    )


def check_dense_parity(
    m,
):
    old = compute_reward_v3(
        metrics=
            m,

        initial_actionable_count=
            80,

        terminal_outcome=
            None,

        resource_abort=
            False,
    )

    new = compute_reward_v5(
        metrics=
            m,

        initial_actionable_count=
            80,

        terminal_outcome=
            None,

        terminal_quality=
            None,

        resource_abort=
            False,
    )

    assert_float_exact(
        new.step,
        old.step,
        name="step",
    )

    assert_float_exact(
        new.tet_growth,
        old.tet_growth,
        name="tet_growth",
    )

    assert_float_exact(
        new.revert,
        old.revert,
        name="revert",
    )

    assert_float_exact(
        new.convergence,
        old.convergence,
        name="convergence",
    )

    assert_float_exact(
        new.total,
        old.total,
        name="ordinary total",
    )

    assert new.terminal == 0.0
    assert new.utility == 0.0
    assert new.quality_available is False


# ================================================================
# V5 defaults: lambda_step stays exactly 1.
# ================================================================

assert (
    REWARD_V5_VERSION
    ==
    "final_v5_quality_aware_v1"
)

assert_float_exact(
    DEFAULT_REWARD_V5_WEIGHTS.step,
    1.0,
    name="lambda_step",
)

assert_float_exact(
    DEFAULT_REWARD_V5_WEIGHTS.tet_growth,
    1.0,
    name="tet_growth_weight",
)

assert_float_exact(
    DEFAULT_REWARD_V5_WEIGHTS.revert,
    0.10,
    name="revert_weight",
)

assert_float_exact(
    DEFAULT_REWARD_V5_WEIGHTS.convergence_loss,
    1.0,
    name="convergence_loss_weight",
)

assert_float_exact(
    DEFAULT_REWARD_V5_WEIGHTS.convergence_recovery,
    1.0,
    name="convergence_recovery_weight",
)


# ================================================================
# Ordinary dense parity across meaningful branches.
# ================================================================

check_dense_parity(
    metrics()
)

check_dense_parity(
    metrics(
        reverted=1,
    )
)

check_dense_parity(
    metrics(
        convergence_delta=-1,
    )
)

check_dense_parity(
    metrics(
        convergence_delta=1,
        first_convergence=0,
    )
)

check_dense_parity(
    metrics(
        convergence_delta=1,
        first_convergence=1,
    )
)


# ================================================================
# Frozen terminal-quality cases.
# ================================================================

mechanical02 = parse_terminal_quality_facts(
    {
        "model": "mechanical02",
        "hex": 880,
        "total_polys": 880,
        "nonhex": 0,
        "d_c": 1.0,
        "q_missing": 0.99853225619739827,
        "q_spurious": 0.99861570853523696,
        "q_shape": 0.99853225619739827,
        "sharp_active": 1,
        "sharp_metrics_valid": 1,
        "q_sharp": 0.99019972262228984,
        "q_fidelity": 0.988746363116073,
    }
)

cactus = parse_terminal_quality_facts(
    {
        "model": "cactus",
        "hex": 534,
        "total_polys": 534,
        "nonhex": 0,
        "d_c": 1.0,
        "q_missing": 0.21984521141416016,
        "q_spurious": 0.99783098101164747,
        "q_shape": 0.21984521141416016,
        "sharp_active": 0,
        "sharp_metrics_valid": 0,
        "q_sharp": "NA",
        "q_fidelity": 0.21984521141416016,
    }
)

bone_femur = parse_terminal_quality_facts(
    {
        "model": "bone_femur",
        "hex": 2102,
        "total_polys": 2106,
        "nonhex": 4,
        "d_c": 0.88780577166826335,
        "q_missing": 0.9974300309629075,
        "q_spurious": 0.99804809503350089,
        "q_shape": 0.9974300309629075,
        "sharp_active": 0,
        "sharp_metrics_valid": 0,
        "q_sharp": "NA",
        "q_fidelity": 0.9974300309629075,
    }
)

mechanical08 = parse_terminal_quality_facts(
    {
        "model": "mechanical08",
        "hex": 11100,
        "total_polys": 11136,
        "nonhex": 36,
        "d_c": 0.78134529313860679,
        "q_missing": 1.0,
        "q_spurious": 1.0,
        "q_shape": 1.0,
        "sharp_active": 1,
        "sharp_metrics_valid": 1,
        "q_sharp": 0.99483903795655504,
        "q_fidelity": 0.99483903795655504,
    }
)


def check_successful_terminal(
    *,
    facts,
    outcome,
):
    m = metrics(
        terminal=1,
    )

    old = compute_reward_v3(
        metrics=
            m,

        initial_actionable_count=
            80,

        terminal_outcome=
            outcome,

        resource_abort=
            False,
    )

    new = compute_reward_v5(
        metrics=
            m,

        initial_actionable_count=
            80,

        terminal_outcome=
            outcome,

        terminal_quality=
            facts,

        resource_abort=
            False,
    )

    #
    # Dense terms must remain exactly V3.
    #
    for name in (
        "step",
        "tet_growth",
        "revert",
        "convergence",
    ):
        assert_float_exact(
            getattr(
                new,
                name,
            ),
            getattr(
                old,
                name,
            ),
            name=
                f"{facts.model}:{name}",
        )

    expected_utility = (
        facts.d_c
        *
        facts.q_fidelity
    )

    expected_terminal = (
        6.0
        *
        expected_utility
        -
        3.0
    )

    assert_float_exact(
        new.utility,
        expected_utility,
        name=
            f"{facts.model}:utility",
    )

    assert_float_exact(
        new.terminal,
        expected_terminal,
        name=
            f"{facts.model}:terminal",
    )

    expected_total = (
        new.step
        +
        new.tet_growth
        +
        new.revert
        +
        new.convergence
        +
        new.terminal
    )

    assert_float_exact(
        new.total,
        expected_total,
        name=
            f"{facts.model}:total",
    )

    assert new.quality_available is True

    return new


r_mech02 = check_successful_terminal(
    facts=
        mechanical02,

    outcome=
        OUTCOME_FULL_HEX,
)

r_cactus = check_successful_terminal(
    facts=
        cactus,

    outcome=
        OUTCOME_FULL_HEX,
)

r_bone = check_successful_terminal(
    facts=
        bone_femur,

    outcome=
        OUTCOME_NON_FULL_HEX,
)

r_mech08 = check_successful_terminal(
    facts=
        mechanical08,

    outcome=
        OUTCOME_NON_FULL_HEX,
)


#
# Critical semantic checks:
#
# full/non-full class no longer chooses +/-3.
#

assert (
    r_mech02.terminal
    !=
    3.0
)

assert (
    r_cactus.terminal
    <
    0.0
)

assert (
    r_bone.terminal
    >
    0.0
)

assert (
    r_mech08.terminal
    >
    0.0
)


# ================================================================
# FINALIZATION_CRASH:
# preserve V3 semantics exactly.
# ================================================================

m = metrics(
    reverted=1,
    terminal=1,
)

old_crash = compute_reward_v3(
    metrics=
        m,

    initial_actionable_count=
        80,

    terminal_outcome=
        OUTCOME_CRASH,

    resource_abort=
        False,
)

new_crash = compute_reward_v5(
    metrics=
        m,

    initial_actionable_count=
        80,

    terminal_outcome=
        OUTCOME_CRASH,

    terminal_quality=
        None,

    resource_abort=
        False,
)


for new_value, old_value, name in (
    (
        new_crash.step,
        old_crash.step,
        "crash.step",
    ),
    (
        new_crash.tet_growth,
        old_crash.tet_growth,
        "crash.tet_growth",
    ),
    (
        new_crash.revert,
        old_crash.revert,
        "crash.revert",
    ),
    (
        new_crash.convergence,
        old_crash.convergence,
        "crash.convergence",
    ),
    (
        new_crash.terminal,
        old_crash.terminal,
        "crash.terminal",
    ),
    (
        new_crash.total,
        old_crash.total,
        "crash.total",
    ),
):
    assert_float_exact(
        new_value,
        old_value,
        name=name,
    )


# ================================================================
# RESOURCE_ABORT:
# preserve V3 exact -4 override.
# ================================================================

old_resource = compute_reward_v3(
    metrics=
        None,

    initial_actionable_count=
        80,

    terminal_outcome=
        OUTCOME_RESOURCE_ABORT,

    resource_abort=
        True,
)

new_resource = compute_reward_v5(
    metrics=
        None,

    initial_actionable_count=
        80,

    terminal_outcome=
        OUTCOME_RESOURCE_ABORT,

    terminal_quality=
        None,

    resource_abort=
        True,
)


assert_float_exact(
    new_resource.total,
    old_resource.total,
    name="resource total",
)

assert_float_exact(
    new_resource.total,
    -4.0,
    name="resource exact -4",
)

assert new_resource.step == 0.0
assert new_resource.tet_growth == 0.0
assert new_resource.revert == 0.0
assert new_resource.convergence == 0.0
assert new_resource.quality_available is False


print(
    "PASS: lambda_step remains exactly 1"
)

print(
    "PASS: ordinary V5 dense reward is bit-exact Reward V3"
)

print(
    "PASS: tet-growth/revert/convergence shaping is preserved"
)

print(
    "PASS: successful FULL/NON_FULL differ only by measured terminal quality"
)

print(
    "PASS: full hex can receive negative quality terminal reward"
)

print(
    "PASS: non-full hex can receive positive quality terminal reward"
)

print(
    "PASS: FINALIZATION_CRASH preserves Reward V3 semantics"
)

print(
    "PASS: RESOURCE_ABORT remains exact -4 override"
)
