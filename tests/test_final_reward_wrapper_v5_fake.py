from __future__ import annotations

import tempfile
import sys

from pathlib import Path

import gymnasium as gym


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


from envs.final_reward_wrapper_v5 import (
    FinalRewardWrapperV5,
)

from envs.finalization_quality_wrapper_v1 import (
    FinalizationQualityWrapperV1,
)

from rewards.transition_metrics import (
    TransitionMetrics,
)


QUALITY = {
    "model":
        "mechanical02",

    "hex":
        880,

    "total_polys":
        880,

    "nonhex":
        0,

    "d_c":
        1.0,

    "q_missing":
        0.99853225619739827,

    "q_spurious":
        0.99861570853523696,

    "q_shape":
        0.99853225619739827,

    "sharp_active":
        1,

    "sharp_metrics_valid":
        1,

    "q_sharp":
        0.99019972262228984,

    "q_fidelity":
        0.988746363116073,
}


class FakeClient:
    def __init__(
        self,
    ):
        self.state = {
            "terminal":
                0,

            "finalized":
                0,
        }

        self.calls = []

    def finalize_quality(
        self,
        quality_ref_path,
    ):
        self.calls.append(
            Path(
                quality_ref_path
            )
        )

        final_state = {
            "terminal":
                1,

            "finalized":
                1,
        }

        self.state = dict(
            final_state
        )

        return (
            {
                "hex":
                    880,

                "total_polys":
                    880,

                "full_hex":
                    1,
            },

            dict(
                QUALITY
            ),

            final_state,
        )


class FakeTerminalEnv(
    gym.Env
):
    def __init__(
        self,
    ):
        super().__init__()

        self.client = FakeClient()

        self.legal_actions = [
            0,
            1,
            2,
            3,
        ]

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        self.client.state = {
            "terminal":
                0,

            "finalized":
                0,
        }

        return (
            0,
            {
                "state":
                    dict(
                        self.client.state
                    ),
            },
        )

    def step(
        self,
        action,
    ):
        self.client.state = {
            "terminal":
                1,

            "finalized":
                0,
        }

        metrics = TransitionMetrics(
            step=
                1,

            loop_id=
                int(
                    action
                ),

            status=
                "COMMITTED",

            committed=
                1,

            reverted=
                0,

            step_cost=
                1.0,

            log_tet_growth=
                0.125,

            log_vert_growth=
                0.05,

            step_time=
                0.25,

            convergence_delta=
                0,

            first_convergence=
                0,

            phase_closed_this_step=
                0,

            terminal=
                1,

            selection_success=
                1,

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
                4,

            available_after=
                0,

            available_drop=
                4,
        )

        return (
            1,
            123.0,
            True,
            False,
            {
                "transition_metrics":
                    metrics.to_dict(),
            },
        )


with tempfile.TemporaryDirectory() as tmp:
    ref = (
        Path(tmp)
        /
        "mechanical02.quality_ref_v1"
    )

    ref.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    base = FakeTerminalEnv()

    env = FinalRewardWrapperV5(
        FinalizationQualityWrapperV1(
            base,

            quality_ref_path=
                ref,

            expected_model=
                "mechanical02",
        )
    )

    observation, info = (
        env.reset()
    )

    assert (
        info[
            "reward_version"
        ]
        ==
        "final_v5_quality_aware_v1"
    )

    assert (
        info[
            "terminal_quality"
        ][
            "available"
        ]
        is False
    )

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(
        0
    )

    assert terminated is True
    assert truncated is False

    assert len(
        base.client.calls
    ) == 1

    assert (
        info[
            "finalization_outcome"
        ][
            "outcome"
        ]
        ==
        "FULL_HEX"
    )

    assert (
        info[
            "finalization_outcome"
        ][
            "attempted"
        ]
        is True
    )

    assert (
        info[
            "terminal_quality"
        ][
            "available"
        ]
        is True
    )

    assert (
        info[
            "terminal_quality"
        ][
            "model"
        ]
        ==
        "mechanical02"
    )

    assert (
        info[
            "terminal_quality"
        ][
            "q_fidelity"
        ]
        ==
        0.988746363116073
    )

    assert (
        info[
            "reward_v5_breakdown"
        ][
            "quality_available"
        ]
        is True
    )

    utility = (
        1.0
        *
        0.988746363116073
    )

    terminal = (
        6.0
        *
        utility
        -
        3.0
    )

    dense = (
        -1.0
        /
        4.0
        -
        0.125
    )

    expected = (
        dense
        +
        terminal
    )

    assert reward == expected

    assert (
        info[
            "reward_v5_breakdown"
        ][
            "step"
        ]
        ==
        -0.25
    )

    assert (
        info[
            "reward_v5_breakdown"
        ][
            "tet_growth"
        ]
        ==
        -0.125
    )

    assert (
        info[
            "reward_v5_breakdown"
        ][
            "revert"
        ]
        ==
        0.0
    )

    assert (
        info[
            "reward_v5_breakdown"
        ][
            "convergence"
        ]
        ==
        0.0
    )

    assert (
        info[
            "reward_v5_breakdown"
        ][
            "terminal"
        ]
        ==
        terminal
    )


# ==================================================================
# Regression: inactive SHARP across the complete wrapper composition.
#
# Raw C++-protocol q_sharp="NA"
#     -> TerminalQualityFacts.q_sharp=None
#     -> FinalizationQualityWrapperV1 info dict
#     -> FinalRewardWrapperV5
#
# This is the exact transport path that formal seed42 exposed.
# ==================================================================

INACTIVE_QUALITY = dict(
    QUALITY
)

INACTIVE_QUALITY.update(
    {
        "sharp_active":
            0,

        "sharp_metrics_valid":
            0,

        "q_sharp":
            "NA",

        "q_fidelity":
            QUALITY[
                "q_shape"
            ],
    }
)

QUALITY.clear()

QUALITY.update(
    INACTIVE_QUALITY
)


with tempfile.TemporaryDirectory() as tmp:
    ref = (
        Path(tmp)
        /
        "mechanical02.quality_ref_v1"
    )

    ref.write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    base = FakeTerminalEnv()

    env = FinalRewardWrapperV5(
        FinalizationQualityWrapperV1(
            base,

            quality_ref_path=
                ref,

            expected_model=
                "mechanical02",
        )
    )

    env.reset()

    (
        observation,
        inactive_reward,
        terminated,
        truncated,
        info,
    ) = env.step(
        0
    )

    assert terminated is True
    assert truncated is False

    assert (
        info[
            "terminal_quality"
        ][
            "available"
        ]
        is True
    )

    assert (
        info[
            "terminal_quality"
        ][
            "sharp_active"
        ]
        ==
        0
    )

    assert (
        info[
            "terminal_quality"
        ][
            "sharp_metrics_valid"
        ]
        ==
        0
    )

    assert (
        info[
            "terminal_quality"
        ][
            "q_sharp_available"
        ]
        is False
    )

    assert (
        info[
            "terminal_quality"
        ][
            "q_sharp"
        ]
        ==
        0.0
    )

    assert (
        info[
            "terminal_quality"
        ][
            "q_fidelity"
        ]
        ==
        QUALITY[
            "q_shape"
        ]
    )

    inactive_utility = (
        QUALITY[
            "d_c"
        ]
        *
        QUALITY[
            "q_fidelity"
        ]
    )

    inactive_terminal = (
        6.0
        *
        inactive_utility
        -
        3.0
    )

    inactive_dense = (
        -1.0
        /
        4.0
        -
        0.125
    )

    assert (
        info[
            "reward_v5_breakdown"
        ][
            "terminal"
        ]
        ==
        inactive_terminal
    )

    assert (
        inactive_reward
        ==
        inactive_dense
        +
        inactive_terminal
    )


print(
    "PASS: V5 wrapper executes FINALIZE_QUALITY exactly once"
)

print(
    "PASS: terminal quality survives into fixed collector schema"
)

print(
    "PASS: V5 keeps dense step/tet shaping"
)

print(
    "PASS: successful terminal uses 6*D_C*Q_fidelity-3"
)

print(
    "PASS: old inner Selection Reward V1 value does not control V5 reward"
)
