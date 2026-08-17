import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import (
    Batch,
)


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


from networks.loopycuts_actor_critic_v1 import (
    NETWORK_VERSION,
    build_loopycuts_actor_critics_v1,
    count_trainable_parameters,
)

from observation.builder import (
    GLOBAL_DIM,
    LOOP_FEATURE_DIM,
    MAX_LOOPS,
)

from policies.masked_discrete_sac import (
    MaskedDiscreteSACPolicy,
)


def parameter_addresses(
    module,
):
    return {
        parameter.data_ptr()
        for parameter
        in module.parameters()
    }


def assert_finite_gradients(
    module,
):
    gradients = [
        parameter.grad
        for parameter
        in module.parameters()
        if parameter.requires_grad
    ]

    assert gradients

    assert any(
        gradient is not None
        for gradient
        in gradients
    )

    for gradient in gradients:
        if gradient is None:
            continue

        assert bool(
            torch.isfinite(
                gradient
            ).all()
        )


def main():
    torch.manual_seed(
        7
    )

    np.random.seed(
        7
    )

    (
        actor,
        critic1,
        critic2,
    ) = build_loopycuts_actor_critics_v1(
        device="cpu"
    )

    print(
        "network version:",
        NETWORK_VERSION,
    )

    print(
        "actor parameters:",
        count_trainable_parameters(
            actor
        ),
    )

    print(
        "critic1 parameters:",
        count_trainable_parameters(
            critic1
        ),
    )

    print(
        "critic2 parameters:",
        count_trainable_parameters(
            critic2
        ),
    )

    # ----------------------------------------------------------
    # No Actor/Critic parameter sharing.
    # ----------------------------------------------------------

    actor_addresses = (
        parameter_addresses(
            actor
        )
    )

    critic1_addresses = (
        parameter_addresses(
            critic1
        )
    )

    critic2_addresses = (
        parameter_addresses(
            critic2
        )
    )

    assert actor_addresses.isdisjoint(
        critic1_addresses
    )

    assert actor_addresses.isdisjoint(
        critic2_addresses
    )

    assert critic1_addresses.isdisjoint(
        critic2_addresses
    )

    # ----------------------------------------------------------
    # Two batched Observation V1 states.
    #
    # Row 0:
    #     4 serialized loops exist
    #     legal actions = {0, 2}
    #
    # Row 1:
    #     3 serialized loops exist
    #     terminal legal mask = all False
    # ----------------------------------------------------------

    global_features = np.random.randn(
        2,
        GLOBAL_DIM,
    ).astype(
        np.float32
    )

    loop_features = np.random.randn(
        2,
        MAX_LOOPS,
        LOOP_FEATURE_DIM,
    ).astype(
        np.float32
    )

    exists = np.zeros(
        (
            2,
            MAX_LOOPS,
        ),
        dtype=np.bool_,
    )

    exists[
        0,
        :4,
    ] = True

    exists[
        1,
        :3,
    ] = True

    mask = np.zeros(
        (
            2,
            MAX_LOOPS,
        ),
        dtype=np.bool_,
    )

    mask[
        0,
        [
            0,
            2,
        ],
    ] = True

    inner_state = Batch(
        {
            "global":
                global_features,

            "loops":
                loop_features,

            "exists":
                exists,
        }
    )

    outer_observation = Batch(
        obs=
            inner_state,

        mask=
            mask,
    )

    # ----------------------------------------------------------
    # Actor interface.
    # ----------------------------------------------------------

    actor_logits, hidden = actor(
        inner_state
    )

    assert hidden is None

    assert (
        tuple(
            actor_logits.shape
        )
        ==
        (
            2,
            MAX_LOOPS,
        )
    )

    assert bool(
        torch.isfinite(
            actor_logits
        ).all()
    )

    # ----------------------------------------------------------
    # Critic supports exact Tianshou outer observation path.
    # ----------------------------------------------------------

    q1_outer = critic1(
        outer_observation
    )

    q1_inner = critic1(
        inner_state
    )

    q2_outer = critic2(
        outer_observation
    )

    assert (
        tuple(
            q1_outer.shape
        )
        ==
        (
            2,
            MAX_LOOPS,
        )
    )

    assert (
        tuple(
            q2_outer.shape
        )
        ==
        (
            2,
            MAX_LOOPS,
        )
    )

    assert torch.allclose(
        q1_outer,
        q1_inner,
    )

    assert bool(
        torch.isfinite(
            q1_outer
        ).all()
    )

    assert bool(
        torch.isfinite(
            q2_outer
        ).all()
    )

    # ----------------------------------------------------------
    # Explicit policy masking.
    # ----------------------------------------------------------

    policy = MaskedDiscreteSACPolicy(
        actor=
            actor,

        action_space=
            gym.spaces.Discrete(
                MAX_LOOPS
            ),

        deterministic_eval=
            False,
    )

    policy_batch = Batch(
        obs=
            outer_observation,

        info=[
            {},
            {},
        ],
    )

    output = policy(
        policy_batch
    )

    probs = (
        output
        .dist
        .probs
        .detach()
    )

    assert (
        tuple(
            probs.shape
        )
        ==
        (
            2,
            MAX_LOOPS,
        )
    )

    # Row 0:
    # only 0 and 2 can have non-zero probability.
    illegal = np.ones(
        MAX_LOOPS,
        dtype=np.bool_,
    )

    illegal[
        [
            0,
            2,
        ]
    ] = False

    assert bool(
        (
            probs[
                0,
                torch.as_tensor(
                    illegal
                ),
            ]
            ==
            0.0
        ).all()
    )

    assert torch.allclose(
        probs[
            0
        ].sum(),
        torch.tensor(
            1.0
        ),
    )

    # Row 1:
    # terminal all-False mask -> internal action-0 fallback.
    assert bool(
        output.mask_fallback[
            1
        ]
    )

    assert float(
        probs[
            1,
            0
        ]
    ) == 1.0

    assert float(
        probs[
            1,
            1:
        ].sum()
    ) == 0.0

    # ----------------------------------------------------------
    # Padding invariance.
    #
    # Changing features of NON-EXISTING padded loops must not
    # change scores of existing actions.
    # ----------------------------------------------------------

    one_global = np.random.randn(
        1,
        GLOBAL_DIM,
    ).astype(
        np.float32
    )

    one_loops = np.random.randn(
        1,
        MAX_LOOPS,
        LOOP_FEATURE_DIM,
    ).astype(
        np.float32
    )

    one_exists = np.zeros(
        (
            1,
            MAX_LOOPS,
        ),
        dtype=np.bool_,
    )

    one_exists[
        0,
        :4,
    ] = True

    modified_loops = (
        one_loops.copy()
    )

    modified_loops[
        0,
        4:,
        :,
    ] = (
        np.random.randn(
            MAX_LOOPS
            -
            4,
            LOOP_FEATURE_DIM,
        ).astype(
            np.float32
        )
        *
        1000.0
    )

    state_a = Batch(
        {
            "global":
                one_global,

            "loops":
                one_loops,

            "exists":
                one_exists,
        }
    )

    state_b = Batch(
        {
            "global":
                one_global.copy(),

            "loops":
                modified_loops,

            "exists":
                one_exists.copy(),
        }
    )

    with torch.no_grad():
        logits_a, _ = actor(
            state_a
        )

        logits_b, _ = actor(
            state_b
        )

        q_a = critic1(
            state_a
        )

        q_b = critic1(
            state_b
        )

    assert torch.allclose(
        logits_a[
            :,
            :4,
        ],
        logits_b[
            :,
            :4,
        ],
        atol=1e-6,
        rtol=1e-6,
    )

    assert torch.allclose(
        q_a[
            :,
            :4,
        ],
        q_b[
            :,
            :4,
        ],
        atol=1e-6,
        rtol=1e-6,
    )

    # ----------------------------------------------------------
    # Gradient smoke.
    # ----------------------------------------------------------

    actor.zero_grad(
        set_to_none=True
    )

    critic1.zero_grad(
        set_to_none=True
    )

    critic2.zero_grad(
        set_to_none=True
    )

    train_logits, _ = actor(
        inner_state
    )

    train_q1 = critic1(
        outer_observation
    )

    train_q2 = critic2(
        outer_observation
    )

    loss = (
        train_logits[
            :,
            0
        ].mean()
        +
        train_q1[
            :,
            0
        ].mean()
        +
        train_q2[
            :,
            0
        ].mean()
    )

    loss.backward()

    assert_finite_gradients(
        actor
    )

    assert_finite_gradients(
        critic1
    )

    assert_finite_gradients(
        critic2
    )

    print()
    print(
        "PASS: LoopyCuts Actor/Critic V1 "
        "shape, masking, padding, independence, "
        "and gradient regression"
    )


if __name__ == "__main__":
    main()
