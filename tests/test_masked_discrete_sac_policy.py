import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


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


from tianshou.data import (
    Batch,
)

from policies.masked_discrete_sac import (
    MaskedDiscreteSACPolicy,
)


ACTION_DIM = 5


class FixedActor(
    torch.nn.Module
):
    """
    Deterministic test actor.

    Raw logits:

        action 0 -> 10
        action 1 ->  9
        action 2 ->  8
        action 3 ->  7
        action 4 ->  6

    This deliberately gives the ILLEGAL actions the highest raw
    logits, so the test can prove that masking really works.
    """

    def __init__(self):
        super().__init__()

        self.register_buffer(
            "base_logits",
            torch.tensor(
                [
                    10.0,
                    9.0,
                    8.0,
                    7.0,
                    6.0,
                ],
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        obs,
        state=None,
        info=None,
    ):
        #
        # The policy must unwrap batch.obs.obs before calling us.
        #
        if hasattr(
            obs,
            "mask",
        ):
            raise AssertionError(
                "Actor incorrectly received "
                "the mask Batch"
            )

        batch_size = (
            int(obs.shape[0])
        )

        logits = (
            self.base_logits
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
            )
        )

        return (
            logits,
            None,
        )


def main():

    torch.manual_seed(0)
    np.random.seed(0)

    action_space = (
        gym.spaces.Discrete(
            ACTION_DIM
        )
    )

    actor = FixedActor()

    #
    # deterministic_eval=False:
    #
    # use actual categorical sampling in this unit test.
    #
    policy = (
        MaskedDiscreteSACPolicy(
            actor=actor,
            action_space=
                action_space,
            deterministic_eval=
                False,
        )
    )

    # ----------------------------------------------------------
    # Two states:
    #
    # row 0:
    #     real legal actions = {2, 4}
    #
    # row 1:
    #     terminal state
    #     no legal actions
    #
    # ----------------------------------------------------------

    features = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )

    mask = np.asarray(
        [
            [
                False,
                False,
                True,
                False,
                True,
            ],

            [
                False,
                False,
                False,
                False,
                False,
            ],
        ],
        dtype=bool,
    )

    batch = Batch(
        obs=Batch(
            obs=features,
            mask=mask,
        ),
        info=[
            {},
            {},
        ],
    )

    output = policy(
        batch
    )

    probs = (
        output.dist.probs.detach()
    )

    print(
        "masked logits:"
    )

    print(
        output.logits
    )

    print()

    print(
        "probabilities:"
    )

    print(
        probs
    )

    print()

    print(
        "sampled action:",
        output.act,
    )

    print(
        "mask fallback:",
        output.mask_fallback,
    )

    # ----------------------------------------------------------
    # Test 1:
    # illegal actions have exactly zero probability.
    # ----------------------------------------------------------

    assert float(
        probs[0, 0]
    ) == 0.0

    assert float(
        probs[0, 1]
    ) == 0.0

    assert float(
        probs[0, 3]
    ) == 0.0

    # ----------------------------------------------------------
    # Test 2:
    # legal probabilities are normalized only over {2, 4}.
    # ----------------------------------------------------------

    assert torch.allclose(
        probs[
            0
        ].sum(),
        torch.tensor(
            1.0
        ),
    )

    expected_legal = (
        torch.softmax(
            torch.tensor(
                [
                    8.0,
                    6.0,
                ]
            ),
            dim=0,
        )
    )

    actual_legal = (
        probs[
            0,
            [
                2,
                4,
            ],
        ]
    )

    assert torch.allclose(
        actual_legal,
        expected_legal,
    )

    # ----------------------------------------------------------
    # Test 3:
    # terminal all-False mask uses only the internal fallback.
    # ----------------------------------------------------------

    assert bool(
        output.mask_fallback[0]
    ) is False

    assert bool(
        output.mask_fallback[1]
    ) is True

    expected_terminal_probs = (
        torch.tensor(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
    )

    assert torch.equal(
        probs[1],
        expected_terminal_probs,
    )

    # ----------------------------------------------------------
    # Test 4:
    # sample many actions directly from the masked distribution.
    #
    # Normal row:
    #     only 2 or 4 may ever appear.
    #
    # Terminal fallback row:
    #     only dummy action 0 may appear.
    # ----------------------------------------------------------

    samples = (
        output.dist.sample(
            (
                5000,
            )
        )
    )

    normal_samples = (
        samples[
            :,
            0,
        ]
    )

    terminal_samples = (
        samples[
            :,
            1,
        ]
    )

    unique_normal = set(
        normal_samples.tolist()
    )

    unique_terminal = set(
        terminal_samples.tolist()
    )

    print()

    print(
        "normal sampled actions:",
        sorted(
            unique_normal
        ),
    )

    print(
        "terminal fallback actions:",
        sorted(
            unique_terminal
        ),
    )

    assert (
        unique_normal
        <=
        {
            2,
            4,
        }
    )

    assert (
        unique_terminal
        ==
        {
            0,
        }
    )

    # ----------------------------------------------------------
    # Test 5:
    # entropy is finite.
    # ----------------------------------------------------------

    entropy = (
        output.dist.entropy()
    )

    print(
        "entropy:",
        entropy,
    )

    assert bool(
        torch.isfinite(
            entropy
        ).all()
    )

    #
    # One-action fallback distribution has zero entropy.
    #
    assert torch.allclose(
        entropy[1],
        torch.tensor(
            0.0
        ),
    )

    # ----------------------------------------------------------
    # Test 6:
    # mask shape errors must fail loudly.
    # ----------------------------------------------------------

    bad_batch = Batch(
        obs=Batch(
            obs=np.asarray(
                [
                    [
                        1.0,
                        2.0,
                        3.0,
                    ],
                ],
                dtype=np.float32,
            ),

            mask=np.asarray(
                [
                    [
                        True,
                        False,
                        True,
                    ],
                ],
                dtype=bool,
            ),
        ),

        info=[
            {},
        ],
    )

    try:
        policy(
            bad_batch
        )

    except ValueError as exc:
        print()

        print(
            "expected shape error:"
        )

        print(
            exc
        )

    else:
        raise AssertionError(
            "Mask shape mismatch "
            "was not rejected"
        )

    print()

    print(
        "PASS: MaskedDiscreteSACPolicy "
        "correctly masks legal actions "
        "and safely handles terminal "
        "all-False masks."
    )


if __name__ == "__main__":
    main()