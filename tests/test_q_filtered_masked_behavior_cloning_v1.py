import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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


from imitation.masked_bc_v1 import (
    Q_FILTERED_BC_VERSION,
    q_filtered_masked_behavior_cloning_loss,
)


ACTION_DIM = 4
BATCH_SIZE = 3


class TableActor(
    torch.nn.Module
):
    """
    Three deterministic rows.

    row 0:
        legal {0,1}
        actor action = 0
        expert action = 1

    row 1:
        legal {0,2}
        actor action = 0
        expert action = 2

    row 2:
        legal {0,1}
        actor action = expert action = 0
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.logits = (
            torch.nn.Parameter(
                torch.tensor(
                    [
                        [
                            3.0,
                            2.0,
                            0.0,
                            -1.0,
                        ],
                        [
                            3.0,
                            1.0,
                            0.0,
                            -1.0,
                        ],
                        [
                            3.0,
                            2.0,
                            1.0,
                            0.0,
                        ],
                    ],
                    dtype=torch.float32,
                )
            )
        )

    def forward(
        self,
        obs,
        state=None,
        info=None,
    ):
        row_ids = torch.as_tensor(
            obs,
            dtype=torch.long,
            device=self.logits.device,
        ).reshape(
            -1
        )

        return (
            self.logits[
                row_ids
            ],
            None,
        )


class TableCritic(
    torch.nn.Module
):
    def __init__(
        self,
        table,
    ):
        super().__init__()

        # Parameter rather than buffer so the test can prove
        # Q-filter evaluation creates NO critic gradients.
        self.table = (
            torch.nn.Parameter(
                torch.tensor(
                    table,
                    dtype=torch.float32,
                )
            )
        )

    def forward(
        self,
        observation,
        state=None,
        info=None,
    ):
        if (
            hasattr(
                observation,
                "obs",
            )
            and
            hasattr(
                observation,
                "mask",
            )
        ):
            observation = (
                observation.obs
            )

        row_ids = torch.as_tensor(
            observation,
            dtype=torch.long,
            device=self.table.device,
        ).reshape(
            -1
        )

        return self.table[
            row_ids
        ]


def main():
    actor = (
        TableActor()
    )

    # ----------------------------------------------------------
    # Conservative min-Q values:
    #
    # row 0:
    #   actor a0  -> min Q = 0
    #   expert a1 -> min Q = 2
    #   margin = +2 -> PASS
    #
    # row 1:
    #   actor a0  -> min Q = 2
    #   expert a2 -> min Q = 1
    #   margin = -1 -> FAIL
    #
    # row 2:
    #   actor == expert == a0
    #   margin = 0 -> FAIL because predicate is strictly > 0.
    # ----------------------------------------------------------

    critic1 = TableCritic(
        [
            [
                0.0,
                3.0,
                0.0,
                0.0,
            ],
            [
                3.0,
                0.0,
                1.0,
                0.0,
            ],
            [
                2.0,
                0.0,
                0.0,
                0.0,
            ],
        ]
    )

    critic2 = TableCritic(
        [
            [
                1.0,
                2.0,
                0.0,
                0.0,
            ],
            [
                2.0,
                0.0,
                2.0,
                0.0,
            ],
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
        ]
    )

    mask = np.asarray(
        [
            [
                True,
                True,
                False,
                False,
            ],
            [
                True,
                False,
                True,
                False,
            ],
            [
                True,
                True,
                False,
                False,
            ],
        ],
        dtype=np.bool_,
    )

    expert_actions = np.asarray(
        [
            1,
            2,
            0,
        ],
        dtype=np.int64,
    )

    batch = Batch(
        obs=Batch(
            obs=np.asarray(
                [
                    0,
                    1,
                    2,
                ],
                dtype=np.int64,
            ),
            mask=mask,
        ),
        act=expert_actions,
    )

    output = (
        q_filtered_masked_behavior_cloning_loss(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            batch=batch,
        )
    )

    print(
        "version:",
        Q_FILTERED_BC_VERSION,
    )

    print(
        "selected:",
        output.selected_count,
    )

    print(
        "fraction:",
        output.filter_fraction,
    )

    print(
        "mean Q margin:",
        output.mean_q_margin,
    )

    print(
        "unfiltered loss:",
        output.unfiltered_loss,
    )

    print(
        "filtered loss:",
        float(
            output.loss.item()
        ),
    )

    assert (
        output.selected_count
        ==
        1
    )

    assert np.isclose(
        output.filter_fraction,
        1.0 / 3.0,
    )

    assert np.isclose(
        output.mean_q_margin,
        (
            2.0
            -
            1.0
            +
            0.0
        )
        /
        3.0,
    )

    # ----------------------------------------------------------
    # Verify normalization is over the COMPLETE minibatch:
    #
    #     loss =
    #       CE(row 0) / 3
    #
    # not CE(row 0) / selected_count.
    # ----------------------------------------------------------

    with torch.no_grad():
        masked_logits = (
            actor.logits.masked_fill(
                ~torch.as_tensor(
                    mask
                ),
                float("-inf"),
            )
        )

        per_sample = (
            F.cross_entropy(
                masked_logits,
                torch.as_tensor(
                    expert_actions
                ),
                reduction="none",
            )
        )

        expected = (
            per_sample[
                0
            ]
            /
            BATCH_SIZE
        )

    assert torch.allclose(
        output.loss.detach(),
        expected,
        atol=1e-7,
        rtol=1e-7,
    )

    # ----------------------------------------------------------
    # Gradient gate:
    # only selected row 0 contributes Actor BC gradient.
    # ----------------------------------------------------------

    output.loss.backward()

    assert (
        actor.logits.grad
        is not None
    )

    assert bool(
        torch.isfinite(
            actor.logits.grad
        ).all()
    )

    assert bool(
        (
            actor.logits.grad[
                0
            ]
            !=
            0.0
        ).any()
    )

    assert torch.equal(
        actor.logits.grad[
            1
        ],
        torch.zeros_like(
            actor.logits.grad[
                1
            ]
        ),
    )

    assert torch.equal(
        actor.logits.grad[
            2
        ],
        torch.zeros_like(
            actor.logits.grad[
                2
            ]
        ),
    )

    # ----------------------------------------------------------
    # Q-filter must never backpropagate into critics.
    # ----------------------------------------------------------

    assert (
        critic1.table.grad
        is None
    )

    assert (
        critic2.table.grad
        is None
    )

    print()
    print(
        "PASS: deterministic Q-filter "
        "selects exactly expert-better rows, "
        "uses full-batch normalization, and "
        "blocks critic gradients"
    )


if __name__ == "__main__":
    main()
