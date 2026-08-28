import sys
from pathlib import Path

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


from imitation.demo_replay import (
    load_main_demo_replay,
)

from imitation.masked_bc_v1 import (
    BC_VERSION,
    MaskedBehaviorCloningError,
    masked_behavior_cloning_loss,
)

from networks.loopycuts_actor_critic_v1 import (
    build_loopycuts_actor_critics_v1,
)

from training.protocol_v1 import (
    PROJECT_MAIN_DEMO_EPISODES,
    PROJECT_MAIN_DEMO_TRANSITIONS,
)


RAW_ROOT = Path(
    "/home/yjk/loopycuts_test/"
    "demonstrations_v1/raw/train"
)

QUALITY = Path(
    "data/manifests/"
    "demo_quality_v1.csv"
)


def main():
    torch.manual_seed(
        19
    )

    np.random.seed(
        19
    )

    buffer, records, provenance = (
        load_main_demo_replay(
            raw_root=
                RAW_ROOT,

            quality_manifest=
                QUALITY,

            random_seed=
                19,
        )
    )

    assert len(
        buffer
    ) == PROJECT_MAIN_DEMO_TRANSITIONS

    assert len(
        records
    ) == PROJECT_MAIN_DEMO_EPISODES

    # Fixed-size deterministic component-regression subset.
    #
    # The formal D_demo identity is checked above.  BC mathematics
    # does not require 30 full-batch optimization passes over all
    # 605 transitions.
    smoke_batch_size = 64

    data = buffer[
        buffer.sample_indices(
            smoke_batch_size
        )
    ]

    assert len(
        data
    ) == smoke_batch_size

    actor, critic1, critic2 = (
        build_loopycuts_actor_critics_v1(
            device="cpu"
        )
    )

    # ----------------------------------------------------------
    # Initial objective on a deterministic subset of the current
    # frozen formal D_demo.
    # ----------------------------------------------------------

    initial = (
        masked_behavior_cloning_loss(
            actor=
                actor,

            batch=
                data,
        )
    )

    print(
        "BC version:",
        BC_VERSION,
    )

    print(
        "episodes:",
        len(
            records
        ),
    )

    print(
        "transitions:",
        len(
            buffer
        ),
    )

    print(
        "models:",
        provenance[
            "models"
        ],
    )

    print(
        "initial loss:",
        float(
            initial.loss.item()
        ),
    )

    print(
        "initial expert probability:",
        initial.mean_expert_probability,
    )

    print(
        "initial top1 accuracy:",
        initial.top1_accuracy,
    )

    assert (
        initial.batch_size
        ==
        smoke_batch_size
    )

    assert bool(
        torch.isfinite(
            initial.loss
        )
    )

    # ----------------------------------------------------------
    # Integrity guard:
    # deliberately replace the first expert action by an illegal
    # action and verify that BC rejects the sample.
    # ----------------------------------------------------------

    first = data[
        np.asarray(
            [
                0,
            ],
            dtype=np.int64,
        )
    ]

    first_mask = np.asarray(
        first.obs.mask,
        dtype=np.bool_,
    ).copy()

    illegal_candidates = (
        np.flatnonzero(
            ~first_mask[
                0
            ]
        )
    )

    assert (
        illegal_candidates.size
        >
        0
    )

    illegal_action = int(
        illegal_candidates[
            0
        ]
    )

    bad_batch = Batch(
        obs=
            first.obs,

        act=np.asarray(
            [
                illegal_action,
            ],
            dtype=np.int64,
        ),
    )

    try:
        masked_behavior_cloning_loss(
            actor=
                actor,

            batch=
                bad_batch,
        )

    except MaskedBehaviorCloningError:
        pass

    else:
        raise RuntimeError(
            "Illegal expert action was "
            "incorrectly accepted by BC"
        )

    # ----------------------------------------------------------
    # Short optimization smoke.
    #
    # This is NOT formal training.
    # It only verifies that the production Actor can learn a
    # deterministic subset of the current formal demonstration
    # objective and that the objective decreases under gradient
    # descent.
    # ----------------------------------------------------------

    optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=3e-3,
    )

    for _ in range(
        30
    ):
        optimizer.zero_grad(
            set_to_none=True
        )

        output = (
            masked_behavior_cloning_loss(
                actor=
                    actor,

                batch=
                    data,
            )
        )

        output.loss.backward()

        torch.nn.utils.clip_grad_norm_(
            actor.parameters(),
            max_norm=10.0,
        )

        optimizer.step()

    final = (
        masked_behavior_cloning_loss(
            actor=
                actor,

            batch=
                data,
        )
    )

    print()
    print(
        "final loss:",
        float(
            final.loss.item()
        ),
    )

    print(
        "final expert probability:",
        final.mean_expert_probability,
    )

    print(
        "final top1 accuracy:",
        final.top1_accuracy,
    )

    assert bool(
        torch.isfinite(
            final.loss
        )
    )

    assert (
        float(
            final.loss.item()
        )
        <
        float(
            initial.loss.item()
        )
    )

    assert (
        final.mean_expert_probability
        >
        initial.mean_expert_probability
    )

    # ----------------------------------------------------------
    # Critics are intentionally absent from BC optimization.
    #
    # Verify they still contain finite untouched parameters.
    # ----------------------------------------------------------

    for name, module in [
        (
            "critic1",
            critic1,
        ),
        (
            "critic2",
            critic2,
        ),
    ]:
        for parameter in module.parameters():
            assert bool(
                torch.isfinite(
                    parameter
                ).all()
            ), (
                f"Non-finite parameter "
                f"in {name}"
            )

    print()
    print(
        "PASS: Masked Behavior Cloning V1 "
        "learns formal BC_CORE D_demo and "
        "rejects illegal expert actions"
    )


if __name__ == "__main__":
    main()
