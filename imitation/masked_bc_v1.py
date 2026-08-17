from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


BC_VERSION = (
    "masked_behavior_cloning_v1"
)


class MaskedBehaviorCloningError(
    ValueError
):
    pass


@dataclass
class MaskedBehaviorCloningOutput:
    loss: torch.Tensor
    mean_expert_probability: float
    top1_accuracy: float
    batch_size: int


def masked_behavior_cloning_loss(
    *,
    actor,
    batch,
):
    """
    Masked discrete behavior-cloning objective for LoopyCuts.

    Expected Tianshou transition Batch:

        batch.obs.obs
            Observation V1 neural-network state:
                global
                loops
                exists

        batch.obs.mask
            current-state legal-action mask [B, 331]

        batch.act
            expert action IDs [B]

    The objective is:

        -log pi(a_expert | s, legal_actions)

    Important:
        This function operates on CURRENT transition states only.

        Unlike MaskedDiscreteSACPolicy target evaluation, an all-False
        action mask is invalid here.  A BC training sample must have a
        genuine legal expert action and therefore must be actionable.
    """

    if not hasattr(
        batch,
        "obs",
    ):
        raise MaskedBehaviorCloningError(
            "BC batch is missing batch.obs"
        )

    if not hasattr(
        batch,
        "act",
    ):
        raise MaskedBehaviorCloningError(
            "BC batch is missing batch.act"
        )

    outer_observation = (
        batch.obs
    )

    if not hasattr(
        outer_observation,
        "obs",
    ):
        raise MaskedBehaviorCloningError(
            "BC expects batch.obs.obs"
        )

    if not hasattr(
        outer_observation,
        "mask",
    ):
        raise MaskedBehaviorCloningError(
            "BC expects batch.obs.mask"
        )

    state_observation = (
        outer_observation.obs
    )

    raw_logits, hidden_state = actor(
        state_observation
    )

    if hidden_state is not None:
        raise MaskedBehaviorCloningError(
            "Masked Behavior Cloning V1 "
            "expects a feed-forward Actor"
        )

    if not isinstance(
        raw_logits,
        torch.Tensor,
    ):
        raise MaskedBehaviorCloningError(
            "Actor logits must be a torch.Tensor"
        )

    if raw_logits.ndim != 2:
        raise MaskedBehaviorCloningError(
            "Actor logits must have shape "
            "[batch_size, action_dim]; "
            f"got {tuple(raw_logits.shape)}"
        )

    device = (
        raw_logits.device
    )

    action_mask = torch.as_tensor(
        outer_observation.mask,
        dtype=torch.bool,
        device=device,
    )

    if (
        action_mask.ndim == 1
        and
        raw_logits.shape[0] == 1
    ):
        action_mask = (
            action_mask.unsqueeze(
                0
            )
        )

    if (
        tuple(
            action_mask.shape
        )
        !=
        tuple(
            raw_logits.shape
        )
    ):
        raise MaskedBehaviorCloningError(
            "BC action-mask shape does not "
            "match Actor logits: "
            f"mask={tuple(action_mask.shape)}, "
            f"logits={tuple(raw_logits.shape)}"
        )

    batch_size = int(
        raw_logits.shape[
            0
        ]
    )

    action_dim = int(
        raw_logits.shape[
            1
        ]
    )

    expert_actions = torch.as_tensor(
        batch.act,
        dtype=torch.long,
        device=device,
    ).reshape(
        -1
    )

    if (
        expert_actions.shape[
            0
        ]
        !=
        batch_size
    ):
        raise MaskedBehaviorCloningError(
            "Expert-action batch dimension "
            "does not match Actor logits"
        )

    if bool(
        (
            ~action_mask.any(
                dim=1
            )
        ).any()
    ):
        bad_row = int(
            torch.nonzero(
                ~action_mask.any(
                    dim=1
                ),
                as_tuple=False,
            )[
                0
            ].item()
        )

        raise MaskedBehaviorCloningError(
            "BC current state has no legal "
            f"actions at batch row {bad_row}; "
            "terminal fallback is forbidden "
            "for behavior cloning"
        )

    outside_range = (
        (expert_actions < 0)
        |
        (expert_actions >= action_dim)
    )

    if bool(
        outside_range.any()
    ):
        bad_row = int(
            torch.nonzero(
                outside_range,
                as_tuple=False,
            )[
                0
            ].item()
        )

        raise MaskedBehaviorCloningError(
            "Expert action is outside the "
            "discrete action range at "
            f"batch row {bad_row}: "
            f"action={int(expert_actions[bad_row])}, "
            f"action_dim={action_dim}"
        )

    expert_is_legal = (
        action_mask.gather(
            1,
            expert_actions.unsqueeze(
                1
            ),
        )
        .squeeze(
            1
        )
    )

    if not bool(
        expert_is_legal.all()
    ):
        bad_row = int(
            torch.nonzero(
                ~expert_is_legal,
                as_tuple=False,
            )[
                0
            ].item()
        )

        raise MaskedBehaviorCloningError(
            "Expert action is illegal under "
            "the current-state mask at "
            f"batch row {bad_row}: "
            f"action={int(expert_actions[bad_row])}"
        )

    masked_logits = (
        raw_logits.masked_fill(
            ~action_mask,
            float("-inf"),
        )
    )

    loss = F.cross_entropy(
        masked_logits,
        expert_actions,
        reduction="mean",
    )

    if not bool(
        torch.isfinite(
            loss
        )
    ):
        raise MaskedBehaviorCloningError(
            "Behavior-cloning loss is non-finite"
        )

    with torch.no_grad():
        probabilities = (
            torch.softmax(
                masked_logits,
                dim=-1,
            )
        )

        expert_probabilities = (
            probabilities.gather(
                1,
                expert_actions.unsqueeze(
                    1
                ),
            )
            .squeeze(
                1
            )
        )

        predicted_actions = (
            masked_logits.argmax(
                dim=-1
            )
        )

        top1_accuracy = (
            predicted_actions
            .eq(
                expert_actions
            )
            .float()
            .mean()
        )

        mean_expert_probability = (
            expert_probabilities.mean()
        )

    return MaskedBehaviorCloningOutput(
        loss=
            loss,

        mean_expert_probability=
            float(
                mean_expert_probability.item()
            ),

        top1_accuracy=
            float(
                top1_accuracy.item()
            ),

        batch_size=
            batch_size,
    )
