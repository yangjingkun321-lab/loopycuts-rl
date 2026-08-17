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


Q_FILTERED_BC_VERSION = (
    "q_filtered_masked_behavior_cloning_v1"
)


@dataclass
class QFilteredMaskedBehaviorCloningOutput:
    loss: torch.Tensor
    unfiltered_loss: float
    mean_expert_probability: float
    top1_accuracy: float
    batch_size: int
    selected_count: int
    filter_fraction: float
    mean_q_margin: float


def q_filtered_masked_behavior_cloning_loss(
    *,
    actor,
    critic1,
    critic2,
    batch,
):
    """
    Q-filtered masked discrete behavior cloning for LoopyCuts.

    Discrete adaptation of the Q-filtered BC term used by the
    reference SAC+BC method:

        expert action:
            a_E

        current deterministic Actor action:
            a_pi = argmax_{legal a} pi(a | s)

        conservative critic:
            Q_min = min(Q1, Q2)

        filter:
            Q_min(s, a_E) > Q_min(s, a_pi)

        discrete BC loss:
            -log pi(a_E | s, legal_actions)

    The Q-filter is treated as a non-differentiable gate.
    Gradients from this objective flow only into the Actor.
    """

    if not hasattr(
        batch,
        "obs",
    ):
        raise MaskedBehaviorCloningError(
            "Q-filtered BC batch is missing batch.obs"
        )

    if not hasattr(
        batch,
        "act",
    ):
        raise MaskedBehaviorCloningError(
            "Q-filtered BC batch is missing batch.act"
        )

    outer_observation = (
        batch.obs
    )

    if not hasattr(
        outer_observation,
        "obs",
    ):
        raise MaskedBehaviorCloningError(
            "Q-filtered BC expects batch.obs.obs"
        )

    if not hasattr(
        outer_observation,
        "mask",
    ):
        raise MaskedBehaviorCloningError(
            "Q-filtered BC expects batch.obs.mask"
        )

    state_observation = (
        outer_observation.obs
    )

    raw_logits, hidden_state = actor(
        state_observation
    )

    if hidden_state is not None:
        raise MaskedBehaviorCloningError(
            "Q-filtered BC expects a feed-forward Actor"
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
            "[batch_size, action_dim]"
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
            "Q-filtered BC mask/logit shape mismatch: "
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

    legal_per_row = (
        action_mask.any(
            dim=1
        )
    )

    if not bool(
        legal_per_row.all()
    ):
        bad_row = int(
            torch.nonzero(
                ~legal_per_row,
                as_tuple=False,
            )[
                0
            ].item()
        )

        raise MaskedBehaviorCloningError(
            "Q-filtered BC current state has "
            "no legal actions at batch row "
            f"{bad_row}"
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
            "action range at batch row "
            f"{bad_row}"
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
            f"batch row {bad_row}"
        )

    masked_logits = (
        raw_logits.masked_fill(
            ~action_mask,
            float("-inf"),
        )
    )

    per_sample_bc_loss = (
        F.cross_entropy(
            masked_logits,
            expert_actions,
            reduction="none",
        )
    )

    if not bool(
        torch.isfinite(
            per_sample_bc_loss
        ).all()
    ):
        raise MaskedBehaviorCloningError(
            "Unfiltered BC loss contains "
            "non-finite values"
        )

    with torch.no_grad():
        q1 = critic1(
            outer_observation
        )

        q2 = critic2(
            outer_observation
        )

        if (
            not isinstance(
                q1,
                torch.Tensor,
            )
            or
            not isinstance(
                q2,
                torch.Tensor,
            )
        ):
            raise MaskedBehaviorCloningError(
                "Critics must return torch.Tensor"
            )

        if (
            tuple(
                q1.shape
            )
            !=
            tuple(
                raw_logits.shape
            )
            or
            tuple(
                q2.shape
            )
            !=
            tuple(
                raw_logits.shape
            )
        ):
            raise MaskedBehaviorCloningError(
                "Critic output shape must match "
                "Actor logits"
            )

        if (
            not bool(
                torch.isfinite(
                    q1
                ).all()
            )
            or
            not bool(
                torch.isfinite(
                    q2
                ).all()
            )
        ):
            raise MaskedBehaviorCloningError(
                "Critic output contains "
                "non-finite Q values"
            )

        q_min = torch.minimum(
            q1,
            q2,
        )

        actor_actions = (
            masked_logits
            .detach()
            .argmax(
                dim=-1
            )
        )

        expert_q = (
            q_min.gather(
                1,
                expert_actions.unsqueeze(
                    1
                ),
            )
            .squeeze(
                1
            )
        )

        actor_q = (
            q_min.gather(
                1,
                actor_actions.unsqueeze(
                    1
                ),
            )
            .squeeze(
                1
            )
        )

        q_margin = (
            expert_q
            -
            actor_q
        )

        q_filter = (
            q_margin
            >
            0.0
        )

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

        top1_accuracy = (
            actor_actions
            .eq(
                expert_actions
            )
            .float()
            .mean()
        )

        mean_expert_probability = (
            expert_probabilities.mean()
        )

        selected_count = int(
            q_filter.sum().item()
        )

        filter_fraction = (
            float(
                selected_count
            )
            /
            float(
                batch_size
            )
        )

        mean_q_margin = float(
            q_margin.mean().item()
        )

    # Keep the normalization over the COMPLETE minibatch.
    #
    # This preserves the effect of the indicator gate:
    # if fewer demonstrations pass the Q-filter, the BC
    # contribution becomes correspondingly smaller.
    loss = (
        per_sample_bc_loss
        *
        q_filter.to(
            dtype=
                per_sample_bc_loss.dtype
        )
    ).mean()

    if not bool(
        torch.isfinite(
            loss
        )
    ):
        raise MaskedBehaviorCloningError(
            "Q-filtered BC loss is non-finite"
        )

    return QFilteredMaskedBehaviorCloningOutput(
        loss=
            loss,

        unfiltered_loss=
            float(
                per_sample_bc_loss
                .mean()
                .detach()
                .item()
            ),

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

        selected_count=
            selected_count,

        filter_fraction=
            filter_fraction,

        mean_q_margin=
            mean_q_margin,
    )
