from __future__ import annotations

import numpy as np
import torch

from observation.builder import MAX_LOOPS


DETERMINISTIC_ACTOR_VERSION = (
    "loopycuts_deterministic_actor_v1"
)


def select_deterministic_actor_action(
    actor,
    observation,
):
    """
    Actor-only deterministic inference.

    Semantics:

        actor raw logits
            ->
        authoritative C++ ACTIONS mask
            ->
        illegal logits = -inf
            ->
        argmax over legal actions

    This is exactly the action-selection semantics of
    MaskedDiscreteSACPolicy deterministic evaluation for a
    non-terminal LoopyCuts state.

    No sampling.
    No epsilon exploration.
    No critic.
    No replay.
    No gradient.
    """

    if not isinstance(
        observation,
        dict,
    ):
        raise TypeError(
            "observation must be a dict"
        )

    if (
        "obs" not in observation
        or
        "mask" not in observation
    ):
        raise ValueError(
            "observation must contain "
            "'obs' and 'mask'"
        )

    mask = np.asarray(
        observation["mask"],
        dtype=np.bool_,
    )

    if mask.shape != (
        MAX_LOOPS,
    ):
        raise ValueError(
            "action mask must have shape "
            f"({MAX_LOOPS},), "
            f"got {mask.shape}"
        )

    legal_actions = np.flatnonzero(
        mask
    )

    if legal_actions.size == 0:
        raise ValueError(
            "Cannot select an action from "
            "a terminal/all-False mask"
        )

    actor.eval()

    with torch.inference_mode():

        (
            raw_logits,
            hidden_state,
        ) = actor(
            observation["obs"]
        )

    if hidden_state is not None:
        raise RuntimeError(
            "LoopyCutsActorV1 unexpectedly "
            "returned hidden state"
        )

    if not torch.is_tensor(
        raw_logits
    ):
        raise TypeError(
            "Actor logits are not a tensor"
        )

    if raw_logits.shape != (
        1,
        MAX_LOOPS,
    ):
        raise ValueError(
            "Actor logits must have shape "
            f"(1,{MAX_LOOPS}), "
            f"got {tuple(raw_logits.shape)}"
        )

    if not bool(
        torch.isfinite(
            raw_logits
        ).all()
    ):
        raise RuntimeError(
            "Actor produced non-finite "
            "raw logits"
        )

    mask_tensor = torch.as_tensor(
        mask,
        dtype=torch.bool,
        device=raw_logits.device,
    ).unsqueeze(
        0
    )

    masked_logits = (
        raw_logits.masked_fill(
            ~mask_tensor,
            float("-inf"),
        )
    )

    action = int(
        torch.argmax(
            masked_logits,
            dim=-1,
        ).item()
    )

    if not bool(
        mask[
            action
        ]
    ):
        raise RuntimeError(
            "Deterministic actor selected "
            "an illegal action"
        )

    return {
        "action":
            action,

        "raw_logits":
            (
                raw_logits[
                    0
                ]
                .detach()
                .cpu()
                .clone()
            ),

        "masked_logits":
            (
                masked_logits[
                    0
                ]
                .detach()
                .cpu()
                .clone()
            ),

        "legal_actions":
            tuple(
                int(x)
                for x
                in legal_actions.tolist()
            ),

        "selected_logit":
            float(
                raw_logits[
                    0,
                    action,
                ].item()
            ),
    }
