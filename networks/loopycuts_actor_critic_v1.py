from __future__ import annotations

from typing import Any

import numpy as np
import torch

from torch import nn

from observation.builder import (
    GLOBAL_DIM,
    LOOP_FEATURE_DIM,
    MAX_LOOPS,
)


NETWORK_VERSION = (
    "loopycuts_actor_critic_v1"
)

GLOBAL_EMBED_DIM = 128
LOOP_EMBED_DIM = 128

CONTEXT_DIM = (
    GLOBAL_EMBED_DIM
    +
    LOOP_EMBED_DIM
    +
    LOOP_EMBED_DIM
)

PER_ACTION_INPUT_DIM = (
    CONTEXT_DIM
    +
    LOOP_EMBED_DIM
)


class LoopyCutsNetworkError(
    ValueError
):
    pass


def _field(
    container,
    key: str,
):
    try:
        return container[
            key
        ]

    except Exception as exc:
        raise LoopyCutsNetworkError(
            "LoopyCuts network observation "
            f"is missing field {key!r}"
        ) from exc


def _unwrap_critic_observation(
    observation,
):
    """
    Actor input
    -----------
    MaskedDiscreteSACPolicy already unwraps:

        outer.obs

    and sends the inner state to Actor.

    Critic input
    ------------
    Tianshou DiscreteSAC directly sends:

        batch.obs

    which is the outer observation containing:

        obs
        mask

    The critic uses the state only.  The explicit action mask is
    deliberately NOT treated as a critic input feature.
    """

    if isinstance(
        observation,
        dict,
    ):
        if (
            "obs" in observation
            and
            "mask" in observation
        ):
            return observation[
                "obs"
            ]

        return observation

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
        return observation.obs

    return observation


class LoopyCutsStateActionNetworkV1(
    nn.Module
):
    """
    Shared ARCHITECTURE for the LoopyCuts V1 Actor and Critics.

    IMPORTANT:
        Actor, Critic-1 and Critic-2 each instantiate their own
        copy of this module.  Parameters are not shared.

    Input:
        global:
            [B, 16]

        loops:
            [B, 331, 14]

        exists:
            [B, 331]

    Output:
        one finite scalar per serialized loop:

            [B, 331]
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.global_encoder = (
            nn.Sequential(
                nn.Linear(
                    GLOBAL_DIM,
                    64,
                ),
                nn.LayerNorm(
                    64
                ),
                nn.SiLU(),

                nn.Linear(
                    64,
                    GLOBAL_EMBED_DIM,
                ),
                nn.LayerNorm(
                    GLOBAL_EMBED_DIM
                ),
                nn.SiLU(),
            )
        )

        self.loop_encoder = (
            nn.Sequential(
                nn.Linear(
                    LOOP_FEATURE_DIM,
                    64,
                ),
                nn.LayerNorm(
                    64
                ),
                nn.SiLU(),

                nn.Linear(
                    64,
                    LOOP_EMBED_DIM,
                ),
                nn.LayerNorm(
                    LOOP_EMBED_DIM
                ),
                nn.SiLU(),
            )
        )

        self.action_head = (
            nn.Sequential(
                nn.Linear(
                    PER_ACTION_INPUT_DIM,
                    256,
                ),
                nn.LayerNorm(
                    256
                ),
                nn.SiLU(),

                nn.Linear(
                    256,
                    128,
                ),
                nn.LayerNorm(
                    128
                ),
                nn.SiLU(),

                nn.Linear(
                    128,
                    1,
                ),
            )
        )

    def _device(
        self,
    ):
        return (
            self
            .global_encoder[
                0
            ]
            .weight
            .device
        )

    def _prepare_state(
        self,
        observation,
    ):
        device = (
            self._device()
        )

        global_features = (
            torch.as_tensor(
                _field(
                    observation,
                    "global",
                ),
                dtype=torch.float32,
                device=device,
            )
        )

        loop_features = (
            torch.as_tensor(
                _field(
                    observation,
                    "loops",
                ),
                dtype=torch.float32,
                device=device,
            )
        )

        exists = (
            torch.as_tensor(
                _field(
                    observation,
                    "exists",
                ),
                dtype=torch.bool,
                device=device,
            )
        )

        # --------------------------------------------------------
        # Support one unbatched observation for direct inference.
        # --------------------------------------------------------

        if global_features.ndim == 1:
            global_features = (
                global_features.unsqueeze(
                    0
                )
            )

        if loop_features.ndim == 2:
            loop_features = (
                loop_features.unsqueeze(
                    0
                )
            )

        if exists.ndim == 1:
            exists = (
                exists.unsqueeze(
                    0
                )
            )

        # --------------------------------------------------------
        # Frozen Observation V1 dimensions.
        # --------------------------------------------------------

        if (
            global_features.ndim
            !=
            2
            or
            global_features.shape[
                1
            ]
            !=
            GLOBAL_DIM
        ):
            raise LoopyCutsNetworkError(
                "Invalid global feature shape: "
                f"{tuple(global_features.shape)}; "
                f"expected [B,{GLOBAL_DIM}]"
            )

        if (
            loop_features.ndim
            !=
            3
            or
            loop_features.shape[
                1
            ]
            !=
            MAX_LOOPS
            or
            loop_features.shape[
                2
            ]
            !=
            LOOP_FEATURE_DIM
        ):
            raise LoopyCutsNetworkError(
                "Invalid loop feature shape: "
                f"{tuple(loop_features.shape)}; "
                "expected "
                f"[B,{MAX_LOOPS},"
                f"{LOOP_FEATURE_DIM}]"
            )

        if (
            exists.ndim
            !=
            2
            or
            exists.shape[
                1
            ]
            !=
            MAX_LOOPS
        ):
            raise LoopyCutsNetworkError(
                "Invalid exists shape: "
                f"{tuple(exists.shape)}; "
                f"expected [B,{MAX_LOOPS}]"
            )

        batch_size = (
            global_features.shape[
                0
            ]
        )

        if (
            loop_features.shape[
                0
            ]
            !=
            batch_size
            or
            exists.shape[
                0
            ]
            !=
            batch_size
        ):
            raise LoopyCutsNetworkError(
                "Observation batch dimensions "
                "do not match"
            )

        existing_count = (
            exists.sum(
                dim=1
            )
        )

        if bool(
            (
                existing_count
                <=
                0
            ).any()
        ):
            raise LoopyCutsNetworkError(
                "Every LoopyCuts state must "
                "contain at least one existing loop"
            )

        return (
            global_features,
            loop_features,
            exists,
        )

    def forward(
        self,
        observation,
    ):
        (
            global_features,
            loop_features,
            exists,
        ) = self._prepare_state(
            observation
        )

        # --------------------------------------------------------
        # Local embeddings.
        # --------------------------------------------------------

        global_embedding = (
            self.global_encoder(
                global_features
            )
        )

        loop_embedding = (
            self.loop_encoder(
                loop_features
            )
        )

        # --------------------------------------------------------
        # Padding-aware global loop context.
        #
        # Pool over EXISTING loops, not over the dynamic legal
        # action subset.
        # --------------------------------------------------------

        exists_3d = (
            exists.unsqueeze(
                -1
            )
        )

        exists_float = (
            exists_3d.to(
                dtype=
                    loop_embedding.dtype
            )
        )

        loop_sum = (
            (
                loop_embedding
                *
                exists_float
            )
            .sum(
                dim=1
            )
        )

        denominator = (
            exists_float
            .sum(
                dim=1
            )
            .clamp_min(
                1.0
            )
        )

        loop_mean = (
            loop_sum
            /
            denominator
        )

        # Use a finite minimum value rather than -inf.
        # The row is guaranteed to contain at least one existing loop.
        finite_min = (
            torch.finfo(
                loop_embedding.dtype
            ).min
        )

        loop_for_max = (
            loop_embedding.masked_fill(
                ~exists_3d,
                finite_min,
            )
        )

        loop_max = (
            loop_for_max.max(
                dim=1
            ).values
        )

        # --------------------------------------------------------
        # State-level context.
        # --------------------------------------------------------

        context = (
            torch.cat(
                [
                    global_embedding,
                    loop_mean,
                    loop_max,
                ],
                dim=-1,
            )
        )

        context_per_loop = (
            context
            .unsqueeze(
                1
            )
            .expand(
                -1,
                MAX_LOOPS,
                -1,
            )
        )

        # --------------------------------------------------------
        # One score / Q value per serialized loop.
        # --------------------------------------------------------

        per_action_features = (
            torch.cat(
                [
                    loop_embedding,
                    context_per_loop,
                ],
                dim=-1,
            )
        )

        output = (
            self.action_head(
                per_action_features
            )
            .squeeze(
                -1
            )
        )

        if (
            output.ndim
            !=
            2
            or
            output.shape[
                1
            ]
            !=
            MAX_LOOPS
        ):
            raise RuntimeError(
                "Internal Actor/Critic V1 "
                "output shape error"
            )

        if not bool(
            torch.isfinite(
                output
            ).all()
        ):
            raise RuntimeError(
                "Actor/Critic V1 produced "
                "non-finite values"
            )

        return output


class LoopyCutsActorV1(
    nn.Module
):
    """
    Actor:

        state
            ->
        one raw logit per loop

        [B,331]

    Dynamic legality is applied afterwards by
    MaskedDiscreteSACPolicy.
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.network = (
            LoopyCutsStateActionNetworkV1()
        )

    def forward(
        self,
        obs,
        state=None,
        info: Any = None,
    ):
        logits = (
            self.network(
                obs
            )
        )

        return (
            logits,
            None,
        )


class LoopyCutsCriticV1(
    nn.Module
):
    """
    Discrete Q critic:

        state
            ->
        Q(s,a) for all 331 serialized loop IDs

        [B,331]

    No action masking is applied to Q itself.  All Q outputs stay
    finite.  The policy distribution supplies zero probability to
    illegal actions.
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.network = (
            LoopyCutsStateActionNetworkV1()
        )

    def forward(
        self,
        observation,
        state=None,
        info: Any = None,
    ):
        state_observation = (
            _unwrap_critic_observation(
                observation
            )
        )

        return self.network(
            state_observation
        )


def build_loopycuts_actor_critics_v1(
    *,
    device: str | torch.device = "cpu",
):
    """
    Build three fully independent networks.

    No parameters are shared between:

        Actor
        Critic-1
        Critic-2
    """

    device = torch.device(
        device
    )

    actor = (
        LoopyCutsActorV1()
        .to(
            device
        )
    )

    critic1 = (
        LoopyCutsCriticV1()
        .to(
            device
        )
    )

    critic2 = (
        LoopyCutsCriticV1()
        .to(
            device
        )
    )

    return (
        actor,
        critic1,
        critic2,
    )


def count_trainable_parameters(
    module: nn.Module,
):
    return sum(
        parameter.numel()
        for parameter
        in module.parameters()
        if parameter.requires_grad
    )
