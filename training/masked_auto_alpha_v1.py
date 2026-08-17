from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from tianshou.algorithm.modelfree.sac import (
    Alpha,
)

from tianshou.algorithm.optim import (
    OptimizerFactory,
)


MASKED_AUTO_ALPHA_VERSION = (
    "loopycuts_masked_auto_alpha_v1"
)


class MaskedAutoAlphaError(
    ValueError
):
    pass


@dataclass(
    frozen=True
)
class MaskedAutoAlphaUpdate:
    loss: float

    alpha_before: float
    alpha_after: float

    mean_entropy: float
    mean_target_entropy: float

    min_legal_actions: int
    max_legal_actions: int

    batch_size: int


class MaskedAutoAlphaV1(
    torch.nn.Module,
    Alpha,
):
    """
    Auto-tuned SAC entropy coefficient for LoopyCuts'
    dynamically masked discrete action space.

    Target entropy for current state s:

        H_target(s)
            =
        coefficient * log(n_legal(s))

    V1 project protocol freezes:

        coefficient = 0.6

    but this class keeps the coefficient explicit for regression
    testing.

    Important:
        update(entropy) alone is deliberately forbidden.

    The current legal-action mask is required, so callers must use:

        update_with_mask(
            entropy,
            mask,
        )
    """

    def __init__(
        self,
        *,
        target_coefficient: float,
        initial_alpha: float,
        optim: OptimizerFactory,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        target_coefficient = float(
            target_coefficient
        )

        initial_alpha = float(
            initial_alpha
        )

        if (
            not np.isfinite(
                target_coefficient
            )
            or
            target_coefficient < 0.0
            or
            target_coefficient > 1.0
        ):
            raise MaskedAutoAlphaError(
                "target_coefficient must be "
                "finite and in [0, 1]"
            )

        if (
            not np.isfinite(
                initial_alpha
            )
            or
            initial_alpha <= 0.0
        ):
            raise MaskedAutoAlphaError(
                "initial_alpha must be "
                "finite and positive"
            )

        self.target_coefficient = (
            target_coefficient
        )

        device = torch.device(
            device
        )

        self._log_alpha = (
            torch.nn.Parameter(
                torch.tensor(
                    math.log(
                        initial_alpha
                    ),
                    dtype=torch.float32,
                    device=device,
                )
            )
        )

        (
            self._optim,
            lr_scheduler,
        ) = optim.create_instances(
            self
        )

        if lr_scheduler is not None:
            raise MaskedAutoAlphaError(
                "MaskedAutoAlphaV1 does not "
                "support an alpha LR scheduler"
            )

        self._last_update: (
            MaskedAutoAlphaUpdate
            | None
        ) = None

    @property
    def value(
        self,
    ) -> float:
        return float(
            self._log_alpha
            .detach()
            .exp()
            .item()
        )

    @property
    def last_update(
        self,
    ) -> (
        MaskedAutoAlphaUpdate
        | None
    ):
        return self._last_update

    def update(
        self,
        entropy: torch.Tensor,
    ) -> float:
        """
        The ordinary Tianshou Alpha interface does not contain the
        current legal-action mask.

        Silently using a fixed entropy target here would violate
        Training Protocol V1, so fail explicitly.
        """

        raise RuntimeError(
            "MaskedAutoAlphaV1 requires "
            "update_with_mask(entropy, mask)"
        )

    def target_entropy_from_mask(
        self,
        *,
        mask,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        mask_tensor = torch.as_tensor(
            mask,
            dtype=torch.bool,
            device=device,
        )

        if (
            mask_tensor.ndim == 1
            and
            batch_size == 1
        ):
            mask_tensor = (
                mask_tensor.unsqueeze(
                    0
                )
            )

        if mask_tensor.ndim != 2:
            raise MaskedAutoAlphaError(
                "Current action mask must have "
                "shape [batch_size, action_dim]"
            )

        if int(
            mask_tensor.shape[
                0
            ]
        ) != int(
            batch_size
        ):
            raise MaskedAutoAlphaError(
                "Mask batch dimension does not "
                "match entropy batch dimension"
            )

        legal_counts = (
            mask_tensor
            .sum(
                dim=-1
            )
        )

        if bool(
            (
                legal_counts
                <=
                0
            ).any()
        ):
            bad_row = int(
                torch.nonzero(
                    legal_counts
                    <=
                    0,
                    as_tuple=False,
                )[
                    0
                ].item()
            )

            raise MaskedAutoAlphaError(
                "Masked auto-alpha current state "
                "has no legal actions at "
                f"batch row {bad_row}"
            )

        target_entropy = (
            self.target_coefficient
            *
            torch.log(
                legal_counts.to(
                    dtype=dtype
                )
            )
        )

        if not bool(
            torch.isfinite(
                target_entropy
            ).all()
        ):
            raise MaskedAutoAlphaError(
                "Target entropy contains "
                "non-finite values"
            )

        return (
            target_entropy,
            legal_counts,
        )

    def update_with_mask(
        self,
        *,
        entropy: torch.Tensor,
        mask,
    ) -> MaskedAutoAlphaUpdate:

        if not isinstance(
            entropy,
            torch.Tensor,
        ):
            raise MaskedAutoAlphaError(
                "entropy must be a torch.Tensor"
            )

        entropy = (
            entropy
            .detach()
            .reshape(
                -1
            )
        )

        if entropy.numel() <= 0:
            raise MaskedAutoAlphaError(
                "entropy batch is empty"
            )

        if not bool(
            torch.isfinite(
                entropy
            ).all()
        ):
            raise MaskedAutoAlphaError(
                "entropy contains non-finite values"
            )

        if (
            entropy.device
            !=
            self._log_alpha.device
        ):
            raise MaskedAutoAlphaError(
                "Entropy device does not match "
                "MaskedAutoAlphaV1 device: "
                f"entropy={entropy.device}, "
                f"alpha={self._log_alpha.device}"
            )

        batch_size = int(
            entropy.shape[
                0
            ]
        )

        (
            target_entropy,
            legal_counts,
        ) = self.target_entropy_from_mask(
            mask=mask,
            dtype=entropy.dtype,
            device=entropy.device,
            batch_size=batch_size,
        )

        entropy_deficit = (
            target_entropy
            -
            entropy
        )

        alpha_before = (
            self.value
        )

        # Match Tianshou AutoAlpha's optimization convention:
        #
        #     L_alpha =
        #       -log(alpha)
        #       * (H_target - H)
        #
        # but H_target is now state-dependent.
        alpha_loss = (
            -(
                self._log_alpha
                *
                entropy_deficit
            )
            .mean()
        )

        if not bool(
            torch.isfinite(
                alpha_loss
            )
        ):
            raise MaskedAutoAlphaError(
                "Alpha loss is non-finite"
            )

        self._optim.zero_grad()

        alpha_loss.backward()

        self._optim.step()

        if not bool(
            torch.isfinite(
                self._log_alpha
            )
        ):
            raise MaskedAutoAlphaError(
                "log_alpha became non-finite"
            )

        alpha_after = (
            self.value
        )

        if (
            not np.isfinite(
                alpha_after
            )
            or
            alpha_after <= 0.0
        ):
            raise MaskedAutoAlphaError(
                "alpha became invalid"
            )

        update = (
            MaskedAutoAlphaUpdate(
                loss=
                    float(
                        alpha_loss
                        .detach()
                        .item()
                    ),

                alpha_before=
                    alpha_before,

                alpha_after=
                    alpha_after,

                mean_entropy=
                    float(
                        entropy
                        .mean()
                        .item()
                    ),

                mean_target_entropy=
                    float(
                        target_entropy
                        .mean()
                        .item()
                    ),

                min_legal_actions=
                    int(
                        legal_counts
                        .min()
                        .item()
                    ),

                max_legal_actions=
                    int(
                        legal_counts
                        .max()
                        .item()
                    ),

                batch_size=
                    batch_size,
            )
        )

        self._last_update = (
            update
        )

        return update
