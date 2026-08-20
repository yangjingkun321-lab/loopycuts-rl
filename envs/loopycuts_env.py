from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bridge.cpp_client import (
    LoopyCutsClient,
    RLServerResourceAbort,
)
from dataset_tools.loop_metadata import parse_loop_metadata
from observation.builder import (
    GLOBAL_DIM,
    LOOP_FEATURE_DIM,
    MAX_LOOPS,
    LoopyCutsObservationBuilder,
)
from rewards.reward_v1 import (
    compute_reward_v1,
)
from rewards.transition_metrics import (
    extract_transition_metrics,
)


class LoopyCutsEnv(gym.Env):
    """
    Gymnasium wrapper for LoopyCuts Stage-2 loop selection.

    V1 responsibility:
        - own one persistent C++ RL-server process per episode;
        - preserve original LoopyCuts loop IDs as actions;
        - expose the frozen Observation V1;
        - enforce C++ ACTIONS as authoritative legality;
        - maintain Python-side executed_loop_ids;
        - terminate when the C++ Stage-2 selection state is terminal.

    Reward:
        Selection Reward V1.

        The reward is computed from the real C++ transition using:
            state_before
            -> STEP_RESULT
            -> state_after
            -> TransitionMetrics
            -> Reward V1

        FINALIZE/full-hex outcome is deliberately not part of
        Selection Reward V1.
    """

    metadata = {
        "render_modes": [],
    }

    reward_range = (
        -np.inf,
        np.inf,
    )

    def __init__(
        self,
        *,
        executable: str | Path,
        mesh_file: str | Path,
        loop_file: str | Path,
        echo_logs: bool = False,
        resource_guard_policy=None,
        resource_guard_sample_interval_seconds: float = 1.0,
        resource_snapshot_reader=None,
        finalize_eval_swap_abort_bytes=None,
    ):
        super().__init__()

        self.executable = Path(
            executable
        )

        self.mesh_file = Path(
            mesh_file
        )

        self.loop_file = Path(
            loop_file
        )

        self.echo_logs = bool(
            echo_logs
        )

        # ResourceGuard remains opt-in at the environment level
        # until the later formal V3 protocol integration.
        self.resource_guard_policy = (
            resource_guard_policy
        )

        self.resource_guard_sample_interval_seconds = float(
            resource_guard_sample_interval_seconds
        )

        self.resource_snapshot_reader = (
            resource_snapshot_reader
        )

        self.finalize_eval_swap_abort_bytes = (
            finalize_eval_swap_abort_bytes
        )

        # ------------------------------------------------------------
        # Fail early on missing input paths.
        # ------------------------------------------------------------

        if not self.executable.is_file():
            raise FileNotFoundError(
                f"LoopyCuts executable not found: "
                f"{self.executable}"
            )

        if not self.mesh_file.is_file():
            raise FileNotFoundError(
                f"Mesh file not found: "
                f"{self.mesh_file}"
            )

        if not self.loop_file.is_file():
            raise FileNotFoundError(
                f"Loop file not found: "
                f"{self.loop_file}"
            )

        # ------------------------------------------------------------
        # Static Stage-1 metadata is immutable across episodes.
        # Parse once.
        # ------------------------------------------------------------

        self.loop_metadata = (
            parse_loop_metadata(
                self.loop_file
            )
        )

        self.num_loops = len(
            self.loop_metadata
        )

        if self.num_loops > MAX_LOOPS:
            raise ValueError(
                f"Model has {self.num_loops} serialized loops, "
                f"but MAX_LOOPS={MAX_LOOPS}. "
                "V1 must not truncate or remap loop IDs."
            )

        # ------------------------------------------------------------
        # Action ID == original LoopyCuts loop ID.
        #
        # Discrete(331) contains the entire fixed author-corpus
        # namespace. Dynamic legality is supplied ONLY by the
        # observation mask emitted from authoritative C++ ACTIONS.
        # ------------------------------------------------------------

        self.action_space = spaces.Discrete(
            MAX_LOOPS
        )

        # ------------------------------------------------------------
        # Observation V1.
        #
        # MultiBinary is used for exists/mask because they are
        # semantically binary arrays. The Observation Builder itself
        # deliberately returns np.bool_ arrays.
        # ------------------------------------------------------------

        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Dict(
                    {
                        "global": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(
                                GLOBAL_DIM,
                            ),
                            dtype=np.float32,
                        ),

                        "loops": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(
                                MAX_LOOPS,
                                LOOP_FEATURE_DIM,
                            ),
                            dtype=np.float32,
                        ),

                        "exists": spaces.MultiBinary(
                            MAX_LOOPS
                        ),
                    }
                ),

                "mask": spaces.MultiBinary(
                    MAX_LOOPS
                ),
            }
        )

        # ------------------------------------------------------------
        # Episode-owned objects.
        # ------------------------------------------------------------

        self._client: (
            LoopyCutsClient
            | None
        ) = None

        self._builder: (
            LoopyCutsObservationBuilder
            | None
        ) = None

        self.executed_loop_ids: set[int] = set()

    # ------------------------------------------------------------------

    @property
    def client(
        self,
    ) -> LoopyCutsClient:
        if self._client is None:
            raise RuntimeError(
                "Environment has not been reset"
            )

        return self._client

    # ------------------------------------------------------------------

    @property
    def legal_actions(
        self,
    ) -> tuple[int, ...]:
        if self._client is None:
            return ()

        return tuple(
            self._client.actions
        )

    # ------------------------------------------------------------------

    @property
    def current_state(
        self,
    ) -> dict[str, Any] | None:
        if (
            self._client is None
            or
            self._client.state is None
        ):
            return None

        return dict(
            self._client.state
        )

    # ------------------------------------------------------------------

    def _build_observation(
        self,
    ) -> dict:
        if self._builder is None:
            raise RuntimeError(
                "Observation builder is not initialized"
            )

        client = self.client

        return self._builder.build(
            state=client.state,
            actions=client.actions,
            used=client.used,
            reverted=client.reverted,
            nico_bug=client.nico_bug,
            top_relevant=client.top_relevant,
            executed=self.executed_loop_ids,
        )

    # ------------------------------------------------------------------

    def _make_info(
        self,
        *,
        step_result: dict | None = None,
    ) -> dict[str, Any]:
        """
        Auxiliary information only.

        None of these fields are required for the policy observation.
        """

        client = self.client

        info: dict[str, Any] = {
            "state":
                dict(
                    client.state
                ),

            "num_legal_actions":
                len(
                    client.actions
                ),

            "num_executed":
                len(
                    self.executed_loop_ids
                ),

            "reward_is_placeholder":
                False,

            "reward_version":
                "selection_v1",
        }

        if step_result is not None:
            info[
                "step_result"
            ] = dict(
                step_result
            )

        return info

    # ------------------------------------------------------------------

    def _resource_abort_transition(
        self,
        *,
        action: int,
        state_before: dict,
        exc: RLServerResourceAbort,
    ):
        """
        Convert one ResourceGuard-aborted C++ STEP into a genuine
        terminal Gym transition.

        The selected action is real, but C++ did not return a complete
        STEP_RESULT/state_after. Therefore:

            - do NOT fabricate geometric transition metrics;
            - do NOT pretend the cut committed/reverted;
            - do preserve the action as an attempted agent action;
            - return a synthetic terminal observation whose action
              mask is empty.

        Reward is deliberately a placeholder here. The outer reward
        layer will define RESOURCE_ABORT reward semantics in the next
        integration phase.
        """

        if self._builder is None:
            raise RuntimeError(
                "Observation builder is not initialized"
            )

        client = self.client

        # The agent really selected/sent this loop ID.
        self.executed_loop_ids.add(
            int(
                action
            )
        )

        # ------------------------------------------------------------
        # Synthetic terminal state.
        #
        # Geometry fields remain at the last authoritative pre-STEP
        # C++ state because the aborted STEP never returned a complete
        # post-state.
        #
        # The RL transition itself still consumes one action, so the
        # synthetic step index advances by one.
        # ------------------------------------------------------------

        terminal_state = dict(
            state_before
        )

        terminal_state[
            "step"
        ] = (
            int(
                state_before[
                    "step"
                ]
            )
            +
            1
        )

        terminal_state[
            "available"
        ] = 0

        terminal_state[
            "terminal"
        ] = 1

        terminal_state[
            "selection_success"
        ] = 0

        terminal_state[
            "finalized"
        ] = 0

        observation = (
            self._builder.build(
                state=
                    terminal_state,

                actions=
                    [],

                used=
                    client.used,

                reverted=
                    client.reverted,

                nico_bug=
                    client.nico_bug,

                top_relevant=
                    client.top_relevant,

                executed=
                    self.executed_loop_ids,
            )
        )

        if not self.observation_space.contains(
            observation
        ):
            raise RuntimeError(
                "RESOURCE_ABORT synthetic terminal observation "
                "is not contained in observation_space"
            )

        snapshot = (
            exc.snapshot
        )

        cpp_memory = (
            snapshot.cpp_memory
        )

        cpp_rss_bytes = 0
        cpp_swap_bytes = 0

        if cpp_memory is not None:
            cpp_rss_bytes = int(
                cpp_memory.rss_bytes
            )

            cpp_swap_bytes = int(
                cpp_memory.swap_bytes
            )

        resource_abort = {
            "outcome":
                "RESOURCE_ABORT",

            "phase":
                str(
                    exc.phase
                ),

            "guard_state":
                str(
                    exc.guard_state
                ),

            "action":
                int(
                    action
                ),

            "return_code":
                (
                    None
                    if exc.return_code is None
                    else int(
                        exc.return_code
                    )
                ),

            "swap_used_bytes":
                int(
                    snapshot.swap_used_bytes
                ),

            "swap_total_bytes":
                int(
                    snapshot.swap_total_bytes
                ),

            "swap_free_bytes":
                int(
                    snapshot.swap_free_bytes
                ),

            "mem_available_bytes":
                int(
                    snapshot.mem_available_bytes
                ),

            "python_rss_bytes":
                int(
                    snapshot
                    .python_memory
                    .rss_bytes
                ),

            "python_swap_bytes":
                int(
                    snapshot
                    .python_memory
                    .swap_bytes
                ),

            "cpp_rss_bytes":
                cpp_rss_bytes,

            "cpp_swap_bytes":
                cpp_swap_bytes,
        }

        info = {
            "state":
                terminal_state,

            "num_legal_actions":
                0,

            "num_executed":
                len(
                    self.executed_loop_ids
                ),

            # The inner Selection Reward V1 cannot be computed because
            # the C++ STEP never yielded real post-transition geometry.
            "reward_is_placeholder":
                True,

            "reward_version":
                "selection_v1_resource_abort_placeholder",

            "resource_abort":
                resource_abort,
        }

        return (
            observation,

            0.0,

            True,

            False,

            info,
        )


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        #
        # Gymnasium seeding contract.
        #
        super().reset(
            seed=seed
        )

        #
        # V1 fixed-model environment currently has no reset options.
        # Accept None or an empty dictionary only.
        #
        if (
            options is not None
            and len(options) != 0
        ):
            raise ValueError(
                "LoopyCutsEnv V1 does not support "
                "non-empty reset options"
            )

        # ------------------------------------------------------------
        # One fresh persistent C++ GlobalState per episode.
        #
        # If reset() is called for a second episode, close the previous
        # child process first.
        # ------------------------------------------------------------

        self.close()

        self.executed_loop_ids = set()

        client = LoopyCutsClient(
            executable=self.executable,
            mesh_file=self.mesh_file,
            loop_file=self.loop_file,
            echo_logs=self.echo_logs,

            resource_guard_policy=
                self.resource_guard_policy,

            resource_guard_sample_interval_seconds=
                self.resource_guard_sample_interval_seconds,

            resource_snapshot_reader=
                self.resource_snapshot_reader,

            finalize_eval_swap_abort_bytes=
                self.finalize_eval_swap_abort_bytes,
        )

        try:
            builder = (
                LoopyCutsObservationBuilder(
                    metadata=self.loop_metadata,
                    initial_state=client.state,
                    initial_actions=client.actions,
                )
            )

            self._client = client
            self._builder = builder

            observation = (
                self._build_observation()
            )

            if not self.observation_space.contains(
                observation
            ):
                raise RuntimeError(
                    "Initial observation is not contained "
                    "in observation_space"
                )

            info = self._make_info()

        except Exception:
            client.close()

            self._client = None
            self._builder = None

            raise

        return (
            observation,
            info,
        )

    # ------------------------------------------------------------------

    def step(
        self,
        action,
    ):
        client = self.client

        # ------------------------------------------------------------
        # Stepping after terminal is a caller error.
        # ------------------------------------------------------------

        if int(
            client.state[
                "terminal"
            ]
        ):
            raise RuntimeError(
                "Cannot call step() after terminal state; "
                "call reset() first"
            )

        # ------------------------------------------------------------
        # Do not silently coerce floats or booleans into loop IDs.
        # ------------------------------------------------------------

        if isinstance(
            action,
            (
                bool,
                np.bool_,
            ),
        ):
            raise TypeError(
                "Action must be an integer loop ID, not bool"
            )

        if not isinstance(
            action,
            (
                int,
                np.integer,
            ),
        ):
            raise TypeError(
                "Action must be an integer loop ID"
            )

        action = int(
            action
        )

        if not self.action_space.contains(
            action
        ):
            raise ValueError(
                f"Action {action} is outside "
                f"Discrete({MAX_LOOPS})"
            )

        # ------------------------------------------------------------
        # C++ ACTIONS is the one and only legality authority.
        #
        # Do NOT reconstruct legality from:
        #
        #     loop type,
        #     used/reverted,
        #     convergence,
        #     regular_phase_closed,
        #     or Python metadata.
        # ------------------------------------------------------------

        if action not in client.actions:
            raise ValueError(
                f"Illegal LoopyCuts action {action}. "
                f"Current legal actions: "
                f"{client.actions}"
            )

        # ------------------------------------------------------------
        # Execute one real Stage-2 action.
        #
        # Reward V1 must be derived from the actual transition, so
        # preserve the pre-action C++ STATE before STEP mutates it.
        # ------------------------------------------------------------

        state_before = dict(
            client.state
        )

        try:
            (
                step_result,
                _,
                _,
            ) = client.step(
                action
            )

        except RLServerResourceAbort as exc:
            return self._resource_abort_transition(
                action=
                    action,

                state_before=
                    state_before,

                exc=
                    exc,
            )

        #
        # executed means the agent actually selected this action.
        #
        # This is deliberately distinct from C++ Loop.used because
        # find_mates() may consume loops that the agent never selected.
        #
        self.executed_loop_ids.add(
            action
        )

        observation = (
            self._build_observation()
        )

        if not self.observation_space.contains(
            observation
        ):
            raise RuntimeError(
                "Post-step observation is not contained "
                "in observation_space"
            )

        # ------------------------------------------------------------
        # Selection Reward V1.
        # ------------------------------------------------------------

        state_after = dict(
            client.state
        )

        transition_metrics = (
            extract_transition_metrics(
                state_before=state_before,
                step_result=step_result,
                state_after=state_after,
            )
        )

        if self._builder is None:
            raise RuntimeError(
                "Observation builder is not initialized"
            )

        reward_breakdown = (
            compute_reward_v1(
                metrics=transition_metrics,
                initial_actionable_count=(
                    self._builder
                    .initial_actionable_count
                ),
            )
        )

        reward = float(
            reward_breakdown.total
        )

        terminated = bool(
            client.state[
                "terminal"
            ]
        )

        #
        # No artificial time limit exists in V1.
        # Stage-2 terminates through the C++ legal-action semantics.
        #
        truncated = False

        info = self._make_info(
            step_result=step_result
        )

        info[
            "transition_metrics"
        ] = transition_metrics.to_dict()

        info[
            "reward_breakdown"
        ] = {
            "step":
                reward_breakdown.step,

            "tet_growth":
                reward_breakdown.tet_growth,

            "revert":
                reward_breakdown.revert,

            "convergence":
                reward_breakdown.convergence,

            "terminal":
                reward_breakdown.terminal,

            "total":
                reward_breakdown.total,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # ------------------------------------------------------------------

    def render(
        self,
    ):
        return None

    # ------------------------------------------------------------------

    def close(
        self,
    ):
        if self._client is not None:
            try:
                self._client.close()

            finally:
                self._client = None
                self._builder = None
