from __future__ import annotations

import math
import sys

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from bridge.resource_guard_v1 import (
    GIB,
    RESOURCE_GUARD_VERSION,
    ResourceGuardPolicyV1,
)

from bridge.cpp_client import (
    CPP_LEGACY_RSS_ASSERT_GUARD_STATE,
    CPP_LEGACY_RSS_ASSERT_SIGNATURE,
)

from rewards.reward_v3 import (
    DEFAULT_REWARD_V3_WEIGHTS,
    REWARD_V3_VERSION,
)

from training.formal_checkpoint_v1 import (
    FORMAL_CHECKPOINT_VERSION,
    formal_resource_guard_contract,
)

from training.formal_run_artifacts_v1 import (
    EVENT_LOG_FILENAME,
    FORMAL_RUN_ARTIFACTS_VERSION,
    FORMAL_RUN_EVENT_SCHEMA,
    FORMAL_RUN_MANIFEST_SCHEMA,
    RUN_MANIFEST_FILENAME,
    formal_resource_guard_manifest_contract,
)

from training.formal_training_v1 import (
    FORMAL_STAGE2_ONLINE_VERSION,
)

from training.protocol_v1 import (
    PROTOCOL_VERSION,
    PROJECT_REWARD_VERSION,
    PROJECT_RUNTIME_REWARD_VERSION,

    PROJECT_STAGE2_RESOURCE_GUARD_VERSION,
    PROJECT_STAGE2_RESOURCE_GUARD_SAMPLE_INTERVAL_SECONDS,
    PROJECT_STAGE2_RESOURCE_GUARD_WARNING_SWAP_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_ABORT_SWAP_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_ABORT_HOLD_SECONDS,
    PROJECT_STAGE2_RESOURCE_GUARD_EMERGENCY_SWAP_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_FINALIZE_EVAL_SWAP_ABORT_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_INITIALIZE_ENABLED,
    PROJECT_STAGE2_RESOURCE_GUARD_FINALIZE_EVAL_ADDS_TRANSITION,
    PROJECT_STAGE2_RESOURCE_GUARD_REARM_SWAP_GIB,
    PROJECT_STAGE2_RESOURCE_GUARD_REARM_TIMEOUT_SECONDS,
    PROJECT_STAGE2_RESOURCE_GUARD_ABORT_SCOPE,
    PROJECT_STAGE2_RESOURCE_GUARD_TERMINAL_OUTCOME,
    PROJECT_STAGE2_RESOURCE_GUARD_TERMINAL_REWARD,
    PROJECT_STAGE2_RESOURCE_GUARD_ABORT_COUNTS_AS_TRANSITION,
    PROJECT_STAGE2_RESOURCE_GUARD_DENSE_SHAPING_ENABLED,
    PROJECT_STAGE2_RESOURCE_GUARD_IMMEDIATE_CHECKPOINT,
    PROJECT_STAGE2_RESOURCE_GUARD_PREFLIGHT_REARM_REQUIRED,
    PROJECT_STAGE2_RESOURCE_GUARD_COLLECTOR_AUTORESET_POLICY,

    PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_COMPAT_ENABLED,
    PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_PHASE,
    PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_SIGNAL,
    PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_SIGNATURE,
    PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_GUARD_STATE,
)

from training.run_formal_training_v1 import (
    DEFAULT_FORMAL_RUN_ROOT,
    FORMAL_RESOURCE_REARM_SAMPLE_INTERVAL_SECONDS,
    FORMAL_RESOURCE_REARM_TIMEOUT_SECONDS,
    FORMAL_RUNNER_VERSION,
)


def main():
    assert (
        PROTOCOL_VERSION
        ==
        "loopycuts_training_protocol_v4_cpp_rss_compat"
    )

    assert (
        PROJECT_REWARD_VERSION
        ==
        "reward_v3"
    )

    assert (
        PROJECT_RUNTIME_REWARD_VERSION
        ==
        REWARD_V3_VERSION
        ==
        "final_v3_resource_guard"
    )

    assert (
        FORMAL_STAGE2_ONLINE_VERSION
        ==
        "loopycuts_formal_stage2_online_v4_cpp_rss_compat"
    )

    assert (
        FORMAL_RUNNER_VERSION
        ==
        "loopycuts_formal_runner_v4_cpp_rss_compat_metrics_v1"
    )

    assert (
        FORMAL_CHECKPOINT_VERSION
        ==
        "loopycuts_formal_checkpoint_v4_cpp_rss_compat"
    )

    assert (
        FORMAL_RUN_ARTIFACTS_VERSION
        ==
        "loopycuts_formal_run_artifacts_v4_cpp_rss_compat_metrics_v1"
    )

    assert (
        FORMAL_RUN_MANIFEST_SCHEMA
        ==
        "loopycuts_formal_run_manifest_v4_cpp_rss_compat_metrics_v1"
    )

    assert (
        FORMAL_RUN_EVENT_SCHEMA
        ==
        "loopycuts_formal_run_event_v4_cpp_rss_compat"
    )

    assert (
        RUN_MANIFEST_FILENAME
        ==
        "run_manifest_v4.json"
    )

    assert (
        EVENT_LOG_FILENAME
        ==
        "events_v4.jsonl"
    )

    assert (
        DEFAULT_FORMAL_RUN_ROOT
        ==
        Path(
            "/home/yjk/loopycuts_test/formal_training_v4"
        )
    )


    # ============================================================
    # Frozen ResourceGuard contract.
    # ============================================================

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_VERSION
        ==
        RESOURCE_GUARD_VERSION
        ==
        "loopycuts_resource_guard_v1"
    )

    assert math.isclose(
        PROJECT_STAGE2_RESOURCE_GUARD_SAMPLE_INTERVAL_SECONDS,
        1.0,
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_WARNING_SWAP_GIB
        ==
        8
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_ABORT_SWAP_GIB
        ==
        10
    )

    assert math.isclose(
        PROJECT_STAGE2_RESOURCE_GUARD_ABORT_HOLD_SECONDS,
        8.0,
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_EMERGENCY_SWAP_GIB
        ==
        12
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_FINALIZE_EVAL_SWAP_ABORT_GIB
        ==
        25
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_INITIALIZE_ENABLED
        is False
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_FINALIZE_EVAL_ADDS_TRANSITION
        is False
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_REARM_SWAP_GIB
        ==
        6
    )

    assert math.isclose(
        PROJECT_STAGE2_RESOURCE_GUARD_REARM_TIMEOUT_SECONDS,
        60.0,
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_ABORT_SCOPE
        ==
        "CURRENT_MODEL_EPISODE_ONLY"
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_TERMINAL_OUTCOME
        ==
        "RESOURCE_ABORT"
    )

    assert math.isclose(
        PROJECT_STAGE2_RESOURCE_GUARD_TERMINAL_REWARD,
        -4.0,
    )

    assert math.isclose(
        DEFAULT_REWARD_V3_WEIGHTS.final_resource_abort,
        4.0,
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_ABORT_COUNTS_AS_TRANSITION
        is True
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_DENSE_SHAPING_ENABLED
        is False
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_IMMEDIATE_CHECKPOINT
        is True
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_PREFLIGHT_REARM_REQUIRED
        is True
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_COLLECTOR_AUTORESET_POLICY
        ==
        "SUPPRESS_POST_TERMINAL_AUTORESET"
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_COMPAT_ENABLED
        is True
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_PHASE
        ==
        "STEP"
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_SIGNAL
        ==
        "SIGABRT"
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_SIGNATURE
        ==
        CPP_LEGACY_RSS_ASSERT_SIGNATURE
        ==
        "memory_usage_in_giga_bytes()<10"
    )

    assert (
        PROJECT_STAGE2_RESOURCE_GUARD_CPP_RSS_ASSERT_GUARD_STATE
        ==
        CPP_LEGACY_RSS_ASSERT_GUARD_STATE
        ==
        "RESOURCE_ABORT_CPP_RSS_LIMIT"
    )


    # ============================================================
    # Generic implementation defaults must match the formal freeze.
    # ============================================================

    policy = (
        ResourceGuardPolicyV1()
    )

    assert (
        policy.warning_swap_used_bytes
        ==
        8 * GIB
    )

    assert (
        policy.abort_swap_used_bytes
        ==
        10 * GIB
    )

    assert (
        policy.emergency_swap_used_bytes
        ==
        12 * GIB
    )

    assert (
        policy.rearm_swap_used_bytes
        ==
        6 * GIB
    )

    assert math.isclose(
        policy.abort_hold_seconds,
        8.0,
    )

    assert math.isclose(
        FORMAL_RESOURCE_REARM_SAMPLE_INTERVAL_SECONDS,
        1.0,
    )

    assert math.isclose(
        FORMAL_RESOURCE_REARM_TIMEOUT_SECONDS,
        60.0,
    )


    checkpoint_contract = (
        formal_resource_guard_contract()
    )

    manifest_contract = (
        formal_resource_guard_manifest_contract()
    )

    assert (
        checkpoint_contract[
            "cpp_rss_assert_compat_enabled"
        ]
        ==
        manifest_contract[
            "cpp_rss_assert_compat_enabled"
        ]
        is True
    )

    assert (
        checkpoint_contract[
            "cpp_rss_assert_phase"
        ]
        ==
        manifest_contract[
            "cpp_rss_assert_phase"
        ]
        ==
        "STEP"
    )

    assert (
        checkpoint_contract[
            "cpp_rss_assert_signal"
        ]
        ==
        manifest_contract[
            "cpp_rss_assert_signal"
        ]
        ==
        "SIGABRT"
    )

    assert (
        checkpoint_contract[
            "cpp_rss_assert_signature"
        ]
        ==
        manifest_contract[
            "cpp_rss_assert_signature"
        ]
        ==
        "memory_usage_in_giga_bytes()<10"
    )

    assert (
        checkpoint_contract[
            "cpp_rss_assert_guard_state"
        ]
        ==
        manifest_contract[
            "cpp_rss_assert_guard_state"
        ]
        ==
        "RESOURCE_ABORT_CPP_RSS_LIMIT"
    )

    assert (
        checkpoint_contract[
            "warning_swap_gib"
        ]
        ==
        manifest_contract[
            "warning_swap_gib"
        ]
        ==
        8
    )

    assert (
        checkpoint_contract[
            "abort_swap_gib"
        ]
        ==
        manifest_contract[
            "abort_swap_gib"
        ]
        ==
        10
    )

    assert (
        checkpoint_contract[
            "emergency_swap_gib"
        ]
        ==
        manifest_contract[
            "emergency_swap_gib"
        ]
        ==
        12
    )

    assert (
        checkpoint_contract[
            "finalize_eval_swap_abort_gib"
        ]
        ==
        manifest_contract[
            "finalize_eval_swap_abort_gib"
        ]
        ==
        25
    )

    assert (
        checkpoint_contract[
            "initialize_guard_enabled"
        ]
        is
        manifest_contract[
            "initialize_guard_enabled"
        ]
        is False
    )

    assert (
        checkpoint_contract[
            "finalize_eval_adds_transition"
        ]
        is
        manifest_contract[
            "finalize_eval_adds_transition"
        ]
        is False
    )

    assert (
        checkpoint_contract[
            "rearm_swap_gib"
        ]
        ==
        manifest_contract[
            "rearm_swap_gib"
        ]
        ==
        6
    )

    assert math.isclose(
        checkpoint_contract[
            "abort_hold_seconds"
        ],
        8.0,
    )

    assert math.isclose(
        checkpoint_contract[
            "terminal_reward"
        ],
        -4.0,
    )

    assert (
        manifest_contract[
            "runtime_reward_version"
        ]
        ==
        "final_v3_resource_guard"
    )


    print(
        "PASS: Training Protocol V4 CPP RSS compatibility identity is frozen"
    )

    print(
        "PASS: STEP 8/10/8s/12 GiB and re-arm 6 GiB thresholds are frozen"
    )

    print(
        "PASS: FINALIZE_EVAL 25 GiB hard cap is frozen"
    )

    print(
        "PASS: INITIALIZE remains unguarded by design"
    )

    print(
        "PASS: FINALIZE_EVAL RESOURCE_ABORT adds no new transition"
    )

    print(
        "PASS: RESOURCE_ABORT=-4 and no dense memory shaping are frozen"
    )

    print(
        "PASS: checkpoint and manifest freeze the same ResourceGuard contract"
    )

    print(
        "PASS: V4 artifacts use isolated manifest/event filenames"
    )

    print(
        "PASS: V4 formal runs use isolated formal_training_v4 root"
    )


if __name__ == "__main__":
    main()
