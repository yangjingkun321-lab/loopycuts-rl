import signal
import subprocess
import threading

from dataclasses import dataclass
from pathlib import Path

from bridge.resource_guard_v1 import (
    RESOURCE_STATE_ABORT,
    RESOURCE_STATE_ABORT_EMERGENCY,

    ResourceGuardPolicyV1,
    ResourceSnapshot,

    read_resource_snapshot,
)


class RLServerProtocolError(RuntimeError):
    """
    The RL server process is alive, but the communication
    protocol returned an invalid or unexpected response.

    This is different from a C++ process crash.
    """

    pass


class RLServerProcessError(RuntimeError):
    """
    The C++ RL server process terminated unexpectedly.

    The exception preserves:
        - execution phase
        - process return code
        - terminating signal, when available
        - partial C++ stdout/stderr received before termination
    """

    def __init__(
        self,
        phase,
        return_code,
        expected_prefix=None,
        lines=None,
    ):
        self.phase = phase
        self.return_code = return_code
        self.expected_prefix = expected_prefix
        self.lines = list(lines or [])

        self.signal_number = None
        self.signal_name = None

        #
        # On POSIX, subprocess.Popen reports a process
        # terminated by a signal using a negative return code.
        #
        # Example:
        #
        #     SIGABRT -> return_code = -6
        #
        if (
            isinstance(return_code, int)
            and return_code < 0
        ):
            self.signal_number = -return_code

            try:
                self.signal_name = (
                    signal.Signals(
                        self.signal_number
                    ).name
                )
            except ValueError:
                self.signal_name = None

        message = (
            "RL server terminated unexpectedly "
            f"during {phase}; "
            f"return_code={return_code}"
        )

        if self.signal_name is not None:
            message += (
                f"; signal={self.signal_name}"
                f"({self.signal_number})"
            )

        if expected_prefix is not None:
            message += (
                f"; while waiting for "
                f"{expected_prefix}"
            )

        super().__init__(message)



@dataclass(frozen=True)
class RLServerResourceAbortRecord:
    phase: str
    guard_state: str
    snapshot: ResourceSnapshot


class RLServerResourceAbort(RuntimeError):
    """
    The ResourceGuard intentionally terminated the CURRENT
    LoopyCuts C++ process.

    This is NOT an unexpected C++ crash and does NOT mean the
    whole formal training run should terminate.

    Higher layers will later convert STEP-time resource aborts
    into a RESOURCE_ABORT terminal RL transition.
    """

    def __init__(
        self,
        *,
        phase,
        guard_state,
        snapshot,
        return_code,
    ):
        self.phase = str(
            phase
        )

        self.guard_state = str(
            guard_state
        )

        self.snapshot = snapshot

        self.return_code = (
            return_code
        )

        swap_used_gib = (
            float(
                snapshot.swap_used_bytes
            )
            /
            float(
                1024 ** 3
            )
        )

        message = (
            "ResourceGuard terminated current "
            "LoopyCuts episode during "
            f"{self.phase}; "
            f"guard_state={self.guard_state}; "
            f"swap_used_gib={swap_used_gib:.3f}; "
            f"return_code={self.return_code}"
        )

        super().__init__(
            message
        )


class RLServerResourceGuardError(
    RuntimeError
):
    """
    Infrastructure failure inside the ResourceGuard monitor.

    Unlike RESOURCE_ABORT, this must NOT later be converted into
    an RL penalty, because it is not caused by the agent's action.
    """

    pass



FINALIZE_EVAL_SWAP_CAP_GUARD_STATE = (
    "RESOURCE_ABORT_FINALIZE_EVAL_SWAP_CAP"
)


class LoopyCutsClient:
    def __init__(
        self,
        executable,
        mesh_file,
        loop_file,
        echo_logs=False,
        resource_guard_policy=None,
        resource_guard_sample_interval_seconds=1.0,
        resource_snapshot_reader=None,
        finalize_eval_swap_abort_bytes=None,
    ):
        self.executable = str(executable)
        self.mesh_file = str(mesh_file)
        self.loop_file = str(loop_file)
        self.echo_logs = echo_logs

        self.resource_guard_policy = (
            resource_guard_policy
        )

        if (
            self.resource_guard_policy
            is not None
            and
            not isinstance(
                self.resource_guard_policy,
                ResourceGuardPolicyV1,
            )
        ):
            raise TypeError(
                "resource_guard_policy must be "
                "ResourceGuardPolicyV1 or None"
            )

        self.resource_guard_sample_interval_seconds = float(
            resource_guard_sample_interval_seconds
        )

        if (
            self.resource_guard_sample_interval_seconds
            <=
            0.0
        ):
            raise ValueError(
                "resource_guard_sample_interval_seconds "
                "must be positive"
            )

        if resource_snapshot_reader is None:
            resource_snapshot_reader = (
                read_resource_snapshot
            )

        if not callable(
            resource_snapshot_reader
        ):
            raise TypeError(
                "resource_snapshot_reader must be callable"
            )

        self._resource_snapshot_reader = (
            resource_snapshot_reader
        )

        self.finalize_eval_swap_abort_bytes = None

        if (
            finalize_eval_swap_abort_bytes
            is not None
        ):
            finalize_eval_swap_abort_bytes = int(
                finalize_eval_swap_abort_bytes
            )

            if finalize_eval_swap_abort_bytes <= 0:
                raise ValueError(
                    "finalize_eval_swap_abort_bytes "
                    "must be positive or None"
                )

            self.finalize_eval_swap_abort_bytes = (
                finalize_eval_swap_abort_bytes
            )

        self._resource_guard_lock = (
            threading.Lock()
        )

        self.last_resource_snapshot = None

        self.last_resource_guard_record = None

        self.last_resource_guard_monitor_error = None

        # A fresh LoopyCutsClient corresponds to one fresh
        # LoopyCuts episode. The >=10 GiB continuous-hold timer
        # is reset exactly here, not once per STEP.
        if self.resource_guard_policy is not None:
            self.resource_guard_policy.reset_episode()

        self.process = subprocess.Popen(
            [
                self.executable,
                self.mesh_file,
                self.loop_file,
                "-rl-server",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if self.process.stdin is None:
            raise RuntimeError(
                "Failed to open RL server stdin"
            )

        if self.process.stdout is None:
            raise RuntimeError(
                "Failed to open RL server stdout"
            )

        self.state = None
        self.actions = []

        #
        # Dynamic per-loop status emitted by the C++ RL server.
        #
        # These are original LoopyCuts loop IDs. They are descriptive
        # state only; self.actions remains the authoritative legality
        # source.
        #
        self.used = []
        self.reverted = []
        self.nico_bug = []
        self.top_relevant = []

        #
        # Preserve the most recent complete or partial
        # response for debugging.
        #
        self.last_response_lines = []

        #
        # Server initialization prints many normal
        # LoopyCuts messages.
        #
        # Wait until the first complete RL state has
        # been emitted.
        #
        lines = self._read_until(
            "[RL] ACTIONS",
            phase="INITIALIZE",
        )

        self._update_from_lines(lines)

        if not any(
            line.startswith("[RL] READY")
            for line in lines
        ):
            raise RLServerProtocolError(
                "RL server did not emit "
                "[RL] READY"
            )

    # ------------------------------------------------------------------

    def _resource_guard_worker(
        self,
        *,
        phase,
        stop_event,
        operation_done_event,
    ):
        """
        Watch one blocking C++ operation.

        The worker NEVER terminates the Python trainer.

        If the frozen ResourceGuard policy reaches an abort state,
        only self.process (the current volumetric_cutter) is killed.
        """

        policy = (
            self.resource_guard_policy
        )

        if policy is None:
            return

        while not stop_event.is_set():
            if (
                self.process.poll()
                is not None
            ):
                return

            try:
                snapshot = (
                    self._resource_snapshot_reader(
                        cpp_pid=
                            self.process.pid
                    )
                )

            except Exception as exc:
                # Fail closed:
                #
                # if the formal resource monitor itself becomes
                # unavailable, do not allow an unprotected C++
                # operation to continue indefinitely.
                with self._resource_guard_lock:
                    self.last_resource_guard_monitor_error = (
                        exc
                    )

                if (
                    not operation_done_event.is_set()
                    and
                    self.process.poll()
                    is None
                ):
                    try:
                        self.process.kill()
                    except ProcessLookupError:
                        pass

                return

            with self._resource_guard_lock:
                self.last_resource_snapshot = (
                    snapshot
                )

            guard_state = (
                policy.evaluate(
                    snapshot
                )
            )

            if guard_state in {
                RESOURCE_STATE_ABORT,
                RESOURCE_STATE_ABORT_EMERGENCY,
            }:
                # If the command has already completed, the response
                # wins. Do not retroactively kill a completed STEP.
                if operation_done_event.is_set():
                    return

                record = (
                    RLServerResourceAbortRecord(
                        phase=str(
                            phase
                        ),
                        guard_state=str(
                            guard_state
                        ),
                        snapshot=snapshot,
                    )
                )

                with self._resource_guard_lock:
                    self.last_resource_guard_record = (
                        record
                    )

                # Deliberately kill ONLY the current C++ server.
                # The surrounding Python formal trainer remains alive.
                if self.process.poll() is None:
                    try:
                        self.process.kill()
                    except ProcessLookupError:
                        pass

                return

            if stop_event.wait(
                self.resource_guard_sample_interval_seconds
            ):
                return


    def _finalize_eval_swap_guard_worker(
        self,
        *,
        stop_event,
        operation_done_event,
    ):
        """
        Protect FINALIZE_EVAL with one independent system-swap
        survival fuse.

        This deliberately does NOT reuse the STEP ResourceGuard
        policy. FINALIZE_EVAL may legitimately exceed the STEP
        10/12 GiB thresholds.

        The only abort condition here is:

            system SwapUsed >= finalize_eval_swap_abort_bytes

        The Python trainer is never killed by this worker.
        """

        cap = (
            self.finalize_eval_swap_abort_bytes
        )

        if cap is None:
            return

        while not stop_event.is_set():
            if (
                self.process.poll()
                is not None
            ):
                return

            try:
                snapshot = (
                    self._resource_snapshot_reader(
                        cpp_pid=
                            self.process.pid
                    )
                )

            except Exception as exc:
                # Same fail-closed infrastructure behavior as the
                # STEP watchdog: an unavailable monitor must not leave
                # an expensive C++ operation running unprotected.
                with self._resource_guard_lock:
                    self.last_resource_guard_monitor_error = (
                        exc
                    )

                if (
                    not operation_done_event.is_set()
                    and
                    self.process.poll()
                    is None
                ):
                    try:
                        self.process.kill()
                    except ProcessLookupError:
                        pass

                return

            with self._resource_guard_lock:
                self.last_resource_snapshot = (
                    snapshot
                )

            if (
                int(
                    snapshot.swap_used_bytes
                )
                >=
                int(
                    cap
                )
            ):
                if operation_done_event.is_set():
                    return

                record = (
                    RLServerResourceAbortRecord(
                        phase=
                            "FINALIZE_EVAL",

                        guard_state=
                            FINALIZE_EVAL_SWAP_CAP_GUARD_STATE,

                        snapshot=
                            snapshot,
                    )
                )

                with self._resource_guard_lock:
                    self.last_resource_guard_record = (
                        record
                    )

                if self.process.poll() is None:
                    try:
                        self.process.kill()
                    except ProcessLookupError:
                        pass

                return

            if stop_event.wait(
                self.resource_guard_sample_interval_seconds
            ):
                return


    def _read_until_with_finalize_eval_swap_guard(
        self,
        *,
        prefix,
        phase,
    ):
        """
        Read one FINALIZE_EVAL response while enforcing only the
        dedicated absolute system-swap hard cap.
        """

        if (
            self.finalize_eval_swap_abort_bytes
            is None
        ):
            return self._read_until(
                prefix,
                phase=phase,
            )

        with self._resource_guard_lock:
            self.last_resource_guard_record = None
            self.last_resource_guard_monitor_error = None

        stop_event = (
            threading.Event()
        )

        operation_done_event = (
            threading.Event()
        )

        thread = threading.Thread(
            target=
                self._finalize_eval_swap_guard_worker,

            kwargs={
                "stop_event":
                    stop_event,

                "operation_done_event":
                    operation_done_event,
            },

            name=(
                "loopycuts-finalize-eval-swap-guard"
            ),

            daemon=True,
        )

        thread.start()

        try:
            lines = self._read_until(
                prefix,
                phase=phase,
            )

            operation_done_event.set()

            return lines

        finally:
            stop_event.set()

            thread.join(
                timeout=max(
                    2.0,
                    (
                        2.0
                        *
                        self.resource_guard_sample_interval_seconds
                    ),
                )
            )


    def _read_until_with_resource_guard(
        self,
        *,
        prefix,
        phase,
    ):
        """
        Resource-guarded version of _read_until().

        Currently used for the real STEP blocking interval.

        Existing behavior is unchanged when no ResourceGuard policy
        is supplied.
        """

        if (
            self.resource_guard_policy
            is None
        ):
            return self._read_until(
                prefix,
                phase=phase,
            )

        with self._resource_guard_lock:
            self.last_resource_guard_record = None
            self.last_resource_guard_monitor_error = None

        stop_event = (
            threading.Event()
        )

        operation_done_event = (
            threading.Event()
        )

        thread = threading.Thread(
            target=
                self._resource_guard_worker,

            kwargs={
                "phase":
                    phase,

                "stop_event":
                    stop_event,

                "operation_done_event":
                    operation_done_event,
            },

            name=(
                "loopycuts-resource-guard"
            ),

            daemon=True,
        )

        thread.start()

        try:
            lines = self._read_until(
                prefix,
                phase=phase,
            )

            # Mark successful completion before asking the watchdog
            # to stop. This prevents a late resource sample from
            # killing a STEP whose complete response was already read.
            operation_done_event.set()

            return lines

        finally:
            stop_event.set()

            thread.join(
                timeout=max(
                    2.0,
                    (
                        2.0
                        *
                        self.resource_guard_sample_interval_seconds
                    ),
                )
            )


    @staticmethod
    def _parse_key_values(line):
        result = {}

        tokens = line.strip().split()

        for token in tokens:
            if "=" not in token:
                continue

            key, value = token.split(
                "=",
                1,
            )

            try:
                if (
                    "." in value
                    or "e" in value.lower()
                ):
                    result[key] = float(value)
                else:
                    result[key] = int(value)

            except ValueError:
                result[key] = value

        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_actions(line):
        tokens = line.strip().split()

        #
        # Expected:
        #
        # [RL] ACTIONS 0 1 2 ...
        #
        if len(tokens) <= 2:
            return []

        return [
            int(x)
            for x in tokens[2:]
        ]

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_id_list(line):
        tokens = line.strip().split()

        #
        # Expected examples:
        #
        # [RL] USED 0 3 8
        # [RL] REVERTED 22
        # [RL] NICO_BUG
        # [RL] TOP_RELEVANT 5 9
        #
        if len(tokens) <= 2:
            return []

        return [
            int(x)
            for x in tokens[2:]
        ]

    # ------------------------------------------------------------------

    @staticmethod
    def _find_server_errors(lines):
        return [
            line
            for line in lines
            if line.startswith("[RL] ERROR")
        ]

    # ------------------------------------------------------------------

    def _get_return_code_after_eof(self):
        """
        Obtain the child return code after stdout reaches EOF.

        Usually poll() is already sufficient. The short wait
        handles the small race where stdout closes immediately
        before the process status becomes visible.
        """

        return_code = self.process.poll()

        if return_code is None:
            try:
                return_code = (
                    self.process.wait(
                        timeout=1
                    )
                )
            except subprocess.TimeoutExpired:
                return_code = (
                    self.process.poll()
                )

        return return_code

    # ------------------------------------------------------------------

    def _read_until(
        self,
        prefix,
        phase,
    ):
        lines = []

        while True:
            line = (
                self.process.stdout.readline()
            )

            if line == "":
                return_code = (
                    self._get_return_code_after_eof()
                )

                self.last_response_lines = list(
                    lines
                )

                with self._resource_guard_lock:
                    guard_error = (
                        self.last_resource_guard_monitor_error
                    )

                    guard_record = (
                        self.last_resource_guard_record
                    )

                if guard_error is not None:
                    raise RLServerResourceGuardError(
                        "ResourceGuard monitor failed during "
                        f"{phase}: {guard_error!r}"
                    ) from guard_error

                if guard_record is not None:
                    raise RLServerResourceAbort(
                        phase=
                            guard_record.phase,

                        guard_state=
                            guard_record.guard_state,

                        snapshot=
                            guard_record.snapshot,

                        return_code=
                            return_code,
                    )

                raise RLServerProcessError(
                    phase=phase,
                    return_code=return_code,
                    expected_prefix=prefix,
                    lines=lines,
                )

            line = line.rstrip("\n")

            if self.echo_logs:
                print(line)

            lines.append(line)

            if line.startswith(prefix):
                break

        self.last_response_lines = list(
            lines
        )

        return lines

    # ------------------------------------------------------------------

    def _send(
        self,
        command,
        phase,
    ):
        return_code = self.process.poll()

        if return_code is not None:
            raise RLServerProcessError(
                phase=phase,
                return_code=return_code,
                expected_prefix=None,
                lines=[],
            )

        try:
            self.process.stdin.write(
                command + "\n"
            )

            self.process.stdin.flush()

        except BrokenPipeError as exc:
            return_code = (
                self._get_return_code_after_eof()
            )

            raise RLServerProcessError(
                phase=phase,
                return_code=return_code,
                expected_prefix=None,
                lines=[],
            ) from exc

    # ------------------------------------------------------------------

    def _update_from_lines(
        self,
        lines,
    ):
        for line in lines:
            if line.startswith(
                "[RL] STATE"
            ):
                self.state = (
                    self._parse_key_values(
                        line
                    )
                )

            elif line.startswith(
                "[RL] USED"
            ):
                self.used = (
                    self._parse_id_list(
                        line
                    )
                )

            elif line.startswith(
                "[RL] REVERTED"
            ):
                self.reverted = (
                    self._parse_id_list(
                        line
                    )
                )

            elif line.startswith(
                "[RL] NICO_BUG"
            ):
                self.nico_bug = (
                    self._parse_id_list(
                        line
                    )
                )

            elif line.startswith(
                "[RL] TOP_RELEVANT"
            ):
                self.top_relevant = (
                    self._parse_id_list(
                        line
                    )
                )

            elif line.startswith(
                "[RL] ACTIONS"
            ):
                self.actions = (
                    self._parse_actions(
                        line
                    )
                )

    # ------------------------------------------------------------------

    def get_state(self):
        self._send(
            "STATE",
            phase="STATE",
        )

        lines = self._read_until(
            "[RL] ACTIONS",
            phase="STATE",
        )

        errors = (
            self._find_server_errors(
                lines
            )
        )

        if errors:
            raise RLServerProtocolError(
                errors[-1]
            )

        self._update_from_lines(
            lines
        )

        return (
            self.state,
            self.actions,
        )

    # ------------------------------------------------------------------

    def step(
        self,
        loop_id,
    ):
        self._send(
            f"STEP {int(loop_id)}",
            phase="STEP",
        )

        lines = self._read_until_with_resource_guard(
            prefix="[RL] ACTIONS",
            phase="STEP",
        )

        #
        # Catch server-side protocol errors.
        #
        errors = (
            self._find_server_errors(
                lines
            )
        )

        if errors:
            raise RLServerProtocolError(
                errors[-1]
            )

        step_result = None

        for line in lines:
            if line.startswith(
                "[RL] STEP_RESULT"
            ):
                step_result = (
                    self._parse_key_values(
                        line
                    )
                )

        if step_result is None:
            raise RLServerProtocolError(
                "STEP did not produce "
                "[RL] STEP_RESULT"
            )

        self._update_from_lines(
            lines
        )

        return (
            step_result,
            self.state,
            self.actions,
        )

    # ------------------------------------------------------------------

    def finalize(
        self,
        output_dir,
    ):
        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        #
        # Current C++ protocol uses whitespace
        # tokenization, so output paths containing
        # spaces are not supported.
        #
        if " " in str(output_dir):
            raise ValueError(
                "FINALIZE output directory "
                "must not contain spaces"
            )

        self._send(
            f"FINALIZE {output_dir}",
            phase="FINALIZE",
        )

        #
        # If the C++ process aborts inside
        # finalization, _read_until() raises
        # RLServerProcessError containing the
        # partial finalization log.
        #
        lines = self._read_until(
            "[RL] ACTIONS",
            phase="FINALIZE",
        )

        errors = (
            self._find_server_errors(
                lines
            )
        )

        if errors:
            raise RLServerProtocolError(
                errors[-1]
            )

        final_result = None

        for line in lines:
            if line.startswith(
                "[RL] FINAL_RESULT"
            ):
                final_result = (
                    self._parse_key_values(
                        line
                    )
                )

        if final_result is None:
            raise RLServerProtocolError(
                "FINALIZE did not produce "
                "[RL] FINAL_RESULT"
            )

        self._update_from_lines(
            lines
        )

        return (
            final_result,
            self.state,
        )

    # ------------------------------------------------------------------

    def finalize_eval(
        self,
    ):
        """
        Execute the complete LoopyCuts finalization pipeline
        without writing intermediate/final mesh files.

        Geometry semantics must remain identical to FINALIZE.

        In particular, a full-hex result still executes
        poly_fix_orientation() on the C++ side.
        """

        self._send(
            "FINALIZE_EVAL",
            phase="FINALIZE_EVAL",
        )

        #
        # Exactly like FINALIZE, successful finalization finishes by
        # emitting the updated STATE/status/ACTIONS block.
        #
        # If C++ aborts anywhere inside finalization,
        # _read_until() raises RLServerProcessError.
        #
        lines = (
            self._read_until_with_finalize_eval_swap_guard(
                prefix="[RL] ACTIONS",
                phase="FINALIZE_EVAL",
            )
        )

        errors = (
            self._find_server_errors(
                lines
            )
        )

        if errors:
            raise RLServerProtocolError(
                errors[-1]
            )

        final_result = None

        for line in lines:
            if line.startswith(
                "[RL] FINAL_RESULT"
            ):
                final_result = (
                    self._parse_key_values(
                        line
                    )
                )

        if final_result is None:
            raise RLServerProtocolError(
                "FINALIZE_EVAL did not produce "
                "[RL] FINAL_RESULT"
            )

        self._update_from_lines(
            lines
        )

        return (
            final_result,
            self.state,
        )

    # ------------------------------------------------------------------

    def quit(self):
        if self.process.poll() is not None:
            return

        self._send(
            "QUIT",
            phase="QUIT",
        )

        self._read_until(
            "[RL] BYE",
            phase="QUIT",
        )

        self.process.wait(
            timeout=10
        )

    # ------------------------------------------------------------------

    def close(self):
        if self.process.poll() is None:
            try:
                self.quit()

            except Exception:
                #
                # Cleanup only.
                #
                # Broad exception handling is acceptable
                # here because close() must not leave a
                # child process behind.
                #
                if self.process.poll() is None:
                    self.process.terminate()

                    try:
                        self.process.wait(
                            timeout=5
                        )

                    except subprocess.TimeoutExpired:
                        self.process.kill()

                        self.process.wait()

    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    # ------------------------------------------------------------------

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()