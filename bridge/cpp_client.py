import signal
import subprocess
from pathlib import Path


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


class LoopyCutsClient:
    def __init__(
        self,
        executable,
        mesh_file,
        loop_file,
        echo_logs=False,
    ):
        self.executable = str(executable)
        self.mesh_file = str(mesh_file)
        self.loop_file = str(loop_file)
        self.echo_logs = echo_logs

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

        lines = self._read_until(
            "[RL] ACTIONS",
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
        lines = self._read_until(
            "[RL] ACTIONS",
            phase="FINALIZE_EVAL",
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