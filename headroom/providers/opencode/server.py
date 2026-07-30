"""Lifecycle management for a shared ``opencode serve`` process.

Why this exists
---------------
``headroom wrap opencode`` launches OpenCode's TUI, and OpenCode creates one
full *instance* per launch: its own plugin set, its own MCP child processes,
and — critically — its own in-memory turn state. Two wraps in the same project
therefore cannot see each other's live generation: the "assistant is thinking"
state lives only in the process that issued the prompt, while the SQLite
database that both processes share carries the messages but no "running" flag.
A second tab consequently shows an in-flight turn as finished.

OpenCode's own answer to this is server mode: one ``opencode serve`` process
owns the state, and every ``opencode attach`` is a thin remote TUI. Verified
behaviour of that mode (measured, not assumed):

* One server serves *many* directories. ``?directory=`` selects the instance;
  the launch directory is only the default.
* Sessions and the event bus are isolated per directory — a mutation in project
  A never reaches a subscriber scoped to project B.
* Within one directory every client shares: a second attach receives the live
  ``message.part.delta`` stream and sees ``/session/status`` report ``busy``.
* MCP servers start per directory, lazily on the first chat turn, as children
  of the single server process. A second client in the same directory reuses
  them instead of spawning its own.

The last point is where the resource win comes from: cost becomes a function of
how many *projects* are open, not how many tabs.

Design notes
------------
The Headroom plugin must live in the **server's** environment, not the
client's: ``attach`` is only a remote TUI, so the server is the process that
talks to the model. This was verified by running a server with
``HEADROOM_PROXY_URL`` set and confirming the plugin rewrote provider base URLs
for two different directories and that requests arrived at the proxy.

Startup is serialized through a lock file because two ``opencode serve``
processes racing on a fresh database make one of them exit with
``database is locked``.
"""

from __future__ import annotations

import errno
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from headroom import paths as _paths

__all__ = [
    "DEFAULT_OPENCODE_SERVER_PORT",
    "OpencodeServerHandle",
    "attach_command",
    "ensure_shared_server",
    "live_server_clients",
    "register_server_client",
    "server_is_reachable",
    "unregister_server_client",
]

# OpenCode's own preferred port for ``serve``/``attach``. ``serve --port 0``
# (its default) binds an ephemeral port, which cannot be attached to by a
# stable URL, so wrap always pins a port.
DEFAULT_OPENCODE_SERVER_PORT = 4096

_READY_TIMEOUT_SECONDS = 60.0
_READY_POLL_SECONDS = 0.25
_LOCK_TIMEOUT_SECONDS = 90.0
_LOCK_POLL_SECONDS = 0.2
_PROBE_TIMEOUT_SECONDS = 1.0


class OpencodeServerHandle:
    """The shared server this wrap is attached to.

    ``process`` is set only when *this* wrap spawned the server; when an
    existing one was reused it stays ``None`` so teardown never kills a server
    another wrap owns.
    """

    def __init__(self, port: int, process: subprocess.Popen[bytes] | None, spawned: bool) -> None:
        self.port = port
        self.process = process
        self.spawned = spawned

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"OpencodeServerHandle(port={self.port}, spawned={self.spawned})"


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------


def server_is_reachable(port: int, *, timeout: float = _PROBE_TIMEOUT_SECONDS) -> bool:
    """Return True when something accepts TCP connections on ``port``."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(("127.0.0.1", port))
            return True
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False


def _port_is_bindable(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))
    except OSError as exc:
        if exc.errno in (errno.EADDRINUSE, errno.EACCES):
            return False
        raise
    except OverflowError:
        return False
    return True


# ---------------------------------------------------------------------------
# Client refcount (same marker scheme the proxy uses)
# ---------------------------------------------------------------------------


def _marker_path(port: int) -> Path:
    directory = _paths.opencode_server_clients_dir(port)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{os.getpid()}.json"


def _proc_identity(pid: int) -> tuple[str, float] | None:
    """Best-effort ``(source, start_time)`` for ``pid``, to defeat PID reuse.

    Returns ``None`` when the start time cannot be determined, in which case
    callers fall back to existence-only liveness.
    """
    try:
        import psutil  # type: ignore[import-untyped]

        return ("psutil", psutil.Process(pid).create_time())
    except Exception:
        pass
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            fields = handle.read().rpartition(b")")[2].split()
        return ("proc", float(fields[19]))
    except (OSError, IndexError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _marker_is_stale(marker: Path, payload: dict[str, Any], pid: int) -> bool:
    if not _pid_alive(pid):
        return True
    recorded_src = payload.get("start_src")
    recorded_time = payload.get("start_time")
    if not isinstance(recorded_src, str) or not isinstance(recorded_time, (int, float)):
        return False
    identity = _proc_identity(pid)
    if identity is None or identity[0] != recorded_src:
        return False
    return abs(identity[1] - float(recorded_time)) > 1e-6


def register_server_client(port: int) -> None:
    """Record this process as an attached client of the shared server."""
    payload: dict[str, Any] = {"pid": os.getpid(), "started_at": time.time()}
    identity = _proc_identity(os.getpid())
    if identity is not None:
        payload["start_src"], payload["start_time"] = identity
    try:
        _marker_path(port).write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        # Best effort: a missing marker only costs us refcount accuracy, and
        # liveness pruning in live_server_clients is the real safety net.
        pass


def unregister_server_client(port: int) -> None:
    """Remove this process's marker (idempotent)."""
    try:
        _marker_path(port).unlink(missing_ok=True)
    except OSError:
        pass


def live_server_clients(port: int, *, exclude_self: bool = True) -> list[int]:
    """Live attached-client PIDs for ``port``, pruning stale markers."""
    directory = _paths.opencode_server_clients_dir(port)
    if not directory.exists():
        return []
    me = os.getpid()
    live: list[int] = []
    for marker in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        pid = payload.get("pid")
        if not isinstance(pid, int):
            try:
                pid = int(marker.stem)
            except ValueError:
                continue
        if _marker_is_stale(marker, payload, pid):
            try:
                marker.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        if exclude_self and pid == me:
            continue
        live.append(pid)
    return live


# ---------------------------------------------------------------------------
# Startup serialization
# ---------------------------------------------------------------------------


class _StartupLock:
    """Cooperative lock so only one wrap probes-then-spawns at a time.

    Uses ``O_EXCL`` file creation (portable, no fcntl) with a PID payload so a
    lock orphaned by a crashed wrap can be reclaimed instead of deadlocking.
    """

    def __init__(self, port: int) -> None:
        self._path = _paths.opencode_server_lock_path(port)
        self._acquired = False

    def __enter__(self) -> _StartupLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if self._reclaim_if_orphaned():
                    continue
                if time.time() >= deadline:
                    # Proceeding without the lock is better than failing the
                    # launch: the worst case is the race this lock avoids.
                    return self
                time.sleep(_LOCK_POLL_SECONDS)
                continue
            except OSError:
                return self
            with os.fdopen(fd, "w") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "at": time.time()}))
            self._acquired = True
            return self

    def __exit__(self, *_exc: object) -> None:
        if not self._acquired:
            return
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass

    def _reclaim_if_orphaned(self) -> bool:
        """Delete the lock when its owner is gone. Returns True if reclaimed."""
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        pid = payload.get("pid")
        if isinstance(pid, int) and _pid_alive(pid) and pid != os.getpid():
            return False
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            return False
        return True


# ---------------------------------------------------------------------------
# Spawning
# ---------------------------------------------------------------------------


def _spawn(
    binary: str,
    port: int,
    env: dict[str, str],
    cwd: Path,
) -> subprocess.Popen[bytes]:
    log_path = _paths.opencode_server_log_path(port)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "ab", buffering=0)  # noqa: SIM115 - owned by the child
    handle.write(
        f"\n=== headroom: starting shared opencode server on port {port} "
        f"at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode()
    )
    return subprocess.Popen(  # noqa: S603 - binary resolved via shutil.which by the caller
        [binary, "serve", "--port", str(port), "--hostname", "127.0.0.1"],
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        cwd=str(cwd),
        start_new_session=os.name == "posix",
    )


def _wait_until_ready(
    port: int,
    process: subprocess.Popen[bytes],
    *,
    timeout: float = _READY_TIMEOUT_SECONDS,
) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        if server_is_reachable(port):
            return True
        time.sleep(_READY_POLL_SECONDS)
    return False


def ensure_shared_server(
    *,
    binary: str,
    port: int,
    env: dict[str, str],
    cwd: Path,
) -> OpencodeServerHandle:
    """Return a handle to a live shared server on ``port``, starting one if needed.

    An already-running server is reused as-is; its environment (and therefore
    its Headroom wiring) is whatever the wrap that started it supplied. We
    deliberately do not restart a reachable server to "fix" its environment:
    other wraps may be attached to it, and killing their server mid-session is
    worse than the mismatch. Callers that need different wiring should use a
    different port.

    Raises:
        RuntimeError: when the server could not be started or the port is held
            by something that never becomes reachable.
    """
    if server_is_reachable(port):
        return OpencodeServerHandle(port, None, spawned=False)

    with _StartupLock(port):
        # Re-probe inside the lock: another wrap may have started the server
        # while we waited for it.
        if server_is_reachable(port):
            return OpencodeServerHandle(port, None, spawned=False)

        if not _port_is_bindable(port):
            raise RuntimeError(
                f"port {port} is in use by a process that is not answering as an "
                "OpenCode server; pass --server-port to pick another port"
            )

        process = _spawn(binary, port, env, cwd)
        if not _wait_until_ready(port, process):
            log_path = _paths.opencode_server_log_path(port)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise RuntimeError(
                f"shared OpenCode server on port {port} did not become ready; see {log_path}"
            )
        return OpencodeServerHandle(port, process, spawned=True)


def attach_command(server_url: str, directory: Path, extra_args: tuple[str, ...] = ()) -> list[str]:
    """Build the ``opencode attach`` argv for ``directory``.

    ``--dir`` is what scopes the remote TUI to a project: the server keeps one
    instance per directory, so this is the difference between sharing a session
    with another tab and being isolated from it.
    """
    return ["attach", server_url, "--dir", str(directory), *extra_args]
