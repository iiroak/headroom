"""Tests for the shared ``opencode serve`` lifecycle helper.

These cover the bookkeeping that decides when a shared server is started,
reused, and torn down. The measured behaviour that motivates the module (one
server serving many directories, per-directory isolation, shared live state
within a directory) belongs to OpenCode itself and is documented in
``headroom/providers/opencode/server.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from headroom.providers.opencode import server as server_mod


@pytest.fixture(autouse=True)
def _isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "hr"))


class _FakeProcess:
    def __init__(self, *, alive: bool = True) -> None:
        self._alive = alive
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return None if self._alive else 0

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True
        self._alive = False


# ---------------------------------------------------------------------------
# attach_command
# ---------------------------------------------------------------------------


def test_attach_command_scopes_to_directory() -> None:
    """``--dir`` is what binds the remote client to one project's instance."""
    argv = server_mod.attach_command("http://127.0.0.1:4096", Path("/repo/app"))
    assert argv == ["attach", "http://127.0.0.1:4096", "--dir", "/repo/app"]


def test_attach_command_appends_passthrough_args() -> None:
    argv = server_mod.attach_command(
        "http://127.0.0.1:4096", Path("/repo/app"), ("--continue", "--mini")
    )
    assert argv[-2:] == ["--continue", "--mini"]


# ---------------------------------------------------------------------------
# Client refcount
# ---------------------------------------------------------------------------


def test_register_then_unregister_round_trips() -> None:
    port = 4096
    server_mod.register_server_client(port)
    assert server_mod.live_server_clients(port, exclude_self=False) == [os.getpid()]

    server_mod.unregister_server_client(port)
    assert server_mod.live_server_clients(port, exclude_self=False) == []


def test_unregister_is_idempotent() -> None:
    server_mod.unregister_server_client(4096)
    server_mod.unregister_server_client(4096)


def test_live_clients_excludes_self_by_default() -> None:
    port = 4096
    server_mod.register_server_client(port)
    assert server_mod.live_server_clients(port) == []


def test_live_clients_is_empty_when_no_directory_exists() -> None:
    assert server_mod.live_server_clients(65000) == []


def test_live_clients_prunes_dead_pids(tmp_path: Path) -> None:
    """A marker whose process is gone must not keep a server alive forever."""
    port = 4096
    server_mod.register_server_client(port)
    directory = tmp_path / "hr" / "opencode-server-clients" / str(port)
    stale = directory / "999999.json"
    stale.write_text(json.dumps({"pid": 999999}), encoding="utf-8")

    server_mod.live_server_clients(port)

    assert not stale.exists()
    assert (directory / f"{os.getpid()}.json").exists()


def test_live_clients_prunes_recycled_pids(tmp_path: Path) -> None:
    """A live PID that is not the process which wrote the marker is stale.

    Without this, PID reuse would make a long-dead wrap look attached and the
    shared server would never be reclaimed.
    """
    port = 4096
    server_mod.register_server_client(port)
    marker = tmp_path / "hr" / "opencode-server-clients" / str(port) / f"{os.getpid()}.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if "start_src" not in payload:
        pytest.skip("process start time unavailable on this platform")
    payload["start_time"] = float(payload["start_time"]) + 5000.0
    marker.write_text(json.dumps(payload), encoding="utf-8")

    assert server_mod.live_server_clients(port, exclude_self=False) == []
    assert not marker.exists()


def test_live_clients_keeps_marker_without_start_time(tmp_path: Path) -> None:
    """Existence-only liveness when start time is unknown — no false pruning."""
    port = 4096
    directory = tmp_path / "hr" / "opencode-server-clients" / str(port)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / f"{os.getpid()}.json"
    marker.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")

    assert server_mod.live_server_clients(port, exclude_self=False) == [os.getpid()]
    assert marker.exists()


def test_live_clients_ignores_unreadable_marker(tmp_path: Path) -> None:
    """Corrupt JSON falls back to the filename's PID rather than crashing."""
    port = 4096
    directory = tmp_path / "hr" / "opencode-server-clients" / str(port)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{os.getpid()}.json").write_text("not json", encoding="utf-8")

    assert server_mod.live_server_clients(port, exclude_self=False) == [os.getpid()]


def test_client_markers_are_namespaced_per_port(tmp_path: Path) -> None:
    """Two shared servers must not see each other's clients."""
    server_mod.register_server_client(4096)
    assert server_mod.live_server_clients(4097, exclude_self=False) == []


# ---------------------------------------------------------------------------
# ensure_shared_server
# ---------------------------------------------------------------------------


def test_ensure_reuses_reachable_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reachable port is reused, and never restarted to fix its environment.

    Restarting would kill the server other tabs are attached to, which is worse
    than an environment mismatch; callers needing different wiring use a
    different port.
    """
    monkeypatch.setattr(server_mod, "server_is_reachable", lambda port, **_kw: True)
    monkeypatch.setattr(
        server_mod, "_spawn", lambda *a, **k: pytest.fail("must not spawn over a live server")
    )

    handle = server_mod.ensure_shared_server(
        binary="opencode", port=4096, env={}, cwd=Path("/repo")
    )

    assert handle.spawned is False
    assert handle.process is None
    assert handle.url == "http://127.0.0.1:4096"


def test_ensure_spawns_when_port_is_free(monkeypatch: pytest.MonkeyPatch) -> None:
    reachable = {"value": False}
    spawn_calls: list[dict[str, object]] = []
    process = _FakeProcess()

    monkeypatch.setattr(server_mod, "server_is_reachable", lambda port, **_kw: reachable["value"])
    monkeypatch.setattr(server_mod, "_port_is_bindable", lambda _port: True)

    def fake_spawn(binary: str, port: int, env: dict[str, str], cwd: Path):  # noqa: ANN202
        spawn_calls.append({"binary": binary, "port": port, "env": env, "cwd": cwd})
        reachable["value"] = True
        return process

    monkeypatch.setattr(server_mod, "_spawn", fake_spawn)

    handle = server_mod.ensure_shared_server(
        binary="opencode", port=4096, env={"HEADROOM_PROXY_URL": "http://x"}, cwd=Path("/repo")
    )

    assert handle.spawned is True
    assert handle.process is process
    assert spawn_calls[0]["env"] == {"HEADROOM_PROXY_URL": "http://x"}
    assert spawn_calls[0]["cwd"] == Path("/repo")


def test_ensure_reprobes_inside_the_startup_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server started by a racing wrap while we waited is reused, not duplicated.

    Concurrent ``opencode serve`` launches on a fresh database make one of them
    exit with ``database is locked``, so this second probe is what keeps a burst
    of tabs from each starting a server.
    """
    probes = {"count": 0}

    def fake_reachable(_port: int, **_kw: object) -> bool:
        probes["count"] += 1
        # False on the pre-lock probe, True once we hold the lock.
        return probes["count"] > 1

    monkeypatch.setattr(server_mod, "server_is_reachable", fake_reachable)
    monkeypatch.setattr(
        server_mod, "_spawn", lambda *a, **k: pytest.fail("must not spawn after re-probe")
    )

    handle = server_mod.ensure_shared_server(
        binary="opencode", port=4096, env={}, cwd=Path("/repo")
    )

    assert handle.spawned is False
    assert probes["count"] >= 2


def test_ensure_rejects_port_held_by_a_foreign_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unbindable, unreachable port is a clear error, not a silent hang."""
    monkeypatch.setattr(server_mod, "server_is_reachable", lambda port, **_kw: False)
    monkeypatch.setattr(server_mod, "_port_is_bindable", lambda _port: False)

    with pytest.raises(RuntimeError, match="not answering as an OpenCode server"):
        server_mod.ensure_shared_server(binary="opencode", port=4096, env={}, cwd=Path("/repo"))


def test_ensure_cleans_up_a_server_that_never_becomes_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that never listens is terminated instead of being left behind."""
    process = _FakeProcess()
    monkeypatch.setattr(server_mod, "server_is_reachable", lambda port, **_kw: False)
    monkeypatch.setattr(server_mod, "_port_is_bindable", lambda _port: True)
    monkeypatch.setattr(server_mod, "_spawn", lambda *a, **k: process)
    monkeypatch.setattr(server_mod, "_wait_until_ready", lambda *a, **k: False)

    with pytest.raises(RuntimeError, match="did not become ready"):
        server_mod.ensure_shared_server(binary="opencode", port=4096, env={}, cwd=Path("/repo"))

    assert process.terminated is True


# ---------------------------------------------------------------------------
# Startup lock
# ---------------------------------------------------------------------------


def test_startup_lock_releases_on_exit(tmp_path: Path) -> None:
    lock_path = tmp_path / "hr" / ".opencode_server_lock_4096"
    with server_mod._StartupLock(4096):
        assert lock_path.exists()
    assert not lock_path.exists()


def test_startup_lock_reclaims_an_orphaned_lock(tmp_path: Path) -> None:
    """A lock left by a crashed wrap must not deadlock every future launch."""
    lock_path = tmp_path / "hr" / ".opencode_server_lock_4096"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"pid": 999999, "at": 0}), encoding="utf-8")

    with server_mod._StartupLock(4096):
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()

    assert not lock_path.exists()


def test_startup_lock_does_not_hang_forever_on_a_live_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rather than fail the launch, we proceed once the wait budget is spent."""
    lock_path = tmp_path / "hr" / ".opencode_server_lock_4096"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # A live holder that is not us: our own PID would be reclaimed.
    monkeypatch.setattr(server_mod, "_pid_alive", lambda _pid: True)
    lock_path.write_text(json.dumps({"pid": 999999, "at": 0}), encoding="utf-8")
    monkeypatch.setattr(server_mod, "_LOCK_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(server_mod, "_LOCK_POLL_SECONDS", 0.0)

    with server_mod._StartupLock(4096) as lock:
        assert lock._acquired is False

    # Someone else's lock is left intact.
    assert lock_path.exists()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def test_server_client_dir_is_separate_from_proxy_clients(tmp_path: Path) -> None:
    """Server clients and proxy clients have independent lifetimes."""
    from headroom import paths

    assert paths.opencode_server_clients_dir(4096) != paths.proxy_clients_dir(4096)
