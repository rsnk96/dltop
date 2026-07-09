"""Clipboard copy: OSC 52 path plus local-tool fallback selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dltop import clipboard

if TYPE_CHECKING:
    import pytest


class _FakeApp:
    def __init__(self) -> None:
        self.osc52: list[str] = []

    def copy_to_clipboard(self, text: str) -> None:
        self.osc52.append(text)


def test_copy_emits_osc52_and_tries_local(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _FakeApp()
    local: list[str] = []
    monkeypatch.setattr(clipboard, "_local_copy", local.append)
    clipboard.copy(app, "hello")
    assert app.osc52 == ["hello"]  # OSC 52 always attempted (works over SSH)
    assert local == ["hello"]  # local fallback also attempted


def test_copy_survives_osc52_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def copy_to_clipboard(self, text: str) -> None:
            raise RuntimeError(text)

    monkeypatch.setattr(clipboard, "_local_copy", lambda _text: True)
    clipboard.copy(_Boom(), "x")  # must not raise


class _Result:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def test_local_copy_picks_first_available_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], bytes]] = []
    # Only xclip is "installed".
    monkeypatch.setattr(clipboard.shutil, "which", lambda exe: exe if exe == "xclip" else None)

    def fake_run(cmd: list[str], *, input: bytes, **_kw: object) -> _Result:  # noqa: A002
        calls.append((cmd, input))
        return _Result(0)

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)
    assert clipboard._local_copy("data") is True
    assert calls == [(["xclip", "-selection", "clipboard"], b"data")]


def test_local_copy_falls_through_when_a_tool_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both wl-copy and xclip "installed", but wl-copy fails (e.g. no Wayland).
    monkeypatch.setattr(clipboard.shutil, "which", lambda exe: exe if exe in {"wl-copy", "xclip"} else None)
    tried: list[str] = []

    def fake_run(cmd: list[str], *, input: bytes, **_kw: object) -> _Result:  # noqa: A002
        del input
        tried.append(cmd[0])
        return _Result(1 if cmd[0] == "wl-copy" else 0)

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)
    assert clipboard._local_copy("data") is True
    assert tried == ["wl-copy", "xclip"]  # fell through the failed wl-copy


def test_local_copy_returns_false_when_no_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clipboard.shutil, "which", lambda _exe: None)
    assert clipboard._local_copy("data") is False
