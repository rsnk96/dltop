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

    def fake_local(text: str, mime: str = "text/plain") -> bool:
        del mime
        local.append(text)
        return True

    monkeypatch.setattr(clipboard, "_local_copy", fake_local)
    clipboard.copy(app, "hello")
    assert app.osc52 == ["hello"]  # OSC 52 always attempted (works over SSH)
    assert local == ["hello"]  # local fallback also attempted


def test_copy_sets_html_flavor_when_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _FakeApp()
    seen: list[tuple[str, str]] = []

    def fake_local(text: str, mime: str = "text/plain") -> bool:
        seen.append((text, mime))
        return mime == "text/html"  # pretend the html flavor lands

    monkeypatch.setattr(clipboard, "_local_copy", fake_local)
    rich = clipboard.copy(app, "PLAIN", html="<table></table>")
    assert rich is True
    assert app.osc52 == ["PLAIN"]  # OSC 52 still carries the plain form (SSH)
    # html flavor set locally; plain local copy skipped because rich succeeded
    assert seen == [("<table></table>", "text/html")]


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


def test_local_copy_html_skips_incapable_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    # xsel is "installed" but can't set text/html, so it must be skipped for xclip.
    monkeypatch.setattr(clipboard.shutil, "which", lambda exe: exe if exe in {"xsel", "xclip"} else None)
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], *, input: bytes, **_kw: object) -> _Result:  # noqa: A002
        del input
        seen.append(cmd)
        return _Result(0)

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)
    assert clipboard._local_copy("<table></table>", mime="text/html") is True
    assert seen == [["xclip", "-selection", "clipboard", "-t", "text/html"]]


def test_local_copy_returns_false_when_no_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clipboard.shutil, "which", lambda _exe: None)
    assert clipboard._local_copy("data") is False
