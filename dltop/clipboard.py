"""Copy text to the user's clipboard, including when dltop runs over SSH.

Two paths, tried together so at least one lands:

* **OSC 52** -- Textual's :meth:`App.copy_to_clipboard` emits the escape
  sequence that tells the *local* terminal to set its clipboard, so it works
  even when dltop runs on a remote box you're SSH'd into. This is the only
  path that can reach your laptop's clipboard from a remote host, but the
  terminal has to honour OSC 52 (kitty, WezTerm, iTerm2, xterm with
  ``allowWindowOps``; GNOME/VTE historically ignore it).
* **Local clipboard tool** -- as a fallback for terminals that drop OSC 52,
  we also pipe the text to whatever clipboard command is on ``PATH``
  (``wl-copy`` / ``xclip`` / ``xsel`` / ``pbcopy``). Running locally this sets
  the real clipboard; over SSH it harmlessly targets the remote's (usually
  absent) clipboard, so it never makes things worse.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.app import App

# First entry whose executable exists on PATH wins. stdin carries the text.
_LOCAL_TOOLS: tuple[list[str], ...] = (
    ["wl-copy"],
    ["xclip", "-selection", "clipboard"],
    ["xsel", "--clipboard", "--input"],
    ["pbcopy"],
)


def copy(app: App, text: str) -> None:
    """Copy ``text`` via OSC 52 (SSH-safe) and any local clipboard tool."""
    with contextlib.suppress(Exception):
        app.copy_to_clipboard(text)
    _local_copy(text)


def _local_copy(text: str) -> bool:
    """Pipe ``text`` to the first local clipboard command that succeeds.

    A tool being installed doesn't mean it works here -- e.g. ``wl-copy`` is
    present but exits non-zero on an X11 session with no Wayland server. Only a
    zero exit counts as success, so we fall through to the next tool (``xclip``)
    instead of stopping at the first one that merely exists.
    """
    data = text.encode("utf-8")
    for cmd in _LOCAL_TOOLS:
        if shutil.which(cmd[0]) is None:
            continue
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            result = subprocess.run(  # noqa: S603
                cmd,
                input=data,
                timeout=2,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                return True
    return False
