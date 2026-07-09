"""Copy text to the user's clipboard, including a rich ``text/html`` flavor.

Three concerns are juggled here:

* **Plain text over SSH** -- Textual's :meth:`App.copy_to_clipboard` emits OSC 52,
  which reaches the *local* terminal's clipboard even when dltop runs on a
  remote box. OSC 52 carries a single plain-text blob only: it cannot advertise
  a ``text/html`` MIME flavor, so rich paste is impossible over SSH.

* **Rich HTML locally** -- a "web table" is only rendered by Teams / email /
  Word when the clipboard exposes a ``text/html`` target. Local clipboard tools
  (``wl-copy --type``/``xclip -t``) can set that target, so running locally the
  table pastes as a real table. ``xsel``/``pbcopy`` can't set HTML and are
  skipped for that flavor.

* **Graceful fallback** -- if no HTML-capable tool is available (or we're over
  SSH), the caller's plain-text form is used instead, so a "web table" copy
  still lands as readable text rather than raw ``<table>`` markup.

Note: a single ``wl-copy``/``xclip`` invocation advertises only the one MIME
target it was given, so after a successful rich copy the local clipboard holds
``text/html`` *only* -- pasting into a strict plain-text consumer yields
nothing. That's an accepted tradeoff for the web-table button (its job is rich
paste; the Markdown/TSV buttons cover plain text).
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.app import App

# Preference order; first tool that exists and succeeds wins.
_TOOLS = ("wl-copy", "xclip", "xsel", "pbcopy")


def copy(app: App, text: str, *, html: str | None = None) -> bool:
    """Copy ``text`` to the clipboard, exposing ``html`` as a rich flavor if given.

    Returns ``True`` if a ``text/html`` flavor was placed on the local clipboard
    (so a paste into Teams/email/Word renders a real table); ``False`` if only
    the plain-text form is available -- e.g. over SSH, where OSC 52 can't send
    MIME flavors, or when no HTML-capable clipboard tool is present.
    """
    with contextlib.suppress(Exception):
        app.copy_to_clipboard(text)  # OSC 52: plain text to the local terminal, SSH-safe
    rich = _local_copy(html, mime="text/html") if html is not None else False
    if not rich:
        _local_copy(text)
    return rich


def _tool_command(tool: str, mime: str) -> list[str] | None:
    """Build the argv for ``tool`` to set ``mime``, or None if it can't set it."""
    html = mime != "text/plain"
    if tool == "wl-copy":
        return ["wl-copy", "--type", mime] if html else ["wl-copy"]
    if tool == "xclip":
        base = ["xclip", "-selection", "clipboard"]
        return [*base, "-t", mime] if html else base
    if tool == "xsel":  # no MIME-target support
        return None if html else ["xsel", "--clipboard", "--input"]
    if tool == "pbcopy":  # plain text only
        return None if html else ["pbcopy"]
    return None


def _local_copy(text: str, mime: str = "text/plain") -> bool:
    """Pipe ``text`` to the first local clipboard tool that can set ``mime`` and succeeds.

    A tool being installed doesn't mean it works here -- e.g. ``wl-copy`` is
    present but exits non-zero on an X11 session with no Wayland server. Only a
    zero exit counts as success, so we fall through to the next capable tool.
    """
    data = text.encode("utf-8")
    for tool in _TOOLS:
        if shutil.which(tool) is None:
            continue
        cmd = _tool_command(tool, mime)
        if cmd is None:
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
